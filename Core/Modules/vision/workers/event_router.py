"""
Event Router - Event-Driven Florence Triggering
Analyzes depth/flow state to determine when Florence analysis is needed
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import queue
import sys
import os

# Add project root to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import state bus from Engine
engine_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Engine')
sys.path.insert(0, engine_path)
from state_bus import get_vision_state, publish_message

class EventRouter:
    """
    Routes vision state changes to appropriate Florence analysis requests
    Implements intelligent triggering to minimize unnecessary analysis
    """
    
    def __init__(self,
                 near_threshold: float = 0.8,     # Near obstacle threshold (m)
                 safe_threshold: float = 1.2,     # Safe clearance threshold (m)
                 flow_threshold: float = 0.02,    # Motion detection threshold
                 ssim_threshold: float = 0.20,    # Scene change threshold
                 persist_ticks: int = 4,          # Ticks before persistent event
                 fov_deg: float = 120.0):         # Field of view
        """
        Initialize event router
        
        Args:
            near_threshold: Distance considered "near obstacle"
            safe_threshold: Distance considered "safe clearance"
            flow_threshold: Optical flow magnitude for motion detection
            ssim_threshold: SSIM change threshold for scene novelty
            persist_ticks: Ticks before marking obstacle as persistent
            fov_deg: Horizontal field of view
        """
        self.near_threshold = near_threshold
        self.safe_threshold = safe_threshold
        self.flow_threshold = flow_threshold
        self.ssim_threshold = ssim_threshold
        self.persist_ticks = persist_ticks
        self.fov_deg = fov_deg
        
        # Event tracking
        self.obstacle_history = {}  # bearing -> tick_count
        self.last_ssim = None
        self.event_counts = {
            'obstacle_detected': 0,
            'motion_detected': 0, 
            'walkway_detected': 0,
            'scene_change': 0,
            'obstacle_persistent': 0
        }
        
        # Priority queue for inspection requests
        self.inspection_queue = queue.PriorityQueue(maxsize=10)
        
        # Worker thread
        self.running = False
        self.worker_thread = None
        
        # Timing
        self.last_process_time = 0
        self.tick_count = 0
        
    def _compute_ssim_delta(self, state: Dict[str, Any]) -> float:
        """
        Compute scene similarity change (mock implementation)
        
        Args:
            state: Vision state with clearance data
            
        Returns:
            SSIM delta [0-1], higher = more change
        """
        # TODO: Implement actual SSIM computation from frame data
        # For now, use clearance variance as proxy for scene change
        
        clearances = [item[1] for item in state.get('clearance_by_bearing', [])]
        if len(clearances) == 0:
            return 0.0
        
        # Use variance in clearances as scene change indicator
        variance = np.var(clearances) if len(clearances) > 1 else 0.0
        
        # Mock SSIM delta based on clearance variance
        ssim_delta = min(1.0, variance / 10.0)  # Normalize
        
        return float(ssim_delta)
    
    def _find_walkways(self, clearances: List[List[float]]) -> List[Tuple[float, float]]:
        """
        Find walkway regions in clearance data
        
        Args:
            clearances: List of [bearing_deg, clearance_m] pairs
            
        Returns:
            List of (center_bearing, width_deg) tuples for walkways
        """
        if len(clearances) == 0:
            return []
        
        walkways = []
        current_start = None
        current_width = 0
        
        for i, (bearing, clearance) in enumerate(clearances):
            is_safe = clearance >= self.safe_threshold
            
            if is_safe:
                if current_start is None:
                    current_start = bearing
                    current_width = 0
                current_width += 4.0  # 4° per bearing bin
            else:
                # End of safe region
                if current_start is not None and current_width >= 20.0:  # Min 20° width
                    center = current_start + current_width / 2
                    walkways.append((float(center), float(current_width)))
                
                current_start = None
                current_width = 0
        
        # Handle walkway at end
        if current_start is not None and current_width >= 20.0:
            center = current_start + current_width / 2
            walkways.append((float(center), float(current_width)))
        
        return walkways
    
    def _enqueue_event(self, event_type: str, priority: int, bearing: float, 
                      distance: Optional[float] = None, **kwargs):
        """
        Add event to inspection queue
        
        Args:
            event_type: Type of event
            priority: Priority (lower = higher priority)
            bearing: Bearing in degrees
            distance: Distance in meters
            **kwargs: Additional event data
        """
        event_data = {
            'type': event_type,
            'bearing': bearing,
            'distance': distance,
            'timestamp': time.time(),
            **kwargs
        }
        
        try:
            # Priority queue with timestamp and unique ID as tiebreaker
            priority_key = (priority, time.time(), id(event_data))
            self.inspection_queue.put_nowait((priority_key, event_data))
            self.event_counts[event_type] += 1
            
        except queue.Full:
            # Drop oldest event of same priority
            try:
                old_item = self.inspection_queue.get_nowait()
                # Create new priority key to avoid comparison issues
                new_priority_key = (priority, time.time(), id(event_data))
                self.inspection_queue.put_nowait((new_priority_key, event_data))
                print(f"Warning: Dropped old event {old_item[1]['type']} for new {event_type}")
            except queue.Empty:
                print(f"Warning: Failed to enqueue {event_type} event")
        except Exception as e:
            print(f"Warning: Priority queue error: {e}")
    
    def _process_vision_state(self, state: Dict[str, Any]) -> None:
        """
        Process vision state and generate events
        
        Args:
            state: Latest vision state from depth worker
        """
        clearances = state.get('clearance_by_bearing', [])
        flows = state.get('flow_by_bearing', [])
        
        if len(clearances) == 0:
            return
        
        # 1. Obstacle Detection - find nearest obstacle
        min_clearance = float('inf')
        obstacle_bearing = 0.0
        
        for bearing, clearance in clearances:
            if abs(bearing) <= 60.0 and clearance < min_clearance:  # Within ±60° FOV
                min_clearance = clearance
                obstacle_bearing = bearing
        
        if min_clearance < self.near_threshold:
            self._enqueue_event('obstacle_detected', priority=1, 
                              bearing=obstacle_bearing, distance=min_clearance)
        
        # 2. Motion Detection - moving objects
        for i, (bearing, clearance) in enumerate(clearances):
            if i < len(flows):
                _, flow_mag = flows[i]
                
                if clearance < 2.0 and flow_mag > self.flow_threshold:
                    self._enqueue_event('motion_detected', priority=1,
                                      bearing=bearing, distance=clearance,
                                      flow_magnitude=flow_mag)
        
        # 3. Walkway Detection - publish directly to Beta (no Florence needed)
        walkways = self._find_walkways(clearances)
        for center_bearing, width_deg in walkways:
            publish_message('vision.walkways', {
                'type': 'walkway_detected',
                'bearing': center_bearing,
                'width_deg': width_deg,
                'timestamp': time.time()
            })
            self.event_counts['walkway_detected'] += 1
        
        # 4. Scene Change Detection
        ssim_delta = self._compute_ssim_delta(state)
        if ssim_delta > self.ssim_threshold:
            self._enqueue_event('scene_change', priority=2,
                              bearing=0.0, ssim_delta=ssim_delta)
        
        # 5. Persistent Obstacle Detection
        current_obstacles = set()
        for bearing, clearance in clearances:
            if clearance < self.near_threshold:
                bearing_key = int(bearing / 5) * 5  # Round to 5° bins
                current_obstacles.add(bearing_key)
                
                if bearing_key in self.obstacle_history:
                    self.obstacle_history[bearing_key] += 1
                    
                    # Check if persistent
                    if self.obstacle_history[bearing_key] >= self.persist_ticks:
                        self._enqueue_event('obstacle_persistent', priority=1,
                                          bearing=float(bearing_key), distance=clearance,
                                          ticks=self.obstacle_history[bearing_key])
                        # Reset counter to avoid spam
                        self.obstacle_history[bearing_key] = self.persist_ticks - 1
                else:
                    self.obstacle_history[bearing_key] = 1
        
        # Clean up old obstacles
        for bearing_key in list(self.obstacle_history.keys()):
            if bearing_key not in current_obstacles:
                del self.obstacle_history[bearing_key]
    
    def _worker_loop(self):
        """Main processing loop"""
        print("STARTED: Event router worker started")
        
        while self.running:
            try:
                # Get latest vision state
                state = get_vision_state()
                
                if state is not None:
                    self._process_vision_state(state)
                    self.tick_count += 1
                    
                    # Periodic stats
                    if self.tick_count % 100 == 0:
                        print(f"STATS: Event Router: {self.tick_count} ticks, "
                              f"queue size: {self.inspection_queue.qsize()}, "
                              f"events: {sum(self.event_counts.values())}")
                
                time.sleep(0.1)  # 10 Hz processing
                
            except Exception as e:
                print(f"Warning: Event router error: {e}")
                time.sleep(0.1)
        
        print("STOPPED: Event router worker stopped")
    
    def start(self) -> bool:
        """Start event router worker thread"""
        if self.running:
            return True
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print("SUCCESS: Event router started")
        return True
    
    def stop(self):
        """Stop event router"""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        print("SUCCESS: Event router stopped")
    
    def get_next_event(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Get next inspection event from queue
        
        Args:
            timeout: Timeout in seconds (0 = non-blocking)
            
        Returns:
            Event dictionary or None
        """
        try:
            if timeout > 0:
                priority_key, event_data = self.inspection_queue.get(timeout=timeout)
            else:
                priority_key, event_data = self.inspection_queue.get_nowait()
            
            return event_data
            
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event router statistics"""
        return {
            'tick_count': self.tick_count,
            'queue_size': self.inspection_queue.qsize(),
            'event_counts': self.event_counts.copy(),
            'obstacle_history_size': len(self.obstacle_history),
            'total_events': sum(self.event_counts.values())
        }

# Global router instance
_event_router = None

def start_event_router(**kwargs) -> bool:
    """Start global event router"""
    global _event_router
    
    if _event_router is not None:
        _event_router.stop()
    
    _event_router = EventRouter(**kwargs)
    return _event_router.start()

def stop_event_router():
    """Stop global event router"""
    global _event_router
    
    if _event_router is not None:
        _event_router.stop()
        _event_router = None

def get_next_inspection_event(timeout: float = 0.0) -> Optional[Dict[str, Any]]:
    """Get next inspection event"""
    if _event_router is not None:
        return _event_router.get_next_event(timeout)
    return None

def get_event_router_stats() -> Optional[Dict[str, Any]]:
    """Get event router statistics"""
    if _event_router is not None:
        return _event_router.get_stats()
    return None

def test_event_router():
    """Test event router with synthetic data"""
    print("TESTING: Event Router...")
    
    # Start router
    if not start_event_router():
        print("FAILED: Failed to start event router")
        return False
    
    # Simulate some vision states
    from FusionCore.Integration.state_bus import init_state_bus, publish_vision_state
    
    init_state_bus(use_zmq=False)
    
    # State 1: Clear path
    publish_vision_state(
        front_m=5.0, left_m=3.0, right_m=4.0,
        edge_risk=0.1, tilt_deg=0.0, mean_flow=0.01, ttc_s=30.0,
        clearance_by_bearing=[[-30, 3.0], [0, 5.0], [30, 4.0]],
        flow_by_bearing=[[-30, 0.01], [0, 0.01], [30, 0.01]]
    )
    
    time.sleep(0.2)
    
    # State 2: Obstacle detected
    publish_vision_state(
        front_m=0.5, left_m=2.0, right_m=3.0,
        edge_risk=0.3, tilt_deg=1.0, mean_flow=0.05, ttc_s=2.0,
        clearance_by_bearing=[[-30, 2.0], [0, 0.5], [30, 3.0]],  
        flow_by_bearing=[[-30, 0.01], [0, 0.05], [30, 0.01]]
    )
    
    time.sleep(0.2)
    
    # Check for events
    event = get_next_inspection_event()
    if event:
        print(f"SUCCESS: Generated event: {event['type']} at {event['bearing']}°")
        
        stats = get_event_router_stats()
        if stats:
            print(f"STATS: {stats['total_events']} total events")
    else:
        print("FAILED: No events generated")
    
    stop_event_router()
    print("SUCCESS: Event router test complete")
    return True

if __name__ == "__main__":
    test_event_router()