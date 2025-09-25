"""
Florence Worker - ROI-Based Florence Analysis System  
Event-driven Florence-2 analysis with intelligent ROI selection and escalation
"""

import time
import threading
import numpy as np
import cv2
import pyautogui
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import state bus from Engine  
engine_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Engine')
sys.path.insert(0, engine_path)
from state_bus import get_vision_state, publish_vision_facts

# Import event router from local workers
from .event_router import get_next_inspection_event

class FlorenceWorker:
    """
    ROI-based Florence analysis worker
    Processes inspection events with intelligent ROI selection and escalation
    """
    
    def __init__(self,
                 roi_width_ratio: float = 0.35,        # ROI width as fraction of frame
                 min_confidence: float = 0.7,          # Confidence threshold for ROI results
                 max_facts: int = 3,                   # Max facts to keep
                 fact_expiry: float = 5.0,             # Fact expiry time (seconds)
                 fov_deg: float = 120.0):              # Field of view
        """
        Initialize Florence worker - event-driven ROI-first analysis
        
        Args:
            roi_width_ratio: ROI width as fraction of frame width
            min_confidence: Minimum confidence for ROI results (unused in ROI-first mode)
            max_facts: Maximum facts to keep in memory
            fact_expiry: Time before facts expire
            fov_deg: Horizontal field of view
        """
        self.roi_width_ratio = roi_width_ratio
        self.min_confidence = min_confidence  # Kept for potential future use
        self.max_facts = max_facts
        self.fact_expiry = fact_expiry
        self.fov_deg = fov_deg
        
        # Florence analyzer (imported from existing system)
        self.florence_analyzer = None
        self.screen_capture = None
        
        # Processing state
        self.running = False
        self.worker_thread = None
        
        # Analysis tracking (no rate limiting)
        self.analysis_count = 0
        
        # Fact memory
        self.current_facts = []
        self.fact_lock = threading.Lock()
        
        # Performance stats
        self.roi_analyses = 0
        self.full_frame_analyses = 0
        self.escalations = 0
        self.start_time = time.time()
        
    def initialize(self) -> bool:
        """
        Initialize Florence analyzer and screen capture
        
        Returns:
            True if successful
        """
        print("INITIALIZING: Florence Worker...")
        
        try:
            # Import and initialize Florence analyzer
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FusionScripts', 'QwenScripts'))
            from florence_analyzer import FlorenceAnalyzer
            
            self.florence_analyzer = FlorenceAnalyzer()
            print("SUCCESS: Florence analyzer loaded")
            
            # Initialize PyAutoGUI screen capture
            pyautogui.FAILSAFE = False
            print("SUCCESS: PyAutoGUI screen capture initialized")
            
            return True
            
        except Exception as e:
            print(f"FAILED: Failed to initialize Florence worker: {e}")
            return False
    
    def _capture_current_frame(self) -> Optional[np.ndarray]:
        """
        Capture current screen frame
        
        Returns:
            BGR frame [H, W, 3] or None if failed
        """
        try:
            # Capture full screen using PyAutoGUI
            screenshot = pyautogui.screenshot()
            
            # Convert PIL Image to numpy array and BGR
            frame = np.array(screenshot)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            return frame
            
        except Exception as e:
            print(f"Warning: Frame capture failed: {e}")
            return None
    
    def _bearing_to_roi(self, frame: np.ndarray, bearing_deg: float) -> np.ndarray:
        """
        Extract ROI from frame based on bearing
        
        Args:
            frame: Full frame [H, W, 3]
            bearing_deg: Target bearing in degrees
            
        Returns:
            ROI crop [H, W, 3]
        """
        height, width = frame.shape[:2]
        
        # Convert bearing to pixel coordinates
        center_x = width / 2
        pixels_per_degree = width / self.fov_deg
        bearing_offset = bearing_deg * pixels_per_degree
        roi_center_x = center_x + bearing_offset
        
        # Calculate ROI bounds
        roi_width = int(width * self.roi_width_ratio)
        roi_left = max(0, int(roi_center_x - roi_width / 2))
        roi_right = min(width, int(roi_center_x + roi_width / 2))
        
        # Extract tall, narrow ROI (full height)
        roi = frame[:, roi_left:roi_right]
        
        return roi
    
    def _analyze_roi(self, roi: np.ndarray, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze ROI with Florence
        
        Args:
            roi: ROI image [H, W, 3]
            event: Event that triggered analysis
            
        Returns:
            List of detected facts
        """
        if self.florence_analyzer is None or roi.size == 0:
            return []
        
        try:
            # Use correct captures directory path
            captures_dir = "FusionCore/FusionScripts/captures"
            os.makedirs(captures_dir, exist_ok=True)
            # Save ROI temporarily for analysis
            timestamp = int(time.time() * 1000)
            roi_path = f"{captures_dir}/roi_temp_{timestamp}.png"
            
            success = cv2.imwrite(roi_path, roi)
            if not success or not os.path.exists(roi_path):
                print(f"ERROR: Failed to save ROI file at {roi_path}")
                return []
            
            # Run Florence analysis
            result = self.florence_analyzer.analyze_image(roi_path)
            
            # Clean up temp file
            try:
                os.remove(roi_path)
            except:
                pass
            
            # Extract facts from result
            facts = self._extract_facts_from_result(result, event)
            self.roi_analyses += 1
            
            return facts
            
        except Exception as e:
            print(f"Warning: ROI analysis failed: {e}")
            return []
    
    def _analyze_full_frame(self, frame: np.ndarray, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze full frame with Florence (escalation)
        
        Args:
            frame: Full frame [H, W, 3] 
            event: Event that triggered analysis
            
        Returns:
            List of detected facts
        """
        if self.florence_analyzer is None:
            return []
        
        try:
            # Use correct captures directory path
            captures_dir = "FusionCore/FusionScripts/captures"
            os.makedirs(captures_dir, exist_ok=True)
            
            # Save frame temporarily
            timestamp = int(time.time() * 1000)
            frame_path = f"{captures_dir}/full_temp_{timestamp}.png"
            cv2.imwrite(frame_path, frame)
            
            # Run Florence analysis
            result = self.florence_analyzer.analyze_image(frame_path)
            
            # Clean up temp file
            try:
                os.remove(frame_path)
            except:
                pass
            
            # Extract facts
            facts = self._extract_facts_from_result(result, event)
            self.full_frame_analyses += 1
            
            return facts
            
        except Exception as e:
            print(f"Warning: Full frame analysis failed: {e}")
            return []
    
    def _extract_facts_from_result(self, result: Dict[str, Any], event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract facts from Florence analysis result
        
        Args:
            result: Florence analysis result
            event: Triggering event
            
        Returns:
            List of fact dictionaries
        """
        facts = []
        
        try:
            # Debug: Log result structure
            if not result:
                print(f"Warning: Empty result from Florence for {event.get('type', 'unknown')}")
                return []
            
            analysis = result.get('analysis', {})
            
            # Extract from detailed caption
            detailed_caption = analysis.get('<DETAILED_CAPTION>', '')
            
            # Log analysis keys for verification
            if len(analysis.keys()) > 0:
                print(f"Florence analysis complete: {len(analysis.keys())} components")
            
            if detailed_caption and len(detailed_caption) > 10:
                # Simple fact extraction (can be improved)
                confidence = 0.8  # Base confidence
                
                # Detect common objects/people
                labels = []
                if 'person' in detailed_caption.lower():
                    labels.append('person')
                if 'chair' in detailed_caption.lower():
                    labels.append('chair')
                if 'door' in detailed_caption.lower():
                    labels.append('door')
                if 'wall' in detailed_caption.lower():
                    labels.append('wall')
                
                # Create facts for detected labels
                for label in labels:
                    fact = {
                        'label': label,
                        'bearing_deg': float(event.get('bearing', 0.0)),
                        'conf': confidence,
                        'dist_m': float(event.get('distance', 2.0)) if event.get('distance') else 2.0,
                        'moving': event.get('flow_magnitude', 0) > 0.02,
                        'timestamp': time.time(),
                        'source': 'florence_roi' if 'roi' in str(result) else 'florence_full'
                    }
                    facts.append(fact)
            
            # If no specific objects found, create generic fact
            if len(facts) == 0 and event.get('type') in ['obstacle_detected', 'motion_detected']:
                fact = {
                    'label': 'obstacle',
                    'bearing_deg': float(event.get('bearing', 0.0)),
                    'conf': 0.6,
                    'dist_m': float(event.get('distance', 1.0)) if event.get('distance') else 1.0,
                    'moving': event.get('flow_magnitude', 0) > 0.02,
                    'timestamp': time.time(),
                    'source': 'depth_based'
                }
                facts.append(fact)
                
        except Exception as e:
            print(f"Warning: Fact extraction failed: {e}")
        
        return facts
    
    def _should_escalate(self, roi_facts: List[Dict[str, Any]], event: Dict[str, Any]) -> bool:
        """
        Determine if ROI result should escalate to full-frame analysis
        Pure event-driven logic - no time-based limits
        
        Args:
            roi_facts: Facts from ROI analysis
            event: Triggering event
            
        Returns:
            True if should escalate
        """
        # Never escalate for now - ROI-first approach per original suggestion
        # ROI analysis should be sufficient for obstacle/motion detection
        # Full-frame escalation disabled to maintain speed focus
        
        # Future: Could escalate only on genuine analysis failure
        # if len(roi_facts) == 0 and roi_analysis_genuinely_failed:
        #     return True
        
        return False
    
    def _update_facts(self, new_facts: List[Dict[str, Any]]) -> None:
        """
        Update fact memory with new facts
        
        Args:
            new_facts: List of new facts to add
        """
        with self.fact_lock:
            current_time = time.time()
            
            # Remove expired facts
            self.current_facts = [
                fact for fact in self.current_facts
                if current_time - fact.get('timestamp', 0) < self.fact_expiry
            ]
            
            # Add new facts
            self.current_facts.extend(new_facts)
            
            # Keep only most recent/important facts
            self.current_facts.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            self.current_facts = self.current_facts[:self.max_facts]
            
            # Publish updated facts
            publish_vision_facts(self.current_facts)
    
    def _process_inspection_event(self, event: Dict[str, Any]) -> None:
        """
        Process single inspection event
        
        Args:
            event: Event to process
        """
        try:
            # Capture current frame
            frame = self._capture_current_frame()
            if frame is None:
                print(f"WARNING: Frame capture failed for {event.get('type', 'unknown')}")
                return
            
            event_type = event.get('type', 'unknown')
            bearing = event.get('bearing', 0.0)
            
            print(f"PROCESSING: {event_type} at {bearing:.1f}°")
            
            # Extract ROI and analyze
            roi = self._bearing_to_roi(frame, bearing)
            roi_facts = self._analyze_roi(roi, event)
            
            # Check if escalation needed
            if self._should_escalate(roi_facts, event):
                print(f"ESCALATING: {event_type} to full frame")
                full_facts = self._analyze_full_frame(frame, event)
                final_facts = full_facts if len(full_facts) > 0 else roi_facts
                self.escalations += 1
            else:
                final_facts = roi_facts
            
            # Update fact memory
            if len(final_facts) > 0:
                self._update_facts(final_facts)
                print(f"SUCCESS: Added {len(final_facts)} facts from {event_type}")
            
        except Exception as e:
            print(f"Warning: Event processing failed: {e}")
    
    def _worker_loop(self):
        """Main Florence worker loop"""
        print("STARTED: Florence worker loop started")
        
        while self.running:
            try:
                # Get next inspection event (blocking with longer timeout for Florence)
                event = get_next_inspection_event(timeout=10.0)
                
                if event is not None:
                    self._process_inspection_event(event)
                    self.analysis_count += 1
                    
                    # Periodic stats
                    if self.analysis_count % 10 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.analysis_count / elapsed if elapsed > 0 else 0
                        print(f"STATS: Florence: {self.analysis_count} analyses ({rate:.2f}/s), "
                              f"{self.roi_analyses} ROI, {self.full_frame_analyses} full, "
                              f"{self.escalations} escalations")
                
            except Exception as e:
                print(f"Warning: Florence worker error: {e}")
                time.sleep(0.1)
        
        print("STOPPED: Florence worker loop stopped")
    
    def start(self) -> bool:
        """Start Florence worker thread"""
        if self.running:
            return True
        
        if not self.initialize():
            return False
        
        self.running = True
        self.start_time = time.time()
        self.analysis_count = 0
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=False)
        self.worker_thread.start()
        
        print("SUCCESS: Florence worker started")
        return True
    
    def stop(self):
        """Stop Florence worker"""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        if self.screen_capture:
            self.screen_capture.close()
        
        print("SUCCESS: Florence worker stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_s': elapsed,
            'total_analyses': self.analysis_count,
            'roi_analyses': self.roi_analyses,
            'full_frame_analyses': self.full_frame_analyses,
            'escalations': self.escalations,
            'avg_rate': self.analysis_count / elapsed if elapsed > 0 else 0,
            'current_facts': len(self.current_facts),
            'escalation_rate': self.escalations / max(1, self.analysis_count)
        }

# Global worker instance
_florence_worker = None

def start_florence_worker(**kwargs) -> bool:
    """Start global Florence worker"""
    global _florence_worker
    
    if _florence_worker is not None:
        _florence_worker.stop()
    
    _florence_worker = FlorenceWorker(**kwargs)
    return _florence_worker.start()

def stop_florence_worker():
    """Stop global Florence worker"""
    global _florence_worker
    
    if _florence_worker is not None:
        _florence_worker.stop()
        _florence_worker = None

def get_florence_worker_stats() -> Optional[Dict[str, Any]]:
    """Get Florence worker statistics"""
    if _florence_worker is not None:
        return _florence_worker.get_stats()
    return None

def test_florence_worker():
    """Test Florence worker (requires Florence analyzer)"""
    print("TESTING: Florence Worker...")
    
    # This test requires the actual Florence system to be available
    try:
        if not start_florence_worker():
            print("FAILED: Failed to start Florence worker")
            return False
        
        print("SUCCESS: Florence worker started successfully")
        
        time.sleep(2.0)  # Let it run briefly
        
        stats = get_florence_worker_stats()
        if stats:
            print(f"STATS: {stats}")
        
        stop_florence_worker()
        print("SUCCESS: Florence worker test complete")
        return True
        
    except Exception as e:
        print(f"WARNING: Florence worker test failed (expected if Florence not available): {e}")
        return False

if __name__ == "__main__":
    test_florence_worker()