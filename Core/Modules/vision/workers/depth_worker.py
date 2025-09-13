"""
Depth Worker - Fast Reflex Loop with Screen Capture
High-frequency vision processing for real-time obstacle avoidance
"""

import time
import threading
import numpy as np
import cv2
import pyautogui
from typing import Optional, Dict, Any, Tuple
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import from migrated locations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'depth_adapter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from depth_adapter.depth_anything_v2_onnx import DepthAnythingV2ONNX
from state_estimator import StateEstimator

# Integration state bus - migrated to Engine
engine_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Engine')
sys.path.insert(0, engine_path)

try:
    from state_bus import init_state_bus, publish_vision_state
    STATE_BUS_AVAILABLE = True
except ImportError:
    print("WARNING: Integration state_bus not available")
    STATE_BUS_AVAILABLE = False
    def init_state_bus(*args, **kwargs):
        pass
    def publish_vision_state(*args, **kwargs):
        pass

class DepthWorker:
    """
    High-frequency depth processing worker
    Captures screen, runs depth estimation, computes navigation state
    """
    
    def __init__(self, 
                 get_frame_rgb = None,
                 capture_mode: str = "screen",
                 screen_region: Optional[Tuple[int, int, int, int]] = None,
                 depth_fps: float = 3.0,
                 target_fps: float = 15.0,
                 short_side: int = 640,
                 ema_alpha: float = 0.6):
        """
        Initialize depth worker
        
        Args:
            get_frame_rgb: Callable that returns RGB frame [H, W, 3] or None for fallback
            capture_mode: "screen" or "video" (fallback mode)
            screen_region: (x, y, width, height) or None for full screen (fallback mode)
            depth_fps: Depth estimation frequency (Hz)
            target_fps: Target frame processing frequency (Hz) 
            short_side: Resize short side for depth processing
            ema_alpha: EMA smoothing factor for depth maps
        """
        self.get_frame_rgb = get_frame_rgb
        self.capture_mode = capture_mode
        self.screen_region = screen_region
        self.depth_fps = depth_fps
        self.target_fps = target_fps
        self.short_side = short_side
        self.ema_alpha = ema_alpha
        
        # Processing components
        self.depth_model = DepthAnythingV2ONNX(short_side=short_side)
        self.state_estimator = StateEstimator(fov_deg=120.0)
        
        # Screen capture - using PyAutoGUI instead of MSS
        self.screen_bbox = None
        
        # Processing state
        self.running = False
        self.worker_thread = None
        
        # Frame buffers
        self.current_depth = None
        self.prev_gray = None
        self.depth_update_counter = 0
        
        # Timing control
        self.last_depth_time = 0
        self.last_frame_time = 0
        
        # Performance stats
        self.frame_count = 0
        self.depth_count = 0
        self.start_time = time.time()
        
    def initialize(self) -> bool:
        """
        Initialize all components
        
        Returns:
            True if successful
        """
        print("INITIALIZING: Depth Worker...")
        
        # Initialize state bus
        if not init_state_bus(use_zmq=False):  # In-process for now
            print("FAILED: Failed to initialize state bus")
            return False
        
        # Load depth model
        if not self.depth_model.load_model():
            print("FAILED: Failed to load depth model")
            return False
        
        # Initialize screen capture using PyAutoGUI
        if self.capture_mode == "screen":
            try:
                # Disable PyAutoGUI failsafe for automated capture
                pyautogui.FAILSAFE = False
                
                if self.screen_region:
                    x, y, w, h = self.screen_region
                    self.screen_bbox = (x, y, w, h)
                else:
                    # Use full screen
                    screen_size = pyautogui.size()
                    self.screen_bbox = (0, 0, screen_size.width, screen_size.height)
                
                print(f"SUCCESS: PyAutoGUI screen capture initialized: {self.screen_bbox}")
                
            except Exception as e:
                print(f"FAILED: Failed to initialize PyAutoGUI screen capture: {e}")
                return False
        
        print("SUCCESS: Depth Worker initialization complete")
        return True
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture frame using callback or fallback to PyAutoGUI
        
        Returns:
            BGR frame [H, W, 3] or None if failed
        """
        # Try callback first (unified capture pipeline)
        if self.get_frame_rgb is not None:
            try:
                frame_rgb = self.get_frame_rgb()
                if frame_rgb is not None:
                    # Convert RGB to BGR (callback returns RGB, OpenCV expects BGR)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    return frame_bgr
            except Exception as e:
                print(f"Warning: Frame callback failed: {e}")
        
        # Fallback to PyAutoGUI screen capture
        if self.capture_mode == "screen":
            try:
                # Capture screen using PyAutoGUI
                if self.screen_bbox:
                    x, y, w, h = self.screen_bbox
                    screenshot = pyautogui.screenshot(region=(x, y, w, h))
                else:
                    screenshot = pyautogui.screenshot()
                
                # Convert PIL Image to numpy array
                frame = np.array(screenshot)
                
                # Convert RGB to BGR (PyAutoGUI returns RGB, OpenCV expects BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return frame
                
            except Exception as e:
                print(f"Warning: PyAutoGUI screen capture failed: {e}")
                return None
        
        # TODO: Add video file capture support
        return None
    
    def _process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process frame to extract navigation state
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            State dictionary or None if failed
        """
        try:
            # Convert to grayscale for optical flow
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Check if depth update is needed
            current_time = time.time()
            depth_interval = 1.0 / self.depth_fps
            need_depth_update = (current_time - self.last_depth_time) >= depth_interval
            
            # Run depth estimation if needed
            if need_depth_update or self.current_depth is None:
                print(f"DEBUG: Running depth estimation on frame shape {frame_bgr.shape}")
                raw_depth = self.depth_model.predict_depth(frame_bgr)
                
                if raw_depth is not None:
                    print(f"DEBUG: Depth prediction SUCCESS - shape {raw_depth.shape}")
                    # Apply EMA smoothing if we have previous depth
                    if self.current_depth is not None:
                        self.current_depth = (self.ema_alpha * raw_depth + 
                                            (1 - self.ema_alpha) * self.current_depth)
                    else:
                        self.current_depth = raw_depth.copy()
                    
                    self.last_depth_time = current_time
                    self.depth_count += 1
                else:
                    print("ERROR: Depth prediction FAILED - depth_model.predict_depth() returned None")
                    if self.current_depth is None:
                        print("ERROR: No current_depth available - returning None from _process_frame")
                        return None
            
            # Compute navigation state
            if self.current_depth is not None:
                print(f"DEBUG: Computing navigation state with depth shape {self.current_depth.shape}")
                state = self.state_estimator.compute_state(
                    self.current_depth, 
                    self.prev_gray, 
                    curr_gray
                )
                
                if state is not None:
                    print(f"DEBUG: State computation SUCCESS - keys: {list(state.keys())}")
                else:
                    print("ERROR: State computation FAILED - state_estimator returned None")
                
                # Update previous frame
                self.prev_gray = curr_gray.copy()
                
                return state
            else:
                print("ERROR: No current_depth available - cannot compute state")
            
            return None
            
        except Exception as e:
            print(f"Warning: Frame processing failed: {e}")
            return None
    
    def _worker_loop(self):
        """Main worker loop"""
        print(f"STARTING: Depth worker loop (target {self.target_fps} FPS, depth {self.depth_fps} FPS)")
        
        frame_interval = 1.0 / self.target_fps
        
        while self.running:
            loop_start = time.time()
            
            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                time.sleep(0.01)  # Brief pause on capture failure
                continue
            
            # Process frame
            state = self._process_frame(frame)
            
            if state is not None:
                # Publish state
                print(f"DEBUG: Publishing vision state to state bus")
                publish_vision_state(**state)
                
                self.frame_count += 1
                print(f"DEBUG: Published frame {self.frame_count} - front_m={state.get('front_m', 'N/A')}")
                
                # Print periodic stats
                if self.frame_count % (self.target_fps * 5) == 0:  # Every 5 seconds
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    avg_depth_fps = self.depth_count / elapsed
                    
                    print(f"STATS: Depth Worker Stats: {avg_fps:.1f} FPS, {avg_depth_fps:.1f} depth/s, "
                          f"front={state['front_m']:.1f}m, ttc={state['ttc_s']:.1f}s")
            else:
                print("DEBUG: No state to publish - _process_frame returned None")
            
            # Timing control
            loop_time = time.time() - loop_start
            sleep_time = max(0, frame_interval - loop_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("STOPPED: Depth worker loop stopped")
    
    def start(self) -> bool:
        """
        Start depth worker in background thread
        
        Returns:
            True if started successfully
        """
        if self.running:
            print("Warning: Depth worker already running")
            return True
        
        if not self.initialize():
            return False
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.depth_count = 0
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print("SUCCESS: Depth worker started")
        return True
    
    def stop(self):
        """Stop depth worker"""
        if not self.running:
            return
        
        print("STOPPING: Depth worker...")
        self.running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        # Cleanup - PyAutoGUI doesn't need explicit cleanup
        pass
        
        print("SUCCESS: Depth worker stopped")
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_s': elapsed,
            'frame_count': self.frame_count,
            'depth_count': self.depth_count,
            'avg_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'avg_depth_fps': self.depth_count / elapsed if elapsed > 0 else 0
        }

# Global worker instance
_depth_worker = None

def start_depth_worker(get_frame_rgb = None,
                      capture_mode: str = "screen", 
                      screen_region: Optional[Tuple[int, int, int, int]] = None,
                      depth_fps: float = 3.0,
                      target_fps: float = 15.0) -> bool:
    """
    Start global depth worker instance
    
    Args:
        get_frame_rgb: Callable that returns RGB frame [H, W, 3] for unified capture
        capture_mode: "screen" or "video" (fallback)
        screen_region: (x, y, width, height) or None (fallback)
        depth_fps: Depth estimation frequency
        target_fps: Frame processing frequency
        
    Returns:
        True if started successfully
    """
    global _depth_worker
    
    if _depth_worker is not None:
        print("Warning: Depth worker already exists, stopping previous instance")
        _depth_worker.stop()
    
    _depth_worker = DepthWorker(
        get_frame_rgb=get_frame_rgb,
        capture_mode=capture_mode,
        screen_region=screen_region,
        depth_fps=depth_fps,
        target_fps=target_fps
    )
    
    return _depth_worker.start()

def stop_depth_worker():
    """Stop global depth worker"""
    global _depth_worker
    
    if _depth_worker is not None:
        _depth_worker.stop()
        _depth_worker = None

def get_depth_worker_stats() -> Optional[Dict[str, float]]:
    """Get depth worker statistics"""
    if _depth_worker is not None:
        return _depth_worker.get_stats()
    return None

def main():
    """CLI entry point for testing"""
    parser = argparse.ArgumentParser(description="Depth Worker - Fast Vision Processing")
    parser.add_argument("--capture", choices=["screen", "video"], default="screen",
                       help="Capture mode")
    parser.add_argument("--fps", type=float, default=15.0,
                       help="Target processing FPS")
    parser.add_argument("--depth-fps", type=float, default=3.0,
                       help="Depth estimation FPS")
    parser.add_argument("--region", type=str, 
                       help="Screen region as 'x,y,w,h' (e.g., '100,100,800,600')")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview window")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = None
    if args.region:
        try:
            x, y, w, h = map(int, args.region.split(','))
            screen_region = (x, y, w, h)
        except ValueError:
            print("Error: Invalid region format. Use 'x,y,w,h'")
            return 1
    
    # Start worker
    if not start_depth_worker(
        capture_mode=args.capture,
        screen_region=screen_region, 
        depth_fps=args.depth_fps,
        target_fps=args.fps
    ):
        print("FAILED: Failed to start depth worker")
        return 1
    
    try:
        # Run for specified duration
        print(f"RUNNING: Running for {args.duration} seconds... Press Ctrl+C to stop early")
        
        start_time = time.time()
        while time.time() - start_time < args.duration:
            time.sleep(1)
            
            # Print stats every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                stats = get_depth_worker_stats()
                if stats:
                    print(f"STATS: {stats['avg_fps']:.1f} FPS, {stats['avg_depth_fps']:.1f} depth/s")
    
    except KeyboardInterrupt:
        print("\nINTERRUPTED: By user")
    
    finally:
        stop_depth_worker()
        
        # Final stats
        stats = get_depth_worker_stats()
        if stats:
            print(f"FINAL STATS:")
            print(f"   Duration: {stats['elapsed_s']:.1f}s")
            print(f"   Frames: {stats['frame_count']}")
            print(f"   Depth updates: {stats['depth_count']}")
            print(f"   Avg FPS: {stats['avg_fps']:.1f}")
            print(f"   Avg depth FPS: {stats['avg_depth_fps']:.1f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())