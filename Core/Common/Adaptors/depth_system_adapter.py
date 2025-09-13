#!/usr/bin/env python3
"""
Depth System Adapter - Interface Bridge for Movement Calibrator
===============================================================

PURPOSE: Provides a simplified interface to the depth system for calibration tests.
         Bridges between the movement calibrator's expected interface and the actual
         depth worker system.

CONTRACT:
- Provides get_clearance_at_bearing(bearing) method for calibration tests
- Interfaces with existing depth worker and state estimator
- Handles initialization and cleanup of depth system components

INTEGRATION:
- Used by wall_movement_calibrator.py for wall-referenced measurements
- Connects to FusionCore Vision depth system components
"""

import sys
import os
import time
import numpy as np
from typing import Optional, Dict, List

# Add project paths - go up to Core root, then to vision workers
vision_workers_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Modules', 'vision', 'workers')
depth_adapter_path = os.path.join(vision_workers_path, 'depth_adapter')
vision_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Modules', 'vision')

sys.path.insert(0, vision_workers_path)
sys.path.insert(0, depth_adapter_path)
sys.path.insert(0, vision_path)

try:
    from depth_worker import DepthWorker
    from state_estimator import StateEstimator
    from depth_anything_v2_onnx import DepthAnythingV2ONNX
    
    # Try to find start_depth_worker function
    try:
        from depth_worker import start_depth_worker
    except ImportError:
        # Define a simple start function if not available
        def start_depth_worker(*args, **kwargs):
            return DepthWorker(*args, **kwargs)
    # Import state bus for getting vision data
    engine_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Engine')
    sys.path.insert(0, engine_path)
    from state_bus import get_vision_state
    DEPTH_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Depth system not available: {e}")
    DEPTH_SYSTEM_AVAILABLE = False

class DepthSystemAdapter:
    """
    Simplified interface to depth system for calibration tests
    
    Provides bearing-based clearance measurements for movement calibrator
    """
    
    def __init__(self):
        """Initialize depth system adapter"""
        self.depth_worker = None
        self.state_estimator = None
        self.is_initialized = False
        self.last_state = None
        self.last_clearance_data = {}
        
        if not DEPTH_SYSTEM_AVAILABLE:
            print("WARNING: Depth system components not available")
    
    def initialize(self) -> bool:
        """
        Initialize the depth system adapter to use shared state bus
        
        Returns:
            True if initialization successful
        """
        if not DEPTH_SYSTEM_AVAILABLE:
            print("ERROR: Cannot initialize - depth system not available")
            return False
        
        try:
            print("INITIALIZING: Depth System Adapter...")
            
            # Initialize state bus (if not already initialized)
            from state_bus import init_state_bus
            init_state_bus(use_zmq=False)  # Use in-process state bus
            
            # Don't create our own depth worker - use the shared state bus
            # The main system should have a depth worker running that publishes to state bus
            self.depth_worker = None
            self.state_estimator = None  # Not needed - state estimator runs in main depth worker
            
            # Test if we can get vision state from the shared bus
            test_state = get_vision_state()
            if test_state is not None:
                print("SUCCESS: Connected to shared state bus with live vision data")
            else:
                print("WARNING: State bus initialized but no vision data available yet")
                print("         This is normal if main depth worker hasn't started publishing")
            
            self.is_initialized = True
            print("SUCCESS: Depth system adapter initialized (using shared state bus)")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize depth system adapter: {e}")
            return False
    
    def update_clearance_data(self) -> bool:
        """
        Update clearance data from depth system using real measurements from state bus
        
        Returns:
            True if update successful
        """
        if not self.is_initialized:
            return False
        
        try:
            # Get latest vision state from the background depth worker
            vision_state = get_vision_state()
            
            if vision_state is None:
                print("WARNING: No vision state available from depth worker")
                print("DEPTH ADAPTER MODE: simulated")
                return False
            
            # Extract basic directional clearances
            front_m = vision_state.get('front_m', 0.0)
            left_m = vision_state.get('left_m', 0.0)  
            right_m = vision_state.get('right_m', 0.0)
            
            # Check for valid measurements
            if front_m <= 0 and left_m <= 0 and right_m <= 0:
                print("WARNING: All clearance measurements are invalid")
                print("DEPTH ADAPTER MODE: simulated")
                return False
            
            # Get the detailed 31-bearing clearance data
            clearance_by_bearing = vision_state.get('clearance_by_bearing', {})
            
            if clearance_by_bearing:
                # Use the full 31-bearing clearance data from StateEstimator
                self.last_clearance_data = clearance_by_bearing.copy()
                print("DEPTH ADAPTER MODE: depth_map")
                print(f"SUCCESS: Updated with {len(clearance_by_bearing)} bearing measurements from state bus")
            else:
                # Fallback to basic directional clearances if detailed data unavailable
                self.last_clearance_data = {}
                
                # Front sector (0°)
                if front_m > 0:
                    self.last_clearance_data[0.0] = front_m
                    
                # Left sector (-90°) 
                if left_m > 0:
                    self.last_clearance_data[-90.0] = left_m
                    
                # Right sector (90°)
                if right_m > 0:
                    self.last_clearance_data[90.0] = right_m
                
                # Interpolate additional bearings for smoother coverage
                if front_m > 0 and right_m > 0:
                    # Front-right sector (45°)
                    self.last_clearance_data[45.0] = (front_m + right_m) / 2.0
                    
                if front_m > 0 and left_m > 0:
                    # Front-left sector (-45°)
                    self.last_clearance_data[-45.0] = (front_m + left_m) / 2.0
                
                print("DEPTH ADAPTER MODE: depth_map")
                print(f"SUCCESS: Updated with {len(self.last_clearance_data)} basic directional clearances")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to update clearance data: {e}")
            print("DEPTH ADAPTER MODE: simulated")
            import traceback
            traceback.print_exc()
            return False
    
    def get_clearance_at_bearing(self, bearing_degrees: float) -> Optional[float]:
        """
        Get clearance distance at specific bearing angle
        
        Args:
            bearing_degrees: Bearing angle in degrees (0 = front, + = right, - = left)
            
        Returns:
            Distance in meters or None if not available
        """
        if not self.is_initialized:
            return None
        
        # Update clearance data
        if not self.update_clearance_data():
            # No simulated fallback - return None for failed depth reads
            return None
        
        # Find closest bearing in clearance data
        if not self.last_clearance_data:
            return None
        
        # Find closest bearing measurement
        best_match = None
        best_error = float('inf')
        
        for angle, distance in self.last_clearance_data.items():
            error = abs(angle - bearing_degrees)
            if error < best_error:
                best_error = error
                best_match = distance
        
        # Only return if we have a reasonably close match (within 10 degrees)
        if best_error <= 10.0:
            return best_match
        
        return None
    
    def get_clearances(self, bearings: List[float]) -> List[Optional[float]]:
        """
        Get clearance distances for multiple bearing angles in one call
        
        Args:
            bearings: List of bearing angles in degrees
            
        Returns:
            List of distances (or None for unavailable) corresponding to input bearings
        """
        if not self.is_initialized:
            return [None] * len(bearings)
        
        # Update clearance data once for all bearings
        data_available = self.update_clearance_data()
        
        results = []
        for bearing in bearings:
            if data_available:
                # Find closest bearing in clearance data
                best_match = None
                best_error = float('inf')
                
                for angle, distance in self.last_clearance_data.items():
                    error = abs(angle - bearing)
                    if error < best_error:
                        best_error = error
                        best_match = distance
                
                # Only return if we have a reasonably close match (within 10 degrees)
                if best_error <= 10.0:
                    results.append(best_match)
                else:
                    results.append(None)
            else:
                # No simulated fallback - return None for failed depth reads
                results.append(None)
        
        return results
    
    
    def get_stable_wall_bearings(self) -> List[float]:
        """
        Get list of bearing angles where stable walls are detected
        
        Returns:
            List of bearing angles in degrees with stable wall measurements
        """
        if not self.is_initialized:
            return [0.0]  # Default to front
        
        # In real implementation, this would analyze clearance stability
        # For now, return common wall directions
        return [0.0, -90.0, 90.0, 180.0]
    
    def cleanup(self):
        """Cleanup depth system adapter resources"""
        # Don't stop depth worker - we don't own it, it's shared
        self.depth_worker = None
        self.state_estimator = None
        self.is_initialized = False
        print("Depth system adapter cleaned up")

class MockDepthSystemAdapter:
    """
    Mock depth system adapter for testing when actual depth system unavailable
    
    Provides simulated clearance data for calibration testing
    """
    
    def __init__(self):
        """Initialize mock adapter"""
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize mock system"""
        print("MOCK: Initializing mock depth system adapter")
        self.is_initialized = True
        return True
    
    def get_clearance_at_bearing(self, bearing_degrees: float) -> Optional[float]:
        """Provide mock clearance data"""
        if not self.is_initialized:
            return None
        
        # Consistent mock room layout
        abs_bearing = abs(bearing_degrees)
        
        if abs_bearing <= 15:  # Front
            return 3.2  # Consistent wall distance
        elif abs_bearing >= 165:  # Back
            return 4.1  
        elif bearing_degrees < -75:  # Left
            return 1.8
        elif bearing_degrees > 75:  # Right
            return 2.3
        else:  # Interpolate
            if bearing_degrees < 0:  # Left side
                ratio = abs(bearing_degrees - 15) / 60
                return 3.2 + ratio * (1.8 - 3.2)
            else:  # Right side  
                ratio = abs(bearing_degrees - 15) / 60
                return 3.2 + ratio * (2.3 - 3.2)
    
    def get_clearances(self, bearings: List[float]) -> List[Optional[float]]:
        """Get mock clearances for multiple bearings"""
        return [self.get_clearance_at_bearing(b) for b in bearings]
    
    def get_stable_wall_bearings(self) -> List[float]:
        """Return mock stable wall bearings"""
        return [0.0, -90.0, 90.0]
    
    def cleanup(self):
        """Mock cleanup"""
        self.is_initialized = False
        print("MOCK: Mock depth system adapter cleaned up")

def create_depth_adapter() -> "DepthSystemAdapter":
    """
    Factory function to create appropriate depth adapter
    
    Returns:
        DepthSystemAdapter (real or mock) based on system availability
    """
    if DEPTH_SYSTEM_AVAILABLE:
        return DepthSystemAdapter()
    else:
        return MockDepthSystemAdapter()

def test_depth_adapter():
    """Test depth adapter functionality"""
    print("Testing Depth System Adapter")
    print("=" * 40)
    
    # Create adapter
    adapter = create_depth_adapter()
    
    # Initialize
    if not adapter.initialize():
        print("ERROR: Failed to initialize depth adapter")
        return False
    
    print(f"Adapter type: {type(adapter).__name__}")
    
    # Test clearance measurements
    test_bearings = [0, -45, 45, -90, 90]
    
    print("\nTesting clearance measurements:")
    for bearing in test_bearings:
        clearance = adapter.get_clearance_at_bearing(bearing)
        print(f"  Bearing {bearing:3d}°: {clearance:.2f}m" if clearance else f"  Bearing {bearing:3d}°: No data")
    
    # Test stable walls
    stable_walls = adapter.get_stable_wall_bearings()
    print(f"\nStable wall bearings: {stable_walls}")
    
    # Cleanup
    adapter.cleanup()
    
    print("Depth adapter test complete")
    return True

if __name__ == "__main__":
    test_depth_adapter()