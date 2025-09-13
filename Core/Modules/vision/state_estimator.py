"""
State Estimator - Depth Analysis and Optical Flow Processing
Converts depth maps and optical flow into navigation-ready state vectors
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
import time

class StateEstimator:
    """
    Processes depth maps and optical flow to extract navigation state
    Outputs compact state vectors for Beta's decision making
    """
    
    def __init__(self, fov_deg: float = 120.0):
        """
        Initialize state estimator
        
        Args:
            fov_deg: Horizontal field of view in degrees (default 120°)
        """
        self.fov_deg = fov_deg
        self.fov_rad = np.radians(fov_deg)
        
        # Create bearing bins for clearance analysis (31 bins = ±60°, 4° per bin)
        self.n_bearings = 31
        self.bearing_angles = np.linspace(-fov_deg/2, fov_deg/2, self.n_bearings)
        
        # Wedge definitions (in degrees)
        self.front_wedge = (-20, 20)    # ±20° for front clearance
        self.left_wedge = (-60, -20)    # -60° to -20° for left clearance  
        self.right_wedge = (20, 60)     # 20° to 60° for right clearance
        
    def _get_wedge_mask(self, width: int, wedge: Tuple[float, float]) -> np.ndarray:
        """
        Create column mask for angular wedge
        
        Args:
            width: Image width
            wedge: (start_angle, end_angle) in degrees
            
        Returns:
            Boolean mask for columns in wedge
        """
        # Convert angles to column indices
        start_angle, end_angle = wedge
        center_col = width / 2
        
        # Convert angles to pixel offsets from center
        pixels_per_degree = width / self.fov_deg
        start_col = center_col + start_angle * pixels_per_degree
        end_col = center_col + end_angle * pixels_per_degree
        
        # Clamp to image bounds
        start_col = max(0, int(start_col))
        end_col = min(width, int(end_col))
        
        # Create mask
        mask = np.zeros(width, dtype=bool)
        if start_col < end_col:
            mask[start_col:end_col] = True
            
        return mask
    
    def _compute_wedge_clearance(self, depth_map: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute median clearance in front, left, and right wedges
        
        Args:
            depth_map: Depth map [H, W]
            
        Returns:
            Tuple of (front_m, left_m, right_m)
        """
        height, width = depth_map.shape
        
        # Get wedge masks
        front_mask = self._get_wedge_mask(width, self.front_wedge)
        left_mask = self._get_wedge_mask(width, self.left_wedge)
        right_mask = self._get_wedge_mask(width, self.right_wedge)
        
        # Extract depth values in each wedge
        front_depths = depth_map[:, front_mask].flatten()
        left_depths = depth_map[:, left_mask].flatten()
        right_depths = depth_map[:, right_mask].flatten()
        
        # Compute median clearance (robust to outliers)
        front_m = np.median(front_depths) if len(front_depths) > 0 else float('inf')
        left_m = np.median(left_depths) if len(left_depths) > 0 else float('inf')
        right_m = np.median(right_depths) if len(right_depths) > 0 else float('inf')
        
        return front_m, left_m, right_m
    
    def _compute_clearance_by_bearing(self, depth_map: np.ndarray) -> List[List[float]]:
        """
        Compute clearance for each bearing bin
        
        Args:
            depth_map: Depth map [H, W]
            
        Returns:
            List of [angle_deg, clearance_m] pairs
        """
        height, width = depth_map.shape
        clearance_by_bearing = []
        
        for angle_deg in self.bearing_angles:
            # Create narrow wedge around this bearing (±2°)
            wedge = (angle_deg - 2, angle_deg + 2)
            mask = self._get_wedge_mask(width, wedge)
            
            if np.any(mask):
                # Get median depth in this wedge
                wedge_depths = depth_map[:, mask].flatten()
                clearance_m = float(np.median(wedge_depths))
            else:
                clearance_m = float('inf')
                
            clearance_by_bearing.append([float(angle_deg), clearance_m])
            
        return clearance_by_bearing
    
    def _compute_edge_risk(self, depth_map: np.ndarray) -> float:
        """
        Compute edge/cliff risk by analyzing vertical depth gradients
        
        Args:
            depth_map: Depth map [H, W]
            
        Returns:
            Edge risk score [0-1], higher = more dangerous
        """
        height, width = depth_map.shape
        
        # Focus on lower third of image (ground plane)
        lower_band_start = int(height * 0.67)
        lower_band = depth_map[lower_band_start:, :]
        
        if lower_band.shape[0] < 2:
            return 0.0
        
        # Compute vertical gradient (downward differences)
        grad_y = np.gradient(lower_band, axis=0)
        
        # Negative gradients indicate potential cliffs/drops
        negative_grads = grad_y[grad_y < 0]
        
        if len(negative_grads) == 0:
            return 0.0
        
        # Edge risk = magnitude of steepest negative gradient
        max_negative_grad = abs(np.min(negative_grads))
        
        # Normalize to [0, 1] range (clamp large gradients)
        edge_risk = min(1.0, max_negative_grad / 2.0)
        
        return float(edge_risk)
    
    def _compute_ground_tilt(self, depth_map: np.ndarray) -> float:
        """
        Estimate ground plane tilt from depth map
        
        Args:
            depth_map: Depth map [H, W]
            
        Returns:
            Tilt angle in degrees (positive = upward slope ahead)
        """
        height, width = depth_map.shape
        
        # Sample ground plane from lower band
        lower_band_start = int(height * 0.67)
        lower_band = depth_map[lower_band_start:, :]
        
        if lower_band.shape[0] < 3:
            return 0.0
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:lower_band.shape[0], 0:lower_band.shape[1]]
        
        # Flatten for plane fitting
        y_flat = y_coords.flatten()
        x_flat = x_coords.flatten()
        z_flat = lower_band.flatten()
        
        # Remove invalid depths
        valid_mask = np.isfinite(z_flat) & (z_flat > 0)
        if np.sum(valid_mask) < 10:
            return 0.0
        
        y_valid = y_flat[valid_mask]
        x_valid = x_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        
        try:
            # Fit plane: z = ax + by + c
            A = np.column_stack([x_valid, y_valid, np.ones(len(x_valid))])
            plane_params, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
            
            # Extract tilt: arctan of y-direction gradient
            tilt_rad = np.arctan(plane_params[1])  # b parameter
            tilt_deg = np.degrees(tilt_rad)
            
            return float(tilt_deg)
            
        except np.linalg.LinAlgError:
            return 0.0
    
    def _compute_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, List[List[float]]]:
        """
        Compute optical flow between consecutive frames
        
        Args:
            prev_gray: Previous grayscale frame [H, W]
            curr_gray: Current grayscale frame [H, W]
            
        Returns:
            Tuple of (mean_flow_magnitude, flow_by_bearing)
        """
        if prev_gray is None:
            return 0.0, [[angle, 0.0] for angle in self.bearing_angles]
        
        try:
            # Compute dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, 
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
        except Exception as e:
            print(f"Warning: Optical flow computation failed: {e}")
            flow = None
        
        if flow is None:
            return 0.0, [[angle, 0.0] for angle in self.bearing_angles]
        
        # Compute flow magnitude
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Mean flow in frontal region
        height, width = flow_magnitude.shape
        front_mask = self._get_wedge_mask(width, self.front_wedge)
        
        if np.any(front_mask):
            mean_flow = float(np.mean(flow_magnitude[:, front_mask]))
        else:
            mean_flow = float(np.mean(flow_magnitude))
        
        # Flow by bearing
        flow_by_bearing = []
        for angle_deg in self.bearing_angles:
            wedge = (angle_deg - 2, angle_deg + 2)
            mask = self._get_wedge_mask(width, wedge)
            
            if np.any(mask):
                flow_mag = float(np.mean(flow_magnitude[:, mask]))
            else:
                flow_mag = 0.0
                
            flow_by_bearing.append([float(angle_deg), flow_mag])
        
        return mean_flow, flow_by_bearing
    
    def _compute_ttc(self, front_clearance: float, mean_flow: float, flow_scale: float = 4.0) -> float:
        """
        Compute time-to-contact based on clearance and flow
        
        Args:
            front_clearance: Distance to nearest obstacle ahead (meters)
            mean_flow: Mean optical flow magnitude in frontal region
            flow_scale: Scaling factor for flow->velocity conversion
            
        Returns:
            Time to contact in seconds
        """
        if mean_flow <= 0.001 or not np.isfinite(front_clearance):
            return float('inf')
        
        # Estimate velocity from optical flow
        velocity_estimate = mean_flow * flow_scale
        
        # TTC = distance / velocity
        ttc_s = front_clearance / (velocity_estimate + 1e-6)
        
        # Clamp to reasonable range
        ttc_s = max(0.1, min(60.0, ttc_s))
        
        return float(ttc_s)
    
    def compute_state(self, depth_map: np.ndarray, prev_gray: Optional[np.ndarray], 
                     curr_gray: np.ndarray) -> Dict:
        """
        Compute complete navigation state from depth and flow
        
        Args:
            depth_map: Depth map [H, W] in meters
            prev_gray: Previous grayscale frame [H, W] (None for first frame)
            curr_gray: Current grayscale frame [H, W]
            
        Returns:
            State dictionary with all navigation parameters
        """
        # Wedge clearances
        front_m, left_m, right_m = self._compute_wedge_clearance(depth_map)
        
        # Detailed clearance by bearing
        clearance_by_bearing = self._compute_clearance_by_bearing(depth_map)
        
        # Edge and tilt analysis
        edge_risk = self._compute_edge_risk(depth_map)
        tilt_deg = self._compute_ground_tilt(depth_map)
        
        # Optical flow analysis
        mean_flow, flow_by_bearing = self._compute_optical_flow(prev_gray, curr_gray)
        
        # Time to contact
        ttc_s = self._compute_ttc(front_m, mean_flow)
        
        # Compile state vector
        state = {
            "front_m": front_m,
            "left_m": left_m, 
            "right_m": right_m,
            "edge_risk": edge_risk,
            "tilt_deg": tilt_deg,
            "mean_flow": mean_flow,
            "ttc_s": ttc_s,
            "clearance_by_bearing": clearance_by_bearing,
            "flow_by_bearing": flow_by_bearing,
            "ts": time.time()
        }
        
        return state

# Test function
def test_state_estimator():
    """Test state estimator with synthetic data"""
    print("TESTING: State Estimator...")
    
    # Create synthetic depth map (corridor scene)
    height, width = 480, 640
    depth_map = np.ones((height, width), dtype=np.float32) * 5.0  # 5m baseline
    
    # Add walls on sides (closer)
    depth_map[:, :100] = 1.5   # Left wall at 1.5m
    depth_map[:, -100:] = 1.5  # Right wall at 1.5m
    
    # Add obstacle ahead
    depth_map[200:280, 280:360] = 0.8  # Obstacle at 0.8m
    
    # Create synthetic grayscale frames
    gray1 = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    gray2 = gray1 + np.random.randint(-10, 10, (height, width), dtype=np.uint8).astype(np.uint8)
    
    # Test estimator
    estimator = StateEstimator(fov_deg=120.0)
    state = estimator.compute_state(depth_map, gray1, gray2)
    
    print(f"SUCCESS: Test successful!")
    print(f"   Front clearance: {state['front_m']:.2f}m")
    print(f"   Left clearance: {state['left_m']:.2f}m") 
    print(f"   Right clearance: {state['right_m']:.2f}m")
    print(f"   Edge risk: {state['edge_risk']:.3f}")
    print(f"   Tilt: {state['tilt_deg']:.1f}°")
    print(f"   Mean flow: {state['mean_flow']:.3f}")
    print(f"   TTC: {state['ttc_s']:.1f}s")
    print(f"   Bearings tracked: {len(state['clearance_by_bearing'])}")
    
    return True

if __name__ == "__main__":
    test_state_estimator()