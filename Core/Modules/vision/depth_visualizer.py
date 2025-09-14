# ==== MODULE CONTRACT =======================================================
# Module: vision/depth_visualizer.py
# Package: MidnightCore.Modules.vision.depth_visualizer  
# Location: Production/MidnightCore/Core/Modules/vision/depth_visualizer.py
# Responsibility: Selective depth visualization for navigation decisions
# PUBLIC: save_navigation_decision_image()
# DEPENDENCIES: cv2, numpy, hybrid_navigation regions
# POLICY: Only saves on major navigation decisions, not depth pulses
# ============================================================================

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

# Navigation decision visualization colors (BGR format for OpenCV)
DECISION_COLORS = {
    'MOVE_FORWARD_CLEAR': (0, 255, 0),      # Green - safe path
    'STOP_CLIFF_DANGER': (0, 0, 255),       # Red - danger  
    'STOP_WALL_CLOSE': (0, 100, 255),       # Orange - wall
    'STOP_BOXED_IN': (0, 0, 200),           # Dark red - boxed
    'NAVIGATE_AROUND': (255, 255, 0),       # Cyan - navigate
    'ASSESS_CAREFULLY': (0, 255, 255),      # Yellow - uncertain
    'INITIALIZING_TEMPORAL_BUFFERS': (128, 128, 128)  # Gray - initializing
}

def create_depth_heatmap(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert depth map to colored heatmap visualization
    Blue = far, Red = close
    """
    if depth_map is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Normalize depth to 0-255 range
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Invert so close = warm colors, far = cool colors
    depth_inverted = 255 - depth_norm
    
    # Apply colormap (COLORMAP_JET: blue=far, red=close)
    heatmap = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)
    
    return heatmap

def draw_navigation_regions(image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Overlay navigation analysis regions on the image
    """
    height, width = image.shape[:2]
    overlay = image.copy()
    
    # Define regions (same as hybrid_navigation.py)
    regions = {
        'immediate_forward': (int(width*0.3), int(height*0.7), int(width*0.7), height),
        'forward_path': (int(width*0.35), int(height*0.4), int(width*0.65), int(height*0.6)), 
        'forward_wall': (int(width*0.4), int(height*0.25), int(width*0.6), int(height*0.45)),
        'left_side': (0, int(height*0.3), int(width*0.3), int(height*0.7)),
        'right_side': (int(width*0.7), int(height*0.3), width, int(height*0.7))
    }
    
    # Region colors (BGR)
    region_colors = {
        'immediate_forward': (0, 255, 255),    # Yellow - critical zone
        'forward_path': (0, 255, 0),           # Green - navigation path
        'forward_wall': (255, 0, 0),           # Blue - wall detection  
        'left_side': (255, 0, 255),            # Magenta - left side
        'right_side': (255, 255, 0)            # Cyan - right side
    }
    
    # Draw region overlays
    for region_name, (x1, y1, x2, y2) in regions.items():
        color = region_colors.get(region_name, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        # Add region label
        cv2.putText(overlay, region_name, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Blend overlay with original
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return result

def add_decision_info(image: np.ndarray, decision_type: str, tick_id: int, 
                     frame_id: int, threat_summary: str = "") -> np.ndarray:
    """
    Add decision information overlay to the image
    """
    height, width = image.shape[:2]
    
    # Decision color
    color = DECISION_COLORS.get(decision_type, (255, 255, 255))
    
    # Add decision banner at top
    cv2.rectangle(image, (0, 0), (width, 60), (0, 0, 0), -1)  # Black background
    cv2.rectangle(image, (0, 0), (width, 60), color, 3)       # Colored border
    
    # Decision text
    cv2.putText(image, f"TICK {tick_id} - {decision_type}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Timestamp and frame info
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(image, f"Frame {frame_id} - {timestamp}", (10, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Threat summary at bottom if provided
    if threat_summary:
        cv2.rectangle(image, (0, height-40), (width, height), (0, 0, 0), -1)
        cv2.putText(image, threat_summary[:80], (10, height-15),  # Truncate if too long
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image

def save_navigation_decision_image(depth_map: np.ndarray, 
                                 decision_type: str,
                                 tick_id: int,
                                 frame_id: int,
                                 threat_summary: str = "",
                                 save_dir: str = r"G:\Experimental\Production\MidnightCore\Core\Engine\Captures\Navigation") -> str:
    """
    Save a depth visualization image for a navigation decision
    
    Args:
        depth_map: Raw depth map from Depth Anything V2
        decision_type: Navigation decision (MOVE_FORWARD_CLEAR, etc.)
        tick_id: Current tick ID
        frame_id: Current frame ID
        threat_summary: Brief description of detected threats
        save_dir: Directory to save image
        
    Returns:
        Path to saved image or empty string if failed
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create depth heatmap
        heatmap = create_depth_heatmap(depth_map)
        
        # Add navigation regions overlay
        visualization = draw_navigation_regions(heatmap, alpha=0.2)
        
        # Add decision information
        final_image = add_decision_info(visualization, decision_type, tick_id, frame_id, threat_summary)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"depth_decision_tick{tick_id:02d}_{decision_type}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Save image
        success = cv2.imwrite(filepath, final_image)
        
        if success:
            print(f"[DEPTH VIZ] Saved navigation decision visualization: {filename}")
            return filepath
        else:
            print(f"[DEPTH VIZ] Failed to save image: {filename}")
            return ""
            
    except Exception as e:
        print(f"[DEPTH VIZ] Error saving navigation image: {e}")
        return ""

def should_save_decision_image(decision_type: str) -> bool:
    """
    Determine if this decision type warrants saving a visualization
    Only save for major navigation decisions, not routine checks
    """
    major_decisions = {
        'MOVE_FORWARD_CLEAR',
        'STOP_CLIFF_DANGER', 
        'STOP_WALL_CLOSE',
        'STOP_BOXED_IN',
        'NAVIGATE_AROUND',
        'ASSESS_CAREFULLY'
        # Note: INITIALIZING_TEMPORAL_BUFFERS excluded (too frequent)
    }
    
    return decision_type in major_decisions