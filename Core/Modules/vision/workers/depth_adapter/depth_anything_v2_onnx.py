"""
Depth Anything V2 Small ONNX Model Loader and Inference
Fast depth estimation using ONNX runtime with GPU acceleration
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Optional, Tuple
import os

# Configuration
MODEL_PATH = r"G:\Experimental\ComfyUI_windows_portable\ComfyUI\models\onnx\depth_anything_v2_small.onnx"
SHORT_SIDE = 640  # Resize short side to this value

class DepthAnythingV2ONNX:
    """
    Depth Anything V2 Small ONNX inference wrapper
    Provides fast monocular depth estimation for collision avoidance
    """
    
    def __init__(self, model_path: str = MODEL_PATH, short_side: int = SHORT_SIDE):
        self.model_path = model_path
        self.short_side = short_side
        self.session = None
        self.input_name = None
        self.output_name = None
        
    def load_model(self) -> bool:
        """
        Load ONNX model with GPU acceleration if available
        Returns: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"ERROR: Model not found at {self.model_path}")
                return False
                
            # Configure providers - try GPU first, fallback to CPU
            providers = []
            if ort.get_device() == 'GPU':
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            
            # Load session
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"SUCCESS: Depth Anything V2 Small loaded successfully")
            print(f"   Model: {os.path.basename(self.model_path)}")
            print(f"   Provider: {self.session.get_providers()[0]}")
            print(f"   Input shape: {self.session.get_inputs()[0].shape}")
            print(f"   Output shape: {self.session.get_outputs()[0].shape}")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading depth model: {e}")
            return False
    
    def preprocess_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess BGR frame for ONNX inference
        
        Args:
            frame_bgr: Input frame in BGR format [H, W, 3]
            
        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        # Store original size
        original_h, original_w = frame_bgr.shape[:2]
        original_size = (original_h, original_w)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio (short side = SHORT_SIDE)
        if original_h < original_w:
            new_h = self.short_side
            new_w = int(original_w * (self.short_side / original_h))
        else:
            new_w = self.short_side
            new_h = int(original_h * (self.short_side / original_w))
            
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] and convert to float32
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # NHWC -> NCHW (add batch dimension and transpose)
        frame_tensor = np.transpose(frame_normalized, (2, 0, 1))  # HWC -> CHW
        frame_tensor = np.expand_dims(frame_tensor, axis=0)       # CHW -> NCHW
        
        return frame_tensor, original_size
    
    def postprocess_depth(self, depth_output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth output to original image size
        
        Args:
            depth_output: Raw depth output from model
            original_size: (height, width) of original image
            
        Returns:
            Depth map in original size [H, W] as float32
        """
        original_h, original_w = original_size
        
        # Remove batch dimension if present
        if len(depth_output.shape) == 4:
            depth_map = depth_output[0, 0]  # [1, 1, H, W] -> [H, W]
        elif len(depth_output.shape) == 3:
            depth_map = depth_output[0]     # [1, H, W] -> [H, W]
        else:
            depth_map = depth_output        # Already [H, W]
        
        # Resize to original dimensions
        depth_resized = cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return depth_resized.astype(np.float32)
    
    def predict_depth(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict depth map from BGR frame
        
        Args:
            frame_bgr: Input frame in BGR format [H, W, 3]
            
        Returns:
            Depth map [H, W] as float32, or None if error
        """
        if self.session is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return None
            
        try:
            # Preprocess
            input_tensor, original_size = self.preprocess_frame(frame_bgr)
            
            # Inference
            depth_output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            
            # Postprocess
            depth_map = self.postprocess_depth(depth_output, original_size)
            
            return depth_map
            
        except Exception as e:
            print(f"ERROR in depth prediction: {e}")
            return None

# Global model instance
_depth_model = None

def load_model(model_path: str = MODEL_PATH) -> ort.InferenceSession:
    """
    Load and return ONNX inference session
    For compatibility with the original API design
    """
    global _depth_model
    if _depth_model is None:
        _depth_model = DepthAnythingV2ONNX(model_path)
        if not _depth_model.load_model():
            return None
    return _depth_model.session

def predict_depth(session: ort.InferenceSession, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Predict depth using loaded session
    For compatibility with the original API design
    
    Args:
        session: ONNX inference session (ignored, uses global model)
        frame_bgr: Input frame in BGR format
        
    Returns:
        Depth map [H, W] as float32
    """
    global _depth_model
    if _depth_model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    return _depth_model.predict_depth(frame_bgr)

# Test function
def test_depth_model():
    """Test the depth model with a dummy frame"""
    print("TESTING: Depth Anything V2 Small ONNX model...")
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Load model
    model = DepthAnythingV2ONNX()
    if not model.load_model():
        print("FAILED: Failed to load model")
        return False
        
    # Predict depth
    depth_map = model.predict_depth(test_frame)
    
    if depth_map is not None:
        print(f"SUCCESS: Test successful!")
        print(f"   Input shape: {test_frame.shape}")
        print(f"   Depth shape: {depth_map.shape}")
        print(f"   Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        return True
    else:
        print("FAILED: Test failed")
        return False

if __name__ == "__main__":
    test_depth_model()