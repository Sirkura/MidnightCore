#!/usr/bin/env python3
# ==== MODULE CONTRACT =======================================================
# Module: vision/workers/depth_adapter/midas_depth_onnx.py
# Package: MidnightCore.Modules.vision.workers.depth_adapter.midas_depth_onnx
# Location: Production/MidnightCore/Core/Modules/vision/workers/depth_adapter/midas_depth_onnx.py
# Responsibility: MiDaS depth estimation with ONNX and PyTorch support for VRChat synthetic rendering
# PUBLIC: MiDaSDepthEstimator class, load_model() function
# DEPENDENCIES: onnxruntime, transformers, torch
# POLICY: NO_FALLBACKS=deny, Telemetry: midas.*
# MIGRATION: Alternative to depth_anything_v2_onnx.py for improved VRChat compatibility
# ============================================================================

"""
MiDaS Depth Estimation for VRChat Synthetic Environments
=========================================================

Unified adapter supporting both:
- dpt_swin2_tiny_256.onnx (ONNX runtime inference)
- dpt_hybrid_midas (PyTorch transformers inference)

Designed to replace Depth Anything V2 for improved VRChat synthetic rendering compatibility.
"""

import numpy as np
import cv2
import os
from typing import Optional, Tuple, Union, Dict, Any
import time

# ONNX Runtime imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("WARNING: ONNXRuntime not available - ONNX models disabled")
    ONNX_AVAILABLE = False

# PyTorch/Transformers imports  
try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image
    PYTORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch/Transformers not available - PyTorch models disabled")
    PYTORCH_AVAILABLE = False

# Model configurations (paths will be resolved relative to project root)
MODELS = {
    "swin2_tiny_onnx": {
        "path": "Core/models/onnx/dpt_swin2_tiny_256.onnx",
        "type": "onnx",
        "input_size": 256,
        "description": "DPT SwinV2-Tiny ONNX - Fast inference with transformer architecture"
    },
    "hybrid_pytorch": {
        "path": "Core/models/onnx/dpt_hybrid_midas",
        "type": "pytorch",
        "model_name": "Intel/dpt-hybrid-midas",
        "input_size": 384,
        "description": "DPT Hybrid PyTorch - CNN-Transformer hybrid for balanced accuracy/speed"
    }
}

def _get_project_root():
    """Get the project root directory (MidnightCore folder)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up: depth_adapter -> workers -> vision -> Modules -> Core -> MidnightCore
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))

class MiDaSDepthEstimator:
    """
    Unified MiDaS depth estimation supporting both ONNX and PyTorch models
    
    Designed as drop-in replacement for DepthAnythingV2ONNX with improved 
    VRChat synthetic rendering compatibility
    """
    
    def __init__(self, model_key: str = "swin2_tiny_onnx"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_key: Model to use ("swin2_tiny_onnx" or "hybrid_pytorch")
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        
        self.model_config = MODELS[model_key].copy()
        self.model_key = model_key
        self.model_type = self.model_config["type"]

        # Resolve relative path to absolute path
        project_root = _get_project_root()
        self.model_config["path"] = os.path.join(project_root, self.model_config["path"])
        
        # Common attributes
        self.is_loaded = False
        self.device = None
        
        # ONNX-specific attributes
        self.session = None
        self.input_name = None
        self.output_name = None
        
        # PyTorch-specific attributes
        self.processor = None
        self.model = None
        
        print(f"INIT: MiDaS Depth Estimator - {self.model_config['description']}")
    
    def load_model(self) -> bool:
        """
        Load the specified model with appropriate inference backend
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model_type == "onnx":
                return self._load_onnx_model()
            elif self.model_type == "pytorch":
                return self._load_pytorch_model()
            else:
                print(f"ERROR: Unknown model type '{self.model_type}'")
                return False
                
        except Exception as e:
            print(f"ERROR loading MiDaS model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_onnx_model(self) -> bool:
        """Load ONNX model with GPU acceleration"""
        if not ONNX_AVAILABLE:
            print("ERROR: ONNX Runtime not available")
            return False
        
        model_path = self.model_config["path"]
        if not os.path.exists(model_path):
            print(f"ERROR: ONNX model not found at {model_path}")
            return False
        
        # Configure providers - GPU first, CPU fallback
        providers = []
        if ort.get_device() == 'GPU':
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        # Load session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get I/O info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Set device info
        self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        
        print(f"SUCCESS: MiDaS ONNX model loaded")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Provider: {self.session.get_providers()[0]}")
        print(f"   Input: {self.input_name} {self.session.get_inputs()[0].shape}")
        print(f"   Output: {self.output_name} {self.session.get_outputs()[0].shape}")
        
        self.is_loaded = True
        return True
    
    def _load_pytorch_model(self) -> bool:
        """Load PyTorch model with GPU acceleration"""
        if not PYTORCH_AVAILABLE:
            print("ERROR: PyTorch/Transformers not available")
            return False
        
        model_path = self.model_config["path"]
        if not os.path.exists(model_path):
            print(f"ERROR: PyTorch model not found at {model_path}")
            return False
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model from local path
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"SUCCESS: MiDaS PyTorch model loaded")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {next(self.model.parameters()).dtype}")
        
        self.is_loaded = True
        return True
    
    def predict_depth(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict depth map from BGR frame
        
        Args:
            frame_bgr: Input frame in BGR format [H, W, 3]
            
        Returns:
            Depth map [H, W] as float32, or None if error
        """
        if not self.is_loaded:
            print("ERROR: Model not loaded. Call load_model() first.")
            return None
        
        try:
            if self.model_type == "onnx":
                return self._predict_onnx(frame_bgr)
            elif self.model_type == "pytorch":
                return self._predict_pytorch(frame_bgr)
            else:
                print(f"ERROR: Unknown model type '{self.model_type}'")
                return None
                
        except Exception as e:
            print(f"ERROR in MiDaS depth prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_onnx(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """ONNX inference pathway"""
        # Preprocess for ONNX
        input_tensor, original_size = self._preprocess_onnx(frame_bgr)
        
        # Run inference
        start_time = time.time()
        depth_output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Postprocess
        depth_map = self._postprocess_depth(depth_output, original_size)
        
        print(f"MIDAS ONNX: {inference_time:.1f}ms inference")
        return depth_map
    
    def _predict_pytorch(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """PyTorch inference pathway"""
        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        original_size = frame_bgr.shape[:2]  # (H, W)
        
        # Preprocess with transformers processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Move to device and match model dtype
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(self.device, dtype=model_dtype) for k, v in inputs.items()}
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth_tensor = outputs.predicted_depth
        inference_time = (time.time() - start_time) * 1000
        
        # Convert to numpy and postprocess
        depth_np = depth_tensor.squeeze().cpu().numpy()
        depth_map = self._postprocess_depth(depth_np, original_size)
        
        print(f"MIDAS PYTORCH: {inference_time:.1f}ms inference ({self.device})")
        return depth_map
    
    def _preprocess_onnx(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess BGR frame for ONNX inference (similar to DepthAnythingV2)
        """
        original_h, original_w = frame_bgr.shape[:2]
        original_size = (original_h, original_w)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (square)
        target_size = self.model_config["input_size"]
        frame_resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] and convert to float32
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # NHWC -> NCHW format
        frame_tensor = np.transpose(frame_normalized, (2, 0, 1))  # HWC -> CHW
        frame_tensor = np.expand_dims(frame_tensor, axis=0)       # CHW -> NCHW
        
        return frame_tensor, original_size
    
    def _postprocess_depth(self, depth_output: Union[np.ndarray, Any], original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth output to original image size
        """
        original_h, original_w = original_size
        
        # Ensure numpy array
        if not isinstance(depth_output, np.ndarray):
            depth_output = np.array(depth_output)
        
        # Handle batch dimensions
        while len(depth_output.shape) > 2:
            depth_output = depth_output[0]  # Remove batch/channel dims
        
        # Resize to original dimensions
        depth_resized = cv2.resize(depth_output, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return depth_resized.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        info = {
            "model_key": self.model_key,
            "model_type": self.model_type,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "description": self.model_config["description"]
        }
        
        if self.model_type == "onnx" and self.session:
            info["providers"] = self.session.get_providers()
            info["input_shape"] = self.session.get_inputs()[0].shape
            info["output_shape"] = self.session.get_outputs()[0].shape
        elif self.model_type == "pytorch" and self.model:
            info["dtype"] = str(next(self.model.parameters()).dtype)
            info["param_count"] = sum(p.numel() for p in self.model.parameters())
        
        return info

# ============================================================================
# Compatibility layer for drop-in replacement of DepthAnythingV2ONNX
# ============================================================================

# Global model instance
_midas_model = None

def load_model(model_key: str = "swin2_tiny_onnx") -> Union[Any, None]:
    """
    Load and return MiDaS model instance
    Compatibility function for existing depth system integration
    """
    global _midas_model
    _midas_model = MiDaSDepthEstimator(model_key)
    if not _midas_model.load_model():
        _midas_model = None
        return None
    return _midas_model

def predict_depth(session: Any, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Predict depth using loaded MiDaS model
    Compatibility function for existing depth system integration
    
    Args:
        session: Model instance (ignored, uses global model)
        frame_bgr: Input frame in BGR format
        
    Returns:
        Depth map [H, W] as float32
    """
    global _midas_model
    if _midas_model is None:
        raise RuntimeError("MiDaS model not loaded. Call load_model() first.")
    
    return _midas_model.predict_depth(frame_bgr)

# ============================================================================
# Testing functions
# ============================================================================

def test_model(model_key: str = "swin2_tiny_onnx") -> bool:
    """Test specified MiDaS model with dummy frame"""
    print(f"TESTING: MiDaS model '{model_key}'")
    print("=" * 50)
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize model
    estimator = MiDaSDepthEstimator(model_key)
    if not estimator.load_model():
        print("FAILED: Model loading failed")
        return False
    
    # Test prediction
    start_time = time.time()
    depth_map = estimator.predict_depth(test_frame)
    total_time = (time.time() - start_time) * 1000
    
    if depth_map is not None:
        print(f"SUCCESS: Test completed in {total_time:.1f}ms")
        print(f"   Input shape: {test_frame.shape}")
        print(f"   Depth shape: {depth_map.shape}")
        print(f"   Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        print(f"   Model info: {estimator.get_model_info()}")
        return True
    else:
        print("FAILED: Depth prediction failed")
        return False

def test_all_models() -> Dict[str, bool]:
    """Test all available MiDaS models"""
    results = {}
    
    print("TESTING: All available MiDaS models")
    print("=" * 60)
    
    for model_key in MODELS.keys():
        print(f"\n>>> Testing {model_key}...")
        results[model_key] = test_model(model_key)
        print()
    
    print("RESULTS SUMMARY:")
    print("-" * 20)
    for model_key, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"   {model_key}: {status}")
    
    return results

if __name__ == "__main__":
    # Test all models by default
    test_all_models()