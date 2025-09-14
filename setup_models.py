#!/usr/bin/env python3
"""
MidnightCore Model Setup Script
Automatically downloads required AI models for the VRChat brain system
"""

import os
import sys
from pathlib import Path

def setup_directories():
    """Create necessary model directories"""
    base_dir = Path(__file__).parent
    model_dirs = [
        "Core/Models/vision",
        "Core/Models/llm", 
        "Core/Engine/models",
        "Core/Common/cache/world_data",
        "Core/Common/cache/spatial_maps",
        "Core/Common/cache/vision_cache"
    ]
    
    for dir_path in model_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def download_florence2():
    """Download Florence-2 Large model"""
    try:
        from huggingface_hub import snapshot_download
        
        model_path = Path(__file__).parent / "Core/Models/vision/florence-2-large"
        
        if model_path.exists() and any(model_path.iterdir()):
            print("✅ Florence-2 already downloaded")
            return
            
        print("⬇️  Downloading Florence-2 Large (~1.5GB)...")
        snapshot_download(
            repo_id="microsoft/Florence-2-large",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✅ Florence-2 downloaded successfully")
        
    except ImportError:
        print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
    except Exception as e:
        print(f"❌ Error downloading Florence-2: {e}")

def download_depth_anything_v2():
    """Download Depth Anything V2 model"""
    model_path = Path(__file__).parent / "Core/Models/vision/depth_anything_v2_small.onnx"
    
    if model_path.exists():
        print("✅ Depth Anything V2 already downloaded")
        return
        
    print("📥 Please manually download Depth Anything V2:")
    print("   1. Visit: https://github.com/DepthAnything/Depth-Anything-V2/releases")
    print("   2. Download: depth_anything_v2_small.onnx")
    print(f"   3. Place in: {model_path}")
    print("   (Automatic download not available)")

def check_dependencies():
    """Check if required Python packages are installed"""
    required = [
        'torch', 'torchvision', 'transformers', 'onnxruntime-gpu',
        'opencv-python', 'pillow', 'numpy', 'python-osc', 'pyautogui'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def main():
    print("🚀 MidnightCore Model Setup")
    print("=" * 40)
    
    # Check dependencies first
    print("\n1. Checking Python dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Create directories
    print("\n2. Creating model directories...")
    setup_directories()
    
    # Download models
    print("\n3. Downloading AI models...")
    download_florence2()
    download_depth_anything_v2()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Manually download Depth Anything V2 if not already done")
    print("2. Configure VRChat OSC settings")
    print("3. Run: python Core/Engine/Qwen_Brain_ActiveTesting.py")

if __name__ == "__main__":
    main()