#!/usr/bin/env python3
"""
Segmentation Setup Script

This script helps set up all three segmentation approaches:
1. Neural Network-based segmentation (PyTorch, DeepLabV3, U-Net)
2. YOLO-based segmentation (Ultralytics YOLO11)
3. SAM 2-based segmentation (Meta's Segment Anything Model 2)
"""

import subprocess
import sys
import os
from pathlib import Path
import urllib.request


def run_command(command, description, check=True):
    """Run a command with error handling."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   📝 Details: {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   📝 Requires Python 3.8 or higher")
        return False


def install_basic_requirements():
    """Install basic requirements."""
    print("\n📦 Installing basic requirements...")
    
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "Pillow",
        "tqdm"
    ]
    
    for req in requirements:
        success = run_command(f"pip install {req}", f"Installing {req}")
        if not success:
            return False
    
    return True


def install_yolo():
    """Install YOLO (Ultralytics) requirements."""
    print("\n🎯 Installing YOLO (Ultralytics) requirements...")
    return run_command("pip install ultralytics>=8.0.0", "Installing Ultralytics YOLO")


def install_sam2():
    """Install SAM 2 requirements."""
    print("\n🎭 Installing SAM 2 requirements...")
    print("   📝 Note: SAM 2 requires manual installation from GitHub")
    
    success = run_command(
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
        "Installing SAM 2 from GitHub",
        check=False
    )
    
    if not success:
        print("   ⚠️  SAM 2 installation failed. This is optional.")
        print("   📝 You can install it manually later with:")
        print("      pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    
    return True  # Don't fail setup if SAM 2 fails


def create_directory_structure():
    """Create necessary directory structure."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data/datasets/dataset_1",
        "data/output",
        "modules/segmentation",
        "modules/segmentation_yolo", 
        "modules/segmentation_sam",
        "models/segmentation_yolo/weights",
        "models/segmentation_yolo/cache",
        "models/segmentation_yolo/runs",
        "models/segmentation_sam/weights",
        "models/segmentation_sam/cache",
        "models/segmentation_sam/outputs",
        "models/segmentation_sam/configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True


def download_sample_model():
    """Download a sample YOLO model for testing."""
    print("\n🤖 Downloading sample YOLO model for testing...")
    
    weights_dir = Path("models/segmentation_yolo/weights")
    model_path = weights_dir / "yolo11n-seg.pt"
    
    if model_path.exists():
        print(f"   ✅ Model already exists: {model_path}")
        return True
    
    # YOLO models are downloaded automatically when first used
    print("   📝 YOLO models will be downloaded automatically on first use")
    return True


def show_sam2_download_instructions():
    """Show instructions for downloading SAM 2 models."""
    print("\n🎭 SAM 2 Model Download Instructions:")
    print("=" * 60)
    print("SAM 2 models need to be downloaded manually from Meta AI:")
    print("https://ai.meta.com/sam2/")
    print()
    
    models = [
        ("sam2_hiera_tiny.pt", "~38MB", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        ("sam2_hiera_small.pt", "~159MB", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        ("sam2_hiera_base_plus.pt", "~80MB", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        ("sam2_hiera_large.pt", "~224MB", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
    ]
    
    weights_dir = Path("models/segmentation_sam/weights")
    print(f"Download to: {weights_dir.absolute()}")
    print()
    
    for model_name, size, url in models:
        exists = "✅" if (weights_dir / model_name).exists() else "❌"
        print(f"{exists} {model_name:25} - {size:8} - {url}")
    
    print()
    print("Quick download commands:")
    print(f"cd {weights_dir.absolute()}")
    for model_name, size, url in models:
        print(f"wget {url}")
    print()


def test_installations():
    """Test all installations."""
    print("\n🧪 Testing installations...")
    
    tests = [
        ("python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", "PyTorch"),
        ("python3 -c 'import torchvision; print(f\"TorchVision: {torchvision.__version__}\")'", "TorchVision"),
        ("python3 -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\")'", "OpenCV"),
        ("python3 -c 'import ultralytics; print(f\"Ultralytics: {ultralytics.__version__}\")'", "Ultralytics YOLO"),
        ("python3 -c 'from modules.segmentation import NeuralImageSegmenter; print(\"Neural Segmentation: OK\")'", "Neural Segmentation Module"),
        ("python3 -c 'from modules.segmentation_yolo import YOLOImageSegmenter; print(\"YOLO Segmentation: OK\")'", "YOLO Segmentation Module"),
        ("python3 -c 'from modules.segmentation_sam import SAMImageSegmenter; print(\"SAM Segmentation: OK\")'", "SAM Segmentation Module")
    ]
    
    success_count = 0
    for command, name in tests:
        if run_command(command, f"Testing {name}", check=False):
            success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{len(tests)} tests passed")
    return success_count == len(tests)


def show_usage_examples():
    """Show usage examples for all three approaches."""
    print("\n🚀 Usage Examples:")
    print("=" * 60)
    
    print("1. Neural Network Segmentation:")
    print("   python app.py --method-set balanced")
    print("   python app.py --methods deeplabv3 neural_clustering")
    print("   python app.py --multi-dataset")
    print()
    
    print("2. YOLO Segmentation:")
    print("   python app_yolo.py --preset balanced")
    print("   python app_yolo.py --model yolo11n-seg")
    print("   python app_yolo.py --multi-dataset --preset fastest")
    print()
    
    print("3. SAM 2 Segmentation:")
    print("   python app_sam.py --preset balanced")
    print("   python app_sam.py --prompt everything")
    print("   python app_sam.py --points 400,300 --preset fastest")
    print()
    
    print("Analysis and Help:")
    print("   python app.py --analyze-only")
    print("   python app_yolo.py --list-models")
    print("   python app_sam.py --download-info")


def main():
    """Main setup function."""
    print("🎯 Image Segmentation Setup Script")
    print("=" * 60)
    print("This script will set up three segmentation approaches:")
    print("1. 🧠 Neural Network-based (PyTorch, DeepLabV3, U-Net)")
    print("2. 🎯 YOLO-based (Ultralytics YOLO11)")
    print("3. 🎭 SAM 2-based (Meta's Segment Anything Model 2)")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        print("❌ Failed to create directory structure")
        sys.exit(1)
    
    # Install basic requirements
    if not install_basic_requirements():
        print("❌ Failed to install basic requirements")
        sys.exit(1)
    
    # Install YOLO
    if not install_yolo():
        print("❌ Failed to install YOLO requirements")
        sys.exit(1)
    
    # Install SAM 2 (optional)
    install_sam2()
    
    # Download sample models
    download_sample_model()
    
    # Show SAM 2 download instructions
    show_sam2_download_instructions()
    
    # Test installations
    all_tests_passed = test_installations()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 Setup completed successfully!")
        print("✅ All segmentation approaches are ready to use")
    else:
        print("⚠️  Setup completed with some issues")
        print("📝 Some optional components may not be fully functional")
    
    print("\n📚 Next steps:")
    print("1. 🧪 Test neural network segmentation: python app.py")
    print("2. 🎯 Test YOLO segmentation: python app_yolo.py")
    print("3. 🎭 Download SAM 2 models and test: python app_sam.py --download-info")
    print("=" * 60)


if __name__ == "__main__":
    main()
