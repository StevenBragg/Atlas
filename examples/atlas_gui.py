#!/usr/bin/env python
"""
Atlas GUI - Desktop Application Entry Point

Launch the Atlas Challenge-Based Learning GUI application with:
- Curriculum Learning Tab: Structured multimodal challenges
- Free Play Tab: Webcam, microphone, chat, and creative canvas

Usage:
    python examples/atlas_gui.py

Requirements:
    - PyQt5 or PyQt6: pip install PyQt5
    - OpenCV (optional, for webcam): pip install opencv-python
    - PyAudio (optional, for microphone): pip install pyaudio
    - CuPy (optional, for GPU acceleration): pip install cupy-cuda11x
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def check_dependencies():
    """Check for required dependencies."""
    missing = []

    # Check PyQt
    try:
        try:
            import PyQt5
            print("[OK] PyQt5 found")
        except ImportError:
            import PyQt6
            print("[OK] PyQt6 found")
    except ImportError:
        missing.append("PyQt5 (pip install PyQt5)")

    # Check NumPy
    try:
        import numpy
        print("[OK] NumPy found")
    except ImportError:
        missing.append("numpy (pip install numpy)")

    # Check PIL
    try:
        from PIL import Image
        print("[OK] Pillow found")
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    # Optional: OpenCV
    try:
        import cv2
        print("[OK] OpenCV found (webcam enabled)")
    except ImportError:
        print("[--] OpenCV not found (webcam will use synthetic frames)")

    # Optional: PyAudio
    try:
        import pyaudio
        print("[OK] PyAudio found (microphone enabled)")
    except ImportError:
        print("[--] PyAudio not found (microphone will use synthetic audio)")

    # Optional: CuPy
    try:
        import cupy
        print("[OK] CuPy found (GPU acceleration enabled)")
    except ImportError:
        print("[--] CuPy not found (using CPU mode)")

    return missing


def main():
    """Main entry point for Atlas GUI."""
    print("=" * 60)
    print("Atlas Challenge-Based Learning System")
    print("Desktop GUI Application")
    print("=" * 60)
    print()

    # Check dependencies
    print("Checking dependencies...")
    print("-" * 40)
    missing = check_dependencies()
    print("-" * 40)
    print()

    if missing:
        print("ERROR: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print()
        print("Please install missing dependencies and try again.")
        sys.exit(1)

    # Import and run
    print("Starting Atlas GUI...")
    print()

    try:
        from self_organizing_av_system.gui.pyqt.app import main as run_app
        run_app()
    except Exception as e:
        print(f"Error starting Atlas GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
