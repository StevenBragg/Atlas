#!/usr/bin/env python3
"""
Test script to verify that AVCapture works in video-only mode when PyAudio is not available.
"""

import sys
import time
import numpy as np
from utils.capture import AVCapture, PYAUDIO_AVAILABLE


def test_video_only_mode():
    """Test the AVCapture class in video-only mode."""
    print(f"PyAudio available: {PYAUDIO_AVAILABLE}")
    
    # Create capture instance
    capture = AVCapture(
        video_width=640,
        video_height=480,
        fps=30,
        sample_rate=22050,
        channels=1,
        chunk_size=1024
    )
    
    # Start capture
    print("Starting capture...")
    if not capture.start():
        print("Failed to start capture!")
        return False
    
    print("Capture started successfully!")
    print("Testing various capture methods for 5 seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 5.0:
            # Test get_av_pair
            frame, audio = capture.get_av_pair(block=False)
            if frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Log every second
                    print(f"Frame {frame_count}: shape={frame.shape}, "
                          f"audio shape={audio.shape if audio is not None else 'None'}")
                    
                    # Verify audio is zeros when PyAudio not available
                    if not PYAUDIO_AVAILABLE and audio is not None:
                        assert np.all(audio == 0), "Audio should be zeros when PyAudio not available"
            
            # Test get_frame only
            frame = capture.get_frame(block=False)
            
            # Test get_audio only
            audio = capture.get_audio(block=False)
            if not PYAUDIO_AVAILABLE and audio is not None:
                assert np.all(audio == 0), "Audio should be zeros when PyAudio not available"
            
            # Small delay to avoid busy loop
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print(f"\nStopping capture... (captured {frame_count} frames)")
        capture.stop()
        print("Capture stopped successfully!")
    
    return True


def test_context_manager():
    """Test using AVCapture as a context manager."""
    print("\nTesting context manager...")
    
    with AVCapture() as capture:
        print("Inside context manager")
        
        # Get a few frames
        for i in range(5):
            frame, audio = capture.get_av_pair(block=True, timeout=1.0)
            if frame is not None:
                print(f"Got frame {i+1}: {frame.shape}")
            time.sleep(0.1)
    
    print("Context manager exited successfully!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing AVCapture in video-only mode")
    print("=" * 60)
    
    # Test basic functionality
    if not test_video_only_mode():
        print("Basic test failed!")
        return 1
    
    # Test context manager
    if not test_context_manager():
        print("Context manager test failed!")
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())