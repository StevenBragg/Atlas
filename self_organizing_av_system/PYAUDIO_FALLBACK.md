# PyAudio Fallback Mode

The AVCapture class has been modified to support running in video-only mode when PyAudio is not available.

## Changes Made

1. **Conditional Audio Initialization**: The `start()` method now checks `PYAUDIO_AVAILABLE` before initializing PyAudio. If PyAudio is not available:
   - Sets `self.audio = None` and `self.stream = None`
   - Logs a warning about running in video-only mode
   - Continues with video capture only

2. **Safe Audio Cleanup**: The `stop()` method checks `PYAUDIO_AVAILABLE` before attempting to stop and close audio streams.

3. **Silent Audio Fallback**: When PyAudio is not available:
   - `get_av_pair()` returns zeros for audio data
   - `get_audio()` returns zeros instead of None
   - Audio shape matches the expected `chunk_size` parameter

4. **Camera Fallback**: The system also handles missing cameras gracefully:
   - Uses test patterns when no camera is available
   - Periodically attempts to reconnect to the camera
   - Continues operation with generated video frames

## Usage

No code changes are required in applications using AVCapture. The class automatically detects whether PyAudio is available and adjusts its behavior accordingly.

```python
from utils.capture import AVCapture

# This works whether PyAudio is available or not
capture = AVCapture()
if capture.start():
    frame, audio = capture.get_av_pair()
    # audio will be zeros if PyAudio is not available
```

## Testing

Run `test_video_only.py` to verify the video-only mode functionality.