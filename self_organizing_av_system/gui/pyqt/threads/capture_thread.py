"""
Capture Thread - Webcam and Microphone Capture

Background thread for capturing AV data that is fed INTO Atlas.
"""

import numpy as np
import logging
from typing import Optional

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class CaptureThread(QThread):
    """
    Background thread for webcam and microphone capture.

    Emits signals:
    - frameReady: New webcam frame available (numpy array)
    - audioReady: New audio buffer available (numpy array)
    - audioLevelReady: Current audio level (float 0-1)
    """

    frameReady = pyqtSignal(object)  # numpy array (H, W, 3) RGB
    audioReady = pyqtSignal(object)  # numpy array (samples,) float32
    audioLevelReady = pyqtSignal(float)

    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TARGET_FPS = 30
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHUNK_SIZE = 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._camera = None
        self._audio_stream = None
        self.last_audio: Optional[np.ndarray] = None

    def run(self):
        """Main capture loop."""
        self._running = True

        # Try to initialize camera
        try:
            import cv2
            self._camera = cv2.VideoCapture(0)
            if self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
                self._camera.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
                logger.info("Camera initialized successfully")
            else:
                self._camera = None
                logger.warning("Could not open camera")
        except ImportError:
            self._camera = None
            logger.warning("OpenCV not available - camera disabled")
        except Exception as e:
            self._camera = None
            logger.error(f"Camera initialization error: {e}")

        # Try to initialize audio
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            self._audio_stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.AUDIO_CHUNK_SIZE
            )
            logger.info("Audio initialized successfully")
        except ImportError:
            self._audio_stream = None
            logger.warning("PyAudio not available - microphone disabled")
        except Exception as e:
            self._audio_stream = None
            logger.error(f"Audio initialization error: {e}")

        # Main loop
        frame_interval_ms = 1000 // self.TARGET_FPS

        while self._running:
            try:
                # Capture video frame
                if self._camera is not None:
                    ret, frame = self._camera.read()
                    if ret:
                        # Convert BGR to RGB
                        import cv2
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frameReady.emit(frame_rgb)
                else:
                    # Generate synthetic frame for testing
                    frame = self._generate_synthetic_frame()
                    self.frameReady.emit(frame)

                # Capture audio
                if self._audio_stream is not None:
                    try:
                        audio_data = self._audio_stream.read(
                            self.AUDIO_CHUNK_SIZE,
                            exception_on_overflow=False
                        )
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                        self.last_audio = audio_array
                        self.audioReady.emit(audio_array)

                        # Calculate level
                        level = float(np.abs(audio_array).mean())
                        self.audioLevelReady.emit(min(1.0, level * 10))
                    except Exception as e:
                        logger.debug(f"Audio read error: {e}")
                else:
                    # Generate synthetic audio for testing
                    audio = self._generate_synthetic_audio()
                    self.last_audio = audio
                    self.audioReady.emit(audio)
                    level = float(np.abs(audio).mean())
                    self.audioLevelReady.emit(min(1.0, level * 10))

                # Sleep to maintain frame rate
                self.msleep(frame_interval_ms)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                self.msleep(100)

        # Cleanup
        self._cleanup()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic test frame when no camera."""
        import time
        t = time.time()

        frame = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)

        # Animated gradient
        for y in range(self.FRAME_HEIGHT):
            for x in range(self.FRAME_WIDTH):
                frame[y, x, 0] = int(127 + 127 * np.sin(t + x / 50))  # R
                frame[y, x, 1] = int(127 + 127 * np.sin(t + y / 50))  # G
                frame[y, x, 2] = int(127 + 127 * np.cos(t))  # B

        return frame

    def _generate_synthetic_audio(self) -> np.ndarray:
        """Generate synthetic test audio when no microphone."""
        import time
        t = time.time()

        # Generate a simple sine wave with noise
        samples = np.arange(self.AUDIO_CHUNK_SIZE)
        freq = 440  # Hz
        audio = 0.1 * np.sin(2 * np.pi * freq * samples / self.AUDIO_SAMPLE_RATE + t)
        audio += 0.02 * np.random.randn(self.AUDIO_CHUNK_SIZE)

        return audio.astype(np.float32)

    def _cleanup(self):
        """Clean up resources."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None

        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self._pa.terminate()
            except Exception:
                pass
            self._audio_stream = None

    def stop(self):
        """Stop the capture thread."""
        self._running = False
