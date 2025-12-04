"""
Input Panel Widget - Webcam and Microphone Display

Shows live webcam feed and microphone level indicator.
AV data is fed INTO Atlas for processing (not just display).
"""

import numpy as np
import logging
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QFrame,
        QHBoxLayout, QProgressBar
    )
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt, pyqtSlot
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QFrame,
        QHBoxLayout, QProgressBar
    )
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtCore import Qt, pyqtSlot
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class InputPanel(QWidget):
    """
    Input Panel - Webcam Feed and Microphone Display.

    Shows:
    - Live webcam feed (fed into Atlas)
    - Microphone level indicator
    - Processing status
    """

    WEBCAM_WIDTH = 320
    WEBCAM_HEIGHT = 240

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Webcam section
        webcam_title = QLabel("Webcam Input")
        webcam_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #64b5f6;")
        layout.addWidget(webcam_title)

        # Webcam frame
        self.webcam_frame = QFrame()
        self.webcam_frame.setFixedSize(self.WEBCAM_WIDTH + 4, self.WEBCAM_HEIGHT + 4)
        self.webcam_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #64b5f6;
                border-radius: 3px;
            }
        """)

        self.webcam_label = QLabel(self.webcam_frame)
        self.webcam_label.setFixedSize(self.WEBCAM_WIDTH, self.WEBCAM_HEIGHT)
        self.webcam_label.move(2, 2)
        self.webcam_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setText("No Camera")
        self.webcam_label.setStyleSheet("color: #666;")

        layout.addWidget(self.webcam_frame)

        # Processing indicator
        self.processing_label = QLabel("PROCESSING")
        self.processing_label.setStyleSheet("""
            QLabel {
                background-color: #1565c0;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        self.processing_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.processing_label)

        # Microphone section
        layout.addSpacing(10)
        mic_title = QLabel("Microphone Input")
        mic_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #81c784;")
        layout.addWidget(mic_title)

        # Mic level
        mic_layout = QHBoxLayout()
        mic_icon = QLabel("MIC")
        mic_icon.setStyleSheet("""
            QLabel {
                background-color: #c62828;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        mic_layout.addWidget(mic_icon)

        self.audio_level = QProgressBar()
        self.audio_level.setRange(0, 100)
        self.audio_level.setValue(0)
        self.audio_level.setTextVisible(False)
        self.audio_level.setFixedHeight(20)
        self.audio_level.setStyleSheet("""
            QProgressBar {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50, stop:0.6 #ffeb3b, stop:1 #f44336
                );
            }
        """)
        mic_layout.addWidget(self.audio_level)

        layout.addLayout(mic_layout)

        # Waveform display
        self.waveform_label = QLabel()
        self.waveform_label.setFixedSize(self.WEBCAM_WIDTH, 60)
        self.waveform_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.waveform_label)

        layout.addStretch()

    @pyqtSlot(object)
    def update_webcam(self, frame: np.ndarray):
        """Update the webcam display."""
        try:
            if frame is None or frame.size == 0:
                return

            # Convert from GPU if needed
            if hasattr(frame, 'get'):
                frame = frame.get()

            # Resize
            from PIL import Image
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.WEBCAM_WIDTH, self.WEBCAM_HEIGHT))
            frame = np.array(img)

            # Convert to QImage
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) > 2 else 1

            if channels == 1:
                bytes_per_line = width
                format_type = QImage.Format_Grayscale8 if PYQT_VERSION == 5 else QImage.Format.Format_Grayscale8
            else:
                bytes_per_line = channels * width
                format_type = QImage.Format_RGB888 if PYQT_VERSION == 5 else QImage.Format.Format_RGB888

            frame = np.ascontiguousarray(frame)
            q_image = QImage(frame.data, width, height, bytes_per_line, format_type)

            pixmap = QPixmap.fromImage(q_image.copy())
            self.webcam_label.setPixmap(pixmap)

        except Exception as e:
            logger.error(f"Error updating webcam: {e}")

    @pyqtSlot(float)
    def update_audio_level(self, level: float):
        """Update the audio level indicator (0.0 to 1.0)."""
        self.audio_level.setValue(int(min(100, max(0, level * 100))))

    @pyqtSlot(object)
    def update_waveform(self, audio_data: np.ndarray):
        """Update the waveform visualization."""
        try:
            width = self.WEBCAM_WIDTH
            height = 60

            # Create waveform image
            waveform_img = np.zeros((height, width, 3), dtype=np.uint8)
            waveform_img[:, :] = [26, 26, 26]

            if audio_data is not None and len(audio_data) > 0:
                if hasattr(audio_data, 'get'):
                    audio_data = audio_data.get()

                # Resample
                indices = np.linspace(0, len(audio_data) - 1, width).astype(int)
                samples = audio_data[indices]

                # Normalize
                samples = np.clip(samples, -1, 1)
                y_values = ((samples + 1) * height / 2).astype(int)

                # Draw
                for x, y in enumerate(y_values):
                    y = np.clip(y, 0, height - 1)
                    waveform_img[y, x] = [0, 255, 0]

            # Convert to QImage
            bytes_per_line = 3 * width
            q_image = QImage(
                np.ascontiguousarray(waveform_img).data,
                width, height, bytes_per_line,
                QImage.Format_RGB888 if PYQT_VERSION == 5 else QImage.Format.Format_RGB888
            )

            pixmap = QPixmap.fromImage(q_image.copy())
            self.waveform_label.setPixmap(pixmap)

        except Exception as e:
            logger.error(f"Error updating waveform: {e}")

    def set_processing_active(self, active: bool):
        """Set processing indicator state."""
        if active:
            self.processing_label.setStyleSheet("""
                QLabel {
                    background-color: #1565c0;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                }
            """)
            self.processing_label.setText("PROCESSING")
        else:
            self.processing_label.setStyleSheet("""
                QLabel {
                    background-color: #424242;
                    color: #888;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
            """)
            self.processing_label.setText("IDLE")
