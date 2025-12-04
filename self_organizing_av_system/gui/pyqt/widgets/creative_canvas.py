"""
Creative Canvas Widget - Atlas's 512x512 RGB Creative Output

This is Atlas's "creative outlet" - it decides what to display.
The canvas shows Atlas's internal state, imagination, learning progress,
or abstract art based on Atlas's autonomous choices.
"""

import numpy as np
import logging
from typing import Optional

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt, pyqtSlot
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtCore import Qt, pyqtSlot
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class CreativeCanvasWidget(QWidget):
    """
    512x512 RGB Canvas - Atlas's Creative Output.

    Atlas autonomously decides what to display based on:
    - Internal neural state
    - Learning activity
    - Creativity level
    - Memory activations
    """

    CANVAS_SIZE = 512

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_image = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Title
        title = QLabel("Atlas Creative Canvas")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ce93d8;")
        title.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Canvas frame
        self.canvas_frame = QFrame()
        self.canvas_frame.setFixedSize(self.CANVAS_SIZE + 4, self.CANVAS_SIZE + 4)
        self.canvas_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 2px solid #ce93d8;
                border-radius: 3px;
            }
        """)

        # Canvas label
        self.canvas_label = QLabel(self.canvas_frame)
        self.canvas_label.setFixedSize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.canvas_label.move(2, 2)
        self.canvas_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)

        # Initialize with blank canvas
        self._set_blank_canvas()

        layout.addWidget(self.canvas_frame, alignment=Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)

        # Mode indicator
        self.mode_label = QLabel("Mode: Initializing...")
        self.mode_label.setStyleSheet("color: #888; font-size: 11px;")
        self.mode_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mode_label)

    def _set_blank_canvas(self):
        """Initialize with a blank canvas."""
        blank = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)
        blank[:, :] = [30, 30, 30]  # Dark gray
        self.update_canvas(blank)

    @pyqtSlot(object)
    def update_canvas(self, rgb_array: np.ndarray):
        """
        Update the canvas with a new RGB array.

        Args:
            rgb_array: numpy array of shape (512, 512, 3) with dtype uint8
        """
        try:
            # Ensure numpy array
            if hasattr(rgb_array, 'get'):
                rgb_array = rgb_array.get()

            # Ensure correct shape
            if rgb_array.shape[:2] != (self.CANVAS_SIZE, self.CANVAS_SIZE):
                from PIL import Image
                img = Image.fromarray(rgb_array.astype(np.uint8))
                img = img.resize((self.CANVAS_SIZE, self.CANVAS_SIZE))
                rgb_array = np.array(img)

            # Ensure uint8
            if rgb_array.dtype != np.uint8:
                if rgb_array.max() <= 1.0:
                    rgb_array = (rgb_array * 255).astype(np.uint8)
                else:
                    rgb_array = rgb_array.astype(np.uint8)

            # Ensure RGB (3 channels)
            if len(rgb_array.shape) == 2:
                rgb_array = np.stack([rgb_array, rgb_array, rgb_array], axis=2)
            elif rgb_array.shape[2] == 4:
                rgb_array = rgb_array[:, :, :3]

            # Convert to QImage
            height, width, channels = rgb_array.shape
            bytes_per_line = channels * width

            rgb_array = np.ascontiguousarray(rgb_array)

            q_image = QImage(
                rgb_array.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888 if PYQT_VERSION == 5 else QImage.Format.Format_RGB888
            )

            # Store copy
            self._current_image = q_image.copy()

            # Update display
            pixmap = QPixmap.fromImage(self._current_image)
            self.canvas_label.setPixmap(pixmap)

        except Exception as e:
            logger.error(f"Error updating canvas: {e}")

    def set_mode(self, mode_name: str):
        """Update the mode indicator."""
        self.mode_label.setText(f"Mode: {mode_name}")
