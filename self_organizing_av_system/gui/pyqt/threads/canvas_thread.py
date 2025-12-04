"""
Canvas Thread - Creative Canvas Generation

Background thread for generating Atlas's 512x512 creative output.
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


class CanvasThread(QThread):
    """
    Background thread for generating creative canvas output.

    Atlas autonomously decides what to display on the 512x512 canvas.
    Emits signal when new canvas is ready.
    """

    canvasReady = pyqtSignal(object)  # numpy array (512, 512, 3) RGB uint8

    TARGET_FPS = 5  # Low FPS to prevent GPU-CPU transfer blocking UI

    def __init__(self, canvas_controller, parent=None):
        super().__init__(parent)
        self.canvas_controller = canvas_controller
        self._running = False

    def run(self):
        """Main canvas generation loop."""
        self._running = True
        frame_interval_ms = 1000 // self.TARGET_FPS

        logger.info("Canvas thread started")

        while self._running:
            try:
                # Generate new canvas (pure learned pixels, no modes)
                canvas = self.canvas_controller.generate_canvas()

                # Emit signal
                self.canvasReady.emit(canvas)

                # Sleep to maintain frame rate
                self.msleep(frame_interval_ms)

            except Exception as e:
                logger.error(f"Canvas generation error: {e}")
                self.msleep(100)

        logger.info("Canvas thread stopped")

    def stop(self):
        """Stop the canvas thread."""
        self._running = False
