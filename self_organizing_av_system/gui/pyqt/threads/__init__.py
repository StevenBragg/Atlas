"""
Background threads for Atlas GUI

- CaptureThread: Webcam and microphone capture
- ProcessingThread: Challenge processing and learning
- CanvasThread: Creative canvas generation
- AVProcessingThread: Feeds AV into Atlas
"""

from .capture_thread import CaptureThread
from .processing_thread import ProcessingThread
from .canvas_thread import CanvasThread

__all__ = [
    'CaptureThread',
    'ProcessingThread',
    'CanvasThread',
]
