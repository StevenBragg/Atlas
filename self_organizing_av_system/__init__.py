from .models.visual.processor import VisualProcessor
from .models.audio.processor import AudioProcessor
from .models.multimodal.system import SelfOrganizingAVSystem
from .utils.capture import AVCapture, VideoFileReader
from .gui.tk_monitor import TkMonitor
from .config.configuration import SystemConfig

__version__ = "0.1.0"
__all__ = [
    'VisualProcessor',
    'AudioProcessor',
    'SelfOrganizingAVSystem',
    'AVCapture',
    'VideoFileReader',
    'TkMonitor',
    'SystemConfig'
] 