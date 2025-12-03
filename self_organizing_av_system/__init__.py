"""
ATLAS: Self-Organizing Audio-Visual Learning System

A unified superintelligence framework integrating cognitive systems for
autonomous learning without supervision, labels, or backpropagation.
"""

__version__ = "0.2.0"

# Core components (always available)
try:
    from .models.visual.processor import VisualProcessor
    from .models.audio.processor import AudioProcessor
    from .models.multimodal.system import SelfOrganizingAVSystem
    from .config.configuration import SystemConfig
except ImportError:
    # Core dependencies not installed
    VisualProcessor = None
    AudioProcessor = None
    SelfOrganizingAVSystem = None
    SystemConfig = None

# Utility components (may require additional dependencies)
try:
    from .utils.capture import AVCapture, VideoFileReader
except ImportError:
    AVCapture = None
    VideoFileReader = None

# GUI components (optional, requires tkinter)
try:
    from .gui.tk_monitor import TkMonitor
except ImportError:
    TkMonitor = None

# Unified Super Intelligence (core cognitive systems)
try:
    from .core.unified_intelligence import UnifiedSuperIntelligence, IntelligenceMode
except ImportError:
    UnifiedSuperIntelligence = None
    IntelligenceMode = None

__all__ = [
    'VisualProcessor',
    'AudioProcessor',
    'SelfOrganizingAVSystem',
    'AVCapture',
    'VideoFileReader',
    'TkMonitor',
    'SystemConfig',
    'UnifiedSuperIntelligence',
    'IntelligenceMode',
] 