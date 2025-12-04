"""
PyQt5 Widgets for Atlas GUI

Curriculum Tab Widgets:
- CurriculumSchoolPanel: Main school interface
- LevelProgressPanel: Level progression display
- ChallengeDisplayPanel: Current challenge details

Free Play Tab Widgets:
- ChatPanel: Chat interface with Atlas
- CreativeCanvasWidget: 512x512 RGB creative canvas
- InputPanel: Webcam and microphone display

Shared Widgets:
- NetworkVizPanel: Neural network visualization
- KnowledgeBasePanel: Episodic + Semantic memory display
"""

from .chat_panel import ChatPanel
from .creative_canvas import CreativeCanvasWidget
from .input_panel import InputPanel
from .curriculum_school_panel import CurriculumSchoolPanel
from .level_progress_panel import LevelProgressPanel
from .network_viz_panel import NetworkVizPanel
from .knowledge_base_panel import KnowledgeBasePanel

__all__ = [
    'ChatPanel',
    'CreativeCanvasWidget',
    'InputPanel',
    'CurriculumSchoolPanel',
    'LevelProgressPanel',
    'NetworkVizPanel',
    'KnowledgeBasePanel',
]
