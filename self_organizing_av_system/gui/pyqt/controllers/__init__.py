"""
Controllers for Atlas GUI

- AtlasController: Unified brain controller (shared between tabs)
- CurriculumController: Manages curriculum/school progression
- KnowledgeBaseController: Memory system access
"""

from .atlas_controller import AtlasController

__all__ = [
    'AtlasController',
]
