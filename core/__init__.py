"""
ATLAS Core Superintelligence Modules

This package contains the core cognitive architecture modules for ATLAS:

1. Recursive Self-Improvement (self_improvement.py)
   - Code analysis and bottleneck detection
   - Automated improvement generation
   - Safe testing and validation

2. Knowledge Integration (knowledge_integration.py)
   - Multi-modal memory unification
   - Cross-modal learning
   - Memory consolidation

3. Autonomous Goals (autonomous_goals.py)
   - Intrinsic motivation system
   - Goal generation and management
   - Curiosity-driven exploration

4. Meta-Cognition (meta_cognition.py)
   - Learning process monitoring
   - Confusion and stuck detection
   - Strategy adaptation
"""

from .self_improvement import (
    RecursiveSelfImprovement,
    ImprovementType,
    SafetyLevel,
    ModificationStatus,
    PerformanceMetric,
    CodeModification,
    CodeMetrics,
    PerformanceBottleneck,
    Capability,
)

from .knowledge_integration import (
    KnowledgeIntegrationSystem,
    MemoryType,
    ModalityType,
    IntegrationStrategy,
    MemoryTrace,
    CrossModalAssociation,
    KnowledgeNode,
    ProceduralSkill,
)

from .autonomous_goals import (
    AutonomousGoalSystem,
    GoalType,
    GoalStatus,
    IntrinsicDrive,
    Goal,
    DriveState,
    ExplorationFrontier,
)

from .meta_cognition import (
    MetaCognitiveMonitor,
    CognitiveState,
    LearningStrategy,
    UncertaintyType,
    LearningEpisode,
    StrategyEffectiveness,
    ConfusionEvent,
    MetaCognitiveAssessment,
)

__all__ = [
    # Self-Improvement
    'RecursiveSelfImprovement',
    'ImprovementType',
    'SafetyLevel',
    'ModificationStatus',
    'PerformanceMetric',
    'CodeModification',
    'CodeMetrics',
    'PerformanceBottleneck',
    'Capability',
    
    # Knowledge Integration
    'KnowledgeIntegrationSystem',
    'MemoryType',
    'ModalityType',
    'IntegrationStrategy',
    'MemoryTrace',
    'CrossModalAssociation',
    'KnowledgeNode',
    'ProceduralSkill',
    
    # Autonomous Goals
    'AutonomousGoalSystem',
    'GoalType',
    'GoalStatus',
    'IntrinsicDrive',
    'Goal',
    'DriveState',
    'ExplorationFrontier',
    
    # Meta-Cognition
    'MetaCognitiveMonitor',
    'CognitiveState',
    'LearningStrategy',
    'UncertaintyType',
    'LearningEpisode',
    'StrategyEffectiveness',
    'ConfusionEvent',
    'MetaCognitiveAssessment',
]
