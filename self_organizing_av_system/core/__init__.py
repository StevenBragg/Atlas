# Import core components
from .multimodal_association import MultimodalAssociation, AssociationMode
from .temporal_prediction import TemporalPrediction, PredictionMode
from .stability import StabilityMechanisms, InhibitionStrategy
from .structural_plasticity import StructuralPlasticity, PlasticityMode
from .system_config import SystemConfig
from .pathway import NeuralPathway

# Import superintelligence components
from .episodic_memory import EpisodicMemory, Episode
from .semantic_memory import SemanticMemory, Concept, Relation, RelationType
from .meta_learning import MetaLearner, LearningStrategy, LearningExperience
from .goal_planning import GoalPlanningSystem, Goal, GoalType, Action, Plan
from .world_model import WorldModel, Variable, CausalEdge, CausalRelationType, Object
from .symbolic_reasoning import SymbolicReasoner, Symbol, Proposition, Rule, LogicType

# Keep this for reference but mark as legacy/deprecated
from .multimodal import LegacyMultimodalAssociation

# Define package version
__version__ = "0.2.0"  # Updated for superintelligence capabilities

# Define public API
__all__ = [
    # Core self-organizing components
    'MultimodalAssociation',
    'AssociationMode',
    'TemporalPrediction',
    'PredictionMode',
    'StabilityMechanisms',
    'InhibitionStrategy',
    'StructuralPlasticity',
    'PlasticityMode',
    'SystemConfig',
    'NeuralPathway',

    # Superintelligence components
    'EpisodicMemory',
    'Episode',
    'SemanticMemory',
    'Concept',
    'Relation',
    'RelationType',
    'MetaLearner',
    'LearningStrategy',
    'LearningExperience',
    'GoalPlanningSystem',
    'Goal',
    'GoalType',
    'Action',
    'Plan',
    'WorldModel',
    'Variable',
    'CausalEdge',
    'CausalRelationType',
    'Object',
    'SymbolicReasoner',
    'Symbol',
    'Proposition',
    'Rule',
    'LogicType',
] 