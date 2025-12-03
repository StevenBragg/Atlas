# Import GPU backend (CuPy or NumPy fallback)
from .backend import xp, HAS_GPU, to_cpu, to_gpu, get_backend_info, sync

# Import core components
from .multimodal_association import MultimodalAssociation, AssociationMode
from .temporal_prediction import TemporalPrediction, PredictionMode
from .stability import StabilityMechanisms, InhibitionStrategy
from .structural_plasticity import StructuralPlasticity, PlasticityMode
from .system_config import SystemConfig
from .pathway import NeuralPathway

# Import image classification components
from .image_classification import (
    ImageClassificationLearner,
    ImageDataset,
    ClassificationResult,
    LearningMetrics,
    ClassPrototypeLayer,
    quick_train_classifier,
)

# Import language grounding components
from .language_grounding import (
    LanguageGrounding,
    TextCorpusLearner,
    GroundedWord,
    ParsedSentence,
    WordType,
)

# Import working memory and attention components
from .working_memory import (
    WorkingMemory,
    AttentionController,
    CognitiveController,
    WorkspaceItem,
    WorkspaceSlotType,
    AttentionType,
)

# Import causal reasoning components
from .causal_reasoning import (
    CausalReasoner,
    CausalModel,
    CausalGraph,
    WorldModel,
    CausalLink,
    CausalVariable,
    CounterfactualQuery,
)

# Import abstract reasoning components
from .abstract_reasoning import (
    AbstractReasoner,
    KnowledgeBase,
    LogicEngine,
    AnalogyEngine,
    PatternDetector,
    RuleInducer,
    Proposition,
    Rule,
    Analogy,
    Pattern,
)

# Import symbolic reasoning components
from .symbolic_reasoning import (
    SymbolicReasoner,
    Symbol,
    LogicType,
)

# Import legacy world model (use causal_reasoning WorldModel for new code)
from .world_model import (
    Variable as WorldModelVariable,
    CausalEdge as WorldModelCausalEdge,
    WorldObject as WorldModelObject,
)

# Keep this for reference but mark as legacy/deprecated
from .multimodal import LegacyMultimodalAssociation

# Import challenge-based learning components
from .challenge import (
    Challenge,
    ChallengeType,
    ChallengeStatus,
    Modality,
    TrainingData,
    SuccessCriteria,
    LearningResult,
    LearnedCapability,
    ProgressReport,
)
from .challenge_parser import ChallengeParser
from .challenge_learner import ChallengeLearner, learn_challenge
from .progress_tracker import ProgressTracker
from .learning_engine import LearningEngine

# Define package version
__version__ = "0.3.0"

# Define public API
__all__ = [
    # GPU Backend
    'xp',
    'HAS_GPU',
    'to_cpu',
    'to_gpu',
    'get_backend_info',
    'sync',
    # Core components
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
    # Image classification
    'ImageClassificationLearner',
    'ImageDataset',
    'ClassificationResult',
    'LearningMetrics',
    'ClassPrototypeLayer',
    'quick_train_classifier',
    # Language grounding
    'LanguageGrounding',
    'TextCorpusLearner',
    'GroundedWord',
    'ParsedSentence',
    'WordType',
    # Working memory and attention
    'WorkingMemory',
    'AttentionController',
    'CognitiveController',
    'WorkspaceItem',
    'WorkspaceSlotType',
    'AttentionType',
    # Causal reasoning
    'CausalReasoner',
    'CausalModel',
    'CausalGraph',
    'WorldModel',
    'CausalLink',
    'CausalVariable',
    'CounterfactualQuery',
    # Abstract reasoning
    'AbstractReasoner',
    'KnowledgeBase',
    'LogicEngine',
    'AnalogyEngine',
    'PatternDetector',
    'RuleInducer',
    'Proposition',
    'Rule',
    'Analogy',
    'Pattern',
    # Symbolic reasoning
    'SymbolicReasoner',
    'Symbol',
    'LogicType',
    # Legacy world model components
    'WorldModelVariable',
    'WorldModelCausalEdge',
    'WorldModelObject',
    # Challenge-based learning
    'Challenge',
    'ChallengeType',
    'ChallengeStatus',
    'Modality',
    'TrainingData',
    'SuccessCriteria',
    'LearningResult',
    'LearnedCapability',
    'ProgressReport',
    'ChallengeParser',
    'ChallengeLearner',
    'learn_challenge',
    'ProgressTracker',
    'LearningEngine',
] 