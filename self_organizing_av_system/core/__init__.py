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

# Keep this for reference but mark as legacy/deprecated
from .multimodal import LegacyMultimodalAssociation

# Define package version
__version__ = "0.2.0"

# Define public API
__all__ = [
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
] 