# Import core components
from .multimodal_association import MultimodalAssociation, AssociationMode
from .temporal_prediction import TemporalPrediction, PredictionMode
from .stability import StabilityMechanisms, InhibitionStrategy
from .structural_plasticity import StructuralPlasticity, PlasticityMode
from .system_config import SystemConfig
from .pathway import NeuralPathway

# Keep this for reference but mark as legacy/deprecated
from .multimodal import LegacyMultimodalAssociation

# Define package version
__version__ = "0.1.0"

# Define public API
__all__ = [
    'MultimodalAssociation',
    'AssociationMode',
    'TemporalPrediction', 
    'PredictionMode',
    'StabilityMechanisms',
    'InhibitionStrategy',
    'StructuralPlasticity',
    'PlasticityMode',
    'SystemConfig',
    'NeuralPathway'
] 