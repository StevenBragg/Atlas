#!/usr/bin/env python3
"""
Basic test to verify imports and initialization
"""

import os
import sys
import logging

# Add the parent directory to the path to allow importing the package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that we can import all the necessary modules"""
    logger.info("Testing basic imports...")
    
    try:
        # Test imports
        from self_organizing_av_system.models.visual.processor import VisualProcessor
        from self_organizing_av_system.models.audio.processor import AudioProcessor
        from self_organizing_av_system.core.system import SelfOrganizingAVSystem
        from self_organizing_av_system.core.structural_plasticity import StructuralPlasticity
        from self_organizing_av_system.core.multimodal_association import MultimodalAssociation
        from self_organizing_av_system.core.temporal_prediction import TemporalPrediction
        from self_organizing_av_system.core.stability import StabilityMechanisms
        
        logger.info("All modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import: {e}")
        return False

def test_basic_initialization():
    """Test basic initialization of components"""
    logger.info("Testing basic initialization...")
    
    try:
        # Import needed classes
        from self_organizing_av_system.models.visual.processor import VisualProcessor
        from self_organizing_av_system.models.audio.processor import AudioProcessor
        from self_organizing_av_system.core.system import SelfOrganizingAVSystem
        
        # Create minimal processors
        logger.info("Initializing visual processor...")
        visual_processor = VisualProcessor(
            input_width=32,
            input_height=32,
            use_grayscale=True,
            patch_size=8,
            stride=8,
            contrast_normalize=True,
            layer_sizes=[8, 6, 4]
        )
        logger.info("Visual processor initialized")
        
        logger.info("Initializing audio processor...")
        audio_processor = AudioProcessor(
            sample_rate=22050,
            window_size=256,
            hop_length=128,
            n_mels=12,
            min_freq=50,
            max_freq=8000,
            normalize=True,
            layer_sizes=[6, 5, 4]
        )
        logger.info("Audio processor initialized")
        
        # Create minimal system
        logger.info("Initializing system...")
        system = SelfOrganizingAVSystem(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config={
                "multimodal_size": 8,
                "learning_rate": 0.01
            }
        )
        logger.info("System initialized")
        
        # Verify components exist
        assert hasattr(system, 'multimodal_association'), "Multimodal association not initialized"
        assert hasattr(system, 'temporal_prediction'), "Temporal prediction not initialized"
        assert hasattr(system, 'stability'), "Stability mechanisms not initialized"
        assert hasattr(system, 'structural_plasticity'), "Structural plasticity not initialized"
        
        logger.info("All components verified")
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_basic_tests():
    """Run the basic tests"""
    logger.info("Starting basic tests...")
    
    import_result = test_basic_imports()
    if not import_result:
        logger.error("Import test failed")
        return
    
    init_result = test_basic_initialization()
    if not init_result:
        logger.error("Initialization test failed")
        return
    
    logger.info("All basic tests passed")

if __name__ == "__main__":
    run_basic_tests() 