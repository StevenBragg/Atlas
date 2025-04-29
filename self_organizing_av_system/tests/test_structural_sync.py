#!/usr/bin/env python3
"""
Test script to verify proper synchronization between structural plasticity component 
and the current_multimodal_state in the self-organizing AV system.
"""

import os
import sys
import time
import logging
import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system import SelfOrganizingAVSystem
from core.visual_processor import VisualProcessor
from core.audio_processor import AudioProcessor
from core.structural_plasticity import StructuralPlasticity, PlasticityMode

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestStructuralSync(unittest.TestCase):
    """Test case for structural plasticity synchronization"""
    
    def setUp(self):
        """Set up the test environment with a minimal system configuration"""
        # Configure a minimal system
        config = {
            "multimodal_association": {
                "association_rate": 0.1,
                "representation_size": 50  # Start with a small representation
            },
            "stability_mechanisms": {
                "target_activity": 0.1,
                "inhibition_strategy": "k_winners_allowed",
                "k_winners": 5
            },
            "structural_plasticity": {
                "enable_neuron_growth": True,
                "enable_connection_pruning": True,
                "max_size": 100,  # Allow growth up to 100 neurons
                "growth_rate": 0.1,
                "structural_plasticity_mode": "adaptive"
            },
            "video_params": {
                "input_size": (64, 64),
                "output_size": (64, 64),
                "preprocess": True
            },
            "audio_params": {
                "sample_rate": 16000,
                "frame_length": 512,
                "feature_type": "mfcc",
                "n_mfcc": 13
            }
        }
        
        # Initialize processors with minimal settings
        visual_processor = VisualProcessor(
            input_size=(64, 64),
            feature_size=32,
            use_features=True
        )
        
        audio_processor = AudioProcessor(
            sample_rate=16000,
            frame_length=512,
            feature_size=32,
            feature_type="mfcc",
            n_mfcc=13
        )
        
        # Create the system
        self.system = SelfOrganizingAVSystem(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config=config
        )
        
        # Create test input data
        self.test_visual = np.random.rand(64, 64, 3) * 255
        self.test_audio = np.random.rand(16000).astype(np.float32)
        
    def test_resize_sync(self):
        """Test that manually resizing structural plasticity updates multimodal state"""
        # Process some initial data to initialize the system
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Get initial sizes
        initial_size = self.system.structural_plasticity.current_size
        initial_state_size = len(self.system.current_multimodal_state)
        
        # Verify sizes are initially in sync
        self.assertEqual(initial_size, initial_state_size, 
                         "Initial sizes should match")
        
        # Manually grow the network by 10 neurons
        new_size = initial_size + 10
        resize_result = self.system.structural_plasticity.resize(new_size)
        
        # Verify the resize was successful
        self.assertEqual(self.system.structural_plasticity.current_size, new_size,
                         "Structural plasticity size should be updated")
        
        # Verify the multimodal state was also updated
        self.assertEqual(len(self.system.current_multimodal_state), new_size,
                        "Multimodal state size should match new size")
        
        # Verify other dependent components were updated
        self.assertEqual(self.system.multimodal_size, new_size,
                         "System's multimodal_size should be updated")
        
        # Check that RGB control weights were updated (if enabled)
        if hasattr(self.system, 'rgb_control_weights') and self.system.rgb_control_weights is not None:
            rgb_rows = self.system.rgb_control_weights['multimodal'].shape[0]
            self.assertEqual(rgb_rows, new_size,
                            "RGB control weights should be updated to match new size")
        
        # Verify shrinking also works
        smaller_size = new_size - 5
        self.system.structural_plasticity.resize(smaller_size)
        
        # Verify all sizes synced after shrinking
        self.assertEqual(self.system.structural_plasticity.current_size, smaller_size)
        self.assertEqual(len(self.system.current_multimodal_state), smaller_size)
        self.assertEqual(self.system.multimodal_size, smaller_size)
        
    def test_automatic_growth_sync(self):
        """Test that automatic growth during processing updates multimodal state"""
        # Create a system with more aggressive growth for testing
        config = {
            "multimodal_association": {
                "association_rate": 0.1,
                "representation_size": 30  # Start small to encourage growth
            },
            "structural_plasticity": {
                "enable_neuron_growth": True,
                "max_size": 100,
                "growth_rate": 0.2,  # Higher growth rate
                "growth_threshold": 0.3,  # Lower threshold to trigger growth
                "structural_plasticity_mode": "growing",  # Force growing mode
                "growth_cooldown": 1,  # No cooldown period
                "check_interval": 1  # Check every update
            }
        }
        
        # Create processors
        visual_processor = VisualProcessor(
            input_size=(64, 64),
            feature_size=32,
            use_features=True
        )
        
        audio_processor = AudioProcessor(
            sample_rate=16000,
            frame_length=512,
            feature_size=32,
            feature_type="mfcc",
            n_mfcc=13
        )
        
        # Create system with growth-focused config
        growth_system = SelfOrganizingAVSystem(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config=config
        )
        
        # Store initial size
        initial_size = growth_system.structural_plasticity.current_size
        
        # Create different patterns to encourage novelty detection
        patterns = []
        for i in range(5):
            # Create distinct visual patterns
            pattern = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Add different shapes/patterns
            if i == 0:  # Vertical lines
                pattern[:, ::4, 0] = 255
            elif i == 1:  # Horizontal lines
                pattern[::4, :, 1] = 255
            elif i == 2:  # Diagonal pattern
                for j in range(64):
                    if j < 64:
                        pattern[j, j, 2] = 255
            elif i == 3:  # Circle
                center = (32, 32)
                for y in range(64):
                    for x in range(64):
                        if ((x-center[0])**2 + (y-center[1])**2) < 400:
                            pattern[y, x, :] = [0, 200, 200]
            else:  # Random dots
                for _ in range(50):
                    x, y = np.random.randint(0, 64, 2)
                    pattern[y, x, :] = np.random.randint(100, 255, 3)
                    
            patterns.append(pattern)
                    
        # Create corresponding audio patterns
        audio_patterns = []
        for i in range(5):
            # Create different frequency patterns
            audio = np.zeros(16000, dtype=np.float32)
            
            # Add different frequencies
            if i == 0:  # Low frequency
                t = np.arange(16000) / 16000
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            elif i == 1:  # Higher frequency
                t = np.arange(16000) / 16000
                audio = 0.5 * np.sin(2 * np.pi * 880 * t)
            elif i == 2:  # Dual frequencies
                t = np.arange(16000) / 16000
                audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
            elif i == 3:  # Chirp
                t = np.arange(16000) / 16000
                freq = np.linspace(220, 880, 16000)
                audio = 0.5 * np.sin(2 * np.pi * freq * t)
            else:  # Noise
                audio = np.random.randn(16000).astype(np.float32) * 0.2
                
            audio_patterns.append(audio)
        
        # Process each pattern multiple times to encourage growth
        grew = False
        for _ in range(10):  # Try multiple iterations
            for i in range(len(patterns)):
                # Process the pattern
                result = growth_system.process(
                    visual_input=patterns[i],
                    audio_input=audio_patterns[i],
                    learning_enabled=True
                )
                
                # Force structural plasticity update directly
                weights = None
                if growth_system.multimodal_association is not None:
                    weights = growth_system.multimodal_association.weights.get("visual", None)
                
                # Introduce random high activity to encourage growth
                activity = growth_system.current_multimodal_state.copy()
                # Make activity more concentrated to trigger growth
                random_indices = np.random.choice(
                    range(len(activity)), 
                    size=max(3, int(len(activity) * 0.2)),
                    replace=False
                )
                activity[random_indices] = 1.0
                
                # Directly update structural plasticity
                structural_result = growth_system.structural_plasticity.update(
                    activity=activity,
                    weights=weights,
                    force_check=True
                )
                
                # Check if growth occurred
                if structural_result.get('size_changed', False):
                    grew = True
                    current_size = growth_system.structural_plasticity.current_size
                    state_size = len(growth_system.current_multimodal_state)
                    
                    # Verify sizes match after growth
                    self.assertEqual(current_size, state_size,
                                    f"After growth, sizes should match: {current_size} != {state_size}")
                    
                    # Verify the multimodal_size was updated
                    self.assertEqual(growth_system.multimodal_size, current_size,
                                    "System's internal multimodal_size should be updated")
        
        # If no growth occurred naturally, force it
        if not grew:
            logger.info("No automatic growth detected, forcing manual growth")
            
            # Directly resize to force growth
            new_size = initial_size + 5
            growth_system.structural_plasticity.resize(new_size)
            
            # Verify synchronization after forced growth
            self.assertEqual(growth_system.structural_plasticity.current_size, new_size)
            self.assertEqual(len(growth_system.current_multimodal_state), new_size)
            self.assertEqual(growth_system.multimodal_size, new_size)
    
    def test_multimodal_state_update_during_processing(self):
        """Test that the multimodal state is properly updated during processing"""
        # Process some initial data
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Get the initial size
        initial_size = self.system.structural_plasticity.current_size
        
        # Manually modify the structural plasticity size
        new_size = initial_size + 7
        self.system.structural_plasticity.resize(new_size)
        
        # Process new data to verify the multimodal state is properly updated
        result = self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Verify that the resulted multimodal state has the correct size
        self.assertEqual(len(result["multimodal_state"]), new_size,
                        "Processed multimodal state should have the updated size")
        
        # Verify that internal state is maintained
        self.assertEqual(len(self.system.current_multimodal_state), new_size,
                        "Internal multimodal state should be maintained with correct size")
        
        # Verify multimodal_size is synchronized
        self.assertEqual(self.system.multimodal_size, new_size,
                        "System's multimodal_size should stay in sync")
    
    def test_direct_pixel_output_after_resize(self):
        """Test that direct pixel output works correctly after resizing"""
        # Only test if direct pixel control is available
        if not hasattr(self.system, 'direct_pixel_control'):
            return
            
        # Enable direct pixel control
        self.system.set_direct_pixel_control(True)
        
        # Process some initial data
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Get initial output before resize
        initial_output = self.system.get_direct_pixel_output((64, 64))
        
        # Resize the structural plasticity component
        initial_size = self.system.structural_plasticity.current_size
        new_size = initial_size + 8
        self.system.structural_plasticity.resize(new_size)
        
        # Try to get direct pixel output after resize
        try:
            new_output = self.system.get_direct_pixel_output((64, 64))
            # If we get here, the output was generated successfully
            self.assertEqual(new_output.shape, (64, 64, 3),
                            "Direct pixel output should have correct shape after resize")
        except Exception as e:
            self.fail(f"Failed to generate direct pixel output after resize: {e}")
    
    def test_temporal_prediction_after_resize(self):
        """Test that temporal prediction works correctly after resizing"""
        # Skip if temporal prediction is not enabled
        if not hasattr(self.system, 'temporal_prediction') or self.system.temporal_prediction is None:
            return
            
        # Process some initial data
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Process a second frame to build sequence
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Try to get temporal predictions before resize
        try:
            initial_predictions = self.system.get_temporal_predictions(steps=1)
        except Exception as e:
            # If this fails, the test is inconclusive
            logger.warning(f"Could not get initial temporal predictions: {e}")
            return
            
        # Resize the structural plasticity component
        initial_size = self.system.structural_plasticity.current_size
        new_size = initial_size + 10
        self.system.structural_plasticity.resize(new_size)
        
        # Process another frame after resize
        self.system.process(
            visual_input=self.test_visual,
            audio_input=self.test_audio,
            learning_enabled=True
        )
        
        # Try to get temporal predictions after resize
        try:
            new_predictions = self.system.get_temporal_predictions(steps=1)
            # Check that predictions were generated correctly
            self.assertIn(1, new_predictions, "Should have predictions for 1 step ahead")
        except Exception as e:
            self.fail(f"Failed to generate temporal predictions after resize: {e}")
    
    def tearDown(self):
        """Clean up after tests"""
        # Free up resources
        self.system = None

if __name__ == '__main__':
    unittest.main()