import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import yaml
import cv2
import numpy as np

from .backend import xp, to_cpu

# Remove the imports of processor classes - they'll be imported inside the load_checkpoint method
# from ..models.visual.processor import VisualProcessor
# from ..models.audio.processor import AudioProcessor

from .multimodal_association import MultimodalAssociation
from .temporal_prediction import TemporalPrediction
from .stability import StabilityMechanisms, InhibitionStrategy
from .structural_plasticity import StructuralPlasticity, PlasticityMode

logger = logging.getLogger(__name__)


class SelfOrganizingAVSystem:
    """
    Self-Organizing Audio-Visual Learning System
    
    This system integrates multiple neural components to learn from synchronized
    audio-visual inputs without supervision, labels, or backpropagation.
    
    The system includes:
    1. Visual pathway for processing visual inputs
    2. Auditory pathway for processing audio inputs
    3. Cross-modal associations for binding modalities
    4. Temporal prediction for sequence learning
    5. Stability mechanisms for balanced learning
    6. Structural plasticity for adapting network structure
    
    Instead of using supervised learning or backpropagation, this system
    relies on local learning rules (Hebbian learning and variants),
    competitive learning, temporal prediction, and cross-modal binding.
    """
    
    def __init__(
        self,
        visual_processor, # Type: VisualProcessor
        audio_processor,  # Type: AudioProcessor
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the self-organizing system with sensory processors.
        
        Args:
            visual_processor: Visual pathway processor
            audio_processor: Audio pathway processor
            config: Configuration dictionary (optional)
        """
        self.visual_processor = visual_processor
        self.audio_processor = audio_processor
        
        # Set default configuration if not provided
        if config is None:
            config = {}
        
        # Get system-level parameters
        self.config = config
        
        # Set up logging
        self.log_level = config.get("log_level", logging.INFO)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Extract key parameters from config
        self.multimodal_size = config.get("multimodal_size", 128)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.learning_rule = config.get("learning_rule", "oja")  # Default to Oja's rule if not specified
        self.prune_interval = config.get("prune_interval", 100)
        self.structural_plasticity_interval = config.get("structural_plasticity_interval", 200)
        self.snapshot_interval = config.get("snapshot_interval", 1000)
        self.enable_learning = config.get("enable_learning", True)
        self.enable_visualization = config.get("enable_visualization", True)
        
        # Initialize model components
        self._init_multimodal_association()
        self._init_temporal_prediction()
        self._init_stability_mechanisms()
        self._init_structural_plasticity()
        
        # State tracking
        self.current_visual_encoding = None
        self.current_audio_encoding = None
        self.current_multimodal_state = None
        self.last_update_time = None
        self.frames_processed = 0
        self.system_ready = False
        
        # Performance tracking
        self.reconstruction_errors = {
            "visual": [],
            "audio": [],
            "cross_modal": [],
        }
        self.prediction_errors = []
        
        # Determine output sizes from actual processors
        self.visual_output_size = self.visual_processor.visual_pathway.layers[-1].layer_size
        self.audio_output_size = self.audio_processor.audio_pathway.layers[-1].layer_size
        
        logger.info(f"Initialized Self-Organizing AV System - "
                  f"multimodal_size: {self.multimodal_size}, "
                  f"visual_size: {self.visual_output_size}, "
                  f"audio_size: {self.audio_output_size}")
        
        self.system_ready = True
        
        self.video_params = {
            'grayscale': False,
            'contrast': 1.0,
            'filter_strength': 0.5,
            'output_size': (320, 240)
        }
        
        # Direct pixel control variables
        self.direct_pixel_control = False
        self.direct_pixel_output = None
        self.rgb_control_weights = None
        self.rgb_channel_bias = None
        self.rgb_channel_gain = None
        self.enhanced_rgb_control = config.get("enhanced_rgb_control", False)
        self._init_rgb_control()
        
        # Learning parameters for RGB control
        self.rgb_learning_rate = config.get("rgb_learning_rate", 0.005)
        self.rgb_learning_enabled = config.get("rgb_learning_enabled", True)
    
    def _init_multimodal_association(self):
        """Initialize multimodal association component"""
        # Extract relevant config
        association_config = self.config.get("multimodal_association", {})
        
        # Use default values if not specified
        association_mode = association_config.get("association_mode", "hebbian")
        normalization = association_config.get("normalization", "softmax")
        lateral_inhibition = association_config.get("lateral_inhibition", 0.2)
        
        # Determine modality sizes directly from the processor pathways
        modality_sizes = {
            "visual": self.visual_processor.visual_pathway.layers[-1].layer_size,
            "audio": self.audio_processor.audio_pathway.layers[-1].layer_size
        }
        
        # Initialize multimodal association
        self.multimodal_association = MultimodalAssociation(
            modality_sizes=modality_sizes,
            association_size=self.multimodal_size,
            learning_rate=self.learning_rate,
            association_mode=association_mode,
            normalization_mode=normalization,
            lateral_inhibition=lateral_inhibition,
            use_sparse_coding=association_config.get("use_sparse_coding", True),
            stability_threshold=association_config.get("stability_threshold", 0.1),
            enable_attention=association_config.get("enable_attention", True)
        )
        
        logger.info(f"Initialized multimodal association: "
                  f"association_size={self.multimodal_size}, "
                  f"mode={association_mode}")
    
    def _init_temporal_prediction(self):
        """Initialize temporal prediction component"""
        # Extract relevant config
        temporal_config = self.config.get("temporal_prediction", {})
        
        # Use default values if not specified
        prediction_mode = temporal_config.get("prediction_mode", "forward")
        sequence_length = temporal_config.get("sequence_length", 5)
        prediction_horizon = temporal_config.get("prediction_horizon", 3)
        
        # Initialize temporal prediction
        self.temporal_prediction = TemporalPrediction(
            representation_size=self.multimodal_size,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            learning_rate=self.learning_rate,
            prediction_mode=prediction_mode,
            use_eligibility_trace=temporal_config.get("use_eligibility_trace", True),
            enable_surprise_detection=temporal_config.get("enable_surprise_detection", True),
            enable_recurrent_connections=temporal_config.get("enable_recurrent_connections", True)
        )
        
        logger.info(f"Initialized temporal prediction: "
                  f"sequence_length={sequence_length}, "
                  f"prediction_horizon={prediction_horizon}, "
                  f"mode={prediction_mode}")
    
    def _init_stability_mechanisms(self):
        """Initialize stability mechanisms component"""
        # Extract relevant config
        stability_config = self.config.get("stability", {})
        
        # Use default values if not specified
        # Make sure inhibition_strategy is one of the valid enum values: none, winner_take_all, k_winners_allowed, mexican_hat, adaptive
        inhibition_strategy = stability_config.get("inhibition_strategy", "k_winners_allowed")
        target_activity = stability_config.get("target_activity", 0.1)
        
        # Initialize stability mechanisms
        self.stability = StabilityMechanisms(
            representation_size=self.multimodal_size,
            target_activity=target_activity,
            homeostatic_rate=stability_config.get("homeostatic_rate", 0.01),
            inhibition_strategy=inhibition_strategy,
            k_winners=stability_config.get("k_winners", int(0.1 * self.multimodal_size)),
            adaptive_threshold_tau=stability_config.get("adaptive_threshold_tau", 0.01)
        )
        
        logger.info(f"Initialized stability mechanisms: "
                  f"target_activity={target_activity}, "
                  f"inhibition_strategy={inhibition_strategy}")
    
    def _init_structural_plasticity(self):
        """Initialize structural plasticity component"""
        # Extract relevant config
        plasticity_config = self.config.get("structural_plasticity", {})
        
        # Use default values if not specified
        initial_size = self.multimodal_size
        max_size = plasticity_config.get("max_size", initial_size * 2)
        
        # Initialize structural plasticity
        self.structural_plasticity = StructuralPlasticity(
            initial_size=initial_size,
            max_size=max_size,
            growth_rate=plasticity_config.get("growth_rate", 0.05),
            prune_threshold=plasticity_config.get("prune_threshold", 0.01),
            novelty_threshold=plasticity_config.get("novelty_threshold", 0.3),
            enable_neuron_growth=plasticity_config.get("enable_neuron_growth", True),
            enable_connection_pruning=plasticity_config.get("enable_connection_pruning", True),
            enable_connection_sprouting=plasticity_config.get("enable_connection_sprouting", True)
        )
        
        # Override the original resize method to ensure it updates the system state
        self._original_resize = self.structural_plasticity.resize
        self.structural_plasticity.resize = self._synced_resize
        
        logger.info(f"Initialized structural plasticity: "
                  f"initial_size={initial_size}, "
                  f"max_size={max_size}")
    
    def _synced_resize(self, new_size: int) -> Dict[str, Any]:
        """
        Override for structural_plasticity.resize that ensures current_multimodal_state stays in sync.
        
        Args:
            new_size: New size for the representation
            
        Returns:
            Dictionary with resize information
        """
        # Call the original resize method
        result = self._original_resize(new_size)
        
        # Update current_multimodal_state to match the new size
        self._sync_multimodal_state_with_structure()
        
        # Log the synchronization
        logger.info(f"Synced multimodal state after resize: structural size={self.structural_plasticity.current_size}, "
                   f"multimodal state size={len(self.current_multimodal_state) if self.current_multimodal_state is not None else 0}")
        
        return result
    
    def _sync_multimodal_state_with_structure(self) -> None:
        """
        Ensure current_multimodal_state matches the current size of structural plasticity.
        This is called after any structural changes to maintain consistency.
        """
        current_size = self.structural_plasticity.current_size
        old_size = self.multimodal_size
        
        # Only proceed if we have a multimodal state
        if self.current_multimodal_state is not None:
            current_state_size = len(self.current_multimodal_state)
            
            # If sizes don't match, resize the multimodal state
            if current_state_size != current_size:
                logger.info(f"Syncing multimodal state: {current_state_size} -> {current_size}")
                
                if current_state_size < current_size:
                    # Pad with zeros
                    self.current_multimodal_state = np.pad(
                        self.current_multimodal_state,
                        (0, current_size - current_state_size)
                    )
                else:
                    # Truncate to new size
                    self.current_multimodal_state = self.current_multimodal_state[:current_size]
        
        # Update temporal prediction weights if they exist
        if hasattr(self, 'temporal_prediction') and self.temporal_prediction is not None:
            if hasattr(self.temporal_prediction, 'forward_weights'):
                logger.info(f"Updating temporal prediction weights for new size: {current_size}")
                
                # Update each timestep's weight matrix
                for t in range(len(self.temporal_prediction.forward_weights)):
                    weights = self.temporal_prediction.forward_weights[t]
                    
                    # Check if we need to resize the weights matrix
                    if weights.shape[0] != current_size or weights.shape[1] != current_size:
                        # Create new weights matrix with proper dimensions
                        new_weights = np.random.normal(0, 0.01, (current_size, current_size))
                        
                        # Copy existing weights where possible
                        min_rows = min(weights.shape[0], current_size)
                        min_cols = min(weights.shape[1], current_size)
                        new_weights[:min_rows, :min_cols] = weights[:min_rows, :min_cols]
                        
                        # Update the weights
                        self.temporal_prediction.forward_weights[t] = new_weights
                
                # Update recurrent weights if they exist
                if hasattr(self.temporal_prediction, 'recurrent_weights') and self.temporal_prediction.recurrent_weights is not None:
                    recurrent_weights = self.temporal_prediction.recurrent_weights
                    if recurrent_weights.shape[0] != current_size or recurrent_weights.shape[1] != current_size:
                        # Create new recurrent weights matrix with proper dimensions
                        new_recurrent_weights = np.random.normal(0, 0.01, (current_size, current_size))
                        
                        # Copy existing weights where possible
                        min_rows = min(recurrent_weights.shape[0], current_size)
                        min_cols = min(recurrent_weights.shape[1], current_size)
                        new_recurrent_weights[:min_rows, :min_cols] = recurrent_weights[:min_rows, :min_cols]
                        
                        # Zero out diagonal (no self-connections)
                        np.fill_diagonal(new_recurrent_weights, 0)
                        
                        # Update recurrent weights
                        self.temporal_prediction.recurrent_weights = new_recurrent_weights
                
                # Update confidence weights
                if hasattr(self.temporal_prediction, 'confidence_weights'):
                    for t in self.temporal_prediction.confidence_weights.keys():
                        confidence_weights = self.temporal_prediction.confidence_weights[t]
                        
                        if confidence_weights.shape[0] != current_size:
                            # Create new confidence weights with proper dimensions
                            new_confidence_weights = np.random.normal(0, 0.01, (current_size, 1))
                            
                            # Copy existing weights where possible
                            min_rows = min(confidence_weights.shape[0], current_size)
                            new_confidence_weights[:min_rows] = confidence_weights[:min_rows]
                            
                            # Update confidence weights
                            self.temporal_prediction.confidence_weights[t] = new_confidence_weights
                
                # Update representation size
                self.temporal_prediction.representation_size = current_size
                
                # Update eligibility traces if they exist
                if hasattr(self.temporal_prediction, 'eligibility_traces') and self.temporal_prediction.use_eligibility_trace:
                    logger.info(f"Updating temporal prediction eligibility traces for new size: {current_size}")
                    for t in range(len(self.temporal_prediction.eligibility_traces)):
                        if t in self.temporal_prediction.eligibility_traces:
                            traces = self.temporal_prediction.eligibility_traces[t]
                            
                            if traces.shape[0] != current_size or traces.shape[1] != current_size:
                                # Create new traces with proper dimensions
                                new_traces = np.zeros((current_size, current_size))
                                
                                # Copy existing traces where possible
                                min_rows = min(traces.shape[0], current_size)
                                min_cols = min(traces.shape[1], current_size)
                                new_traces[:min_rows, :min_cols] = traces[:min_rows, :min_cols]
                                
                                # Update traces
                                self.temporal_prediction.eligibility_traces[t] = new_traces
        
        # Update RGB control matrices if direct pixel control is enabled
        if hasattr(self, 'rgb_control_weights') and self.rgb_control_weights is not None:
            if 'multimodal' in self.rgb_control_weights:
                old_weights = self.rgb_control_weights['multimodal']
                
                if old_weights.shape[0] != current_size:
                    logger.info(f"Updating RGB control weights for new size: {current_size}")
                    
                    # Get output size
                    output_size = old_weights.shape[1]
                    
                    if old_weights.shape[0] < current_size:
                        # Add rows with small random values
                        new_rows = np.random.normal(0, 0.01, (current_size - old_weights.shape[0], output_size))
                        self.rgb_control_weights['multimodal'] = np.row_stack((old_weights, new_rows))
                    else:
                        # Truncate to the new size
                        self.rgb_control_weights['multimodal'] = old_weights[:current_size, :]
        
        # Update the stored multimodal size
        self.multimodal_size = current_size
        
        # Update stability mechanisms homeostatic factors if needed
        if hasattr(self, 'stability') and self.stability is not None:
            if hasattr(self.stability, 'homeostatic_factors') and len(self.stability.homeostatic_factors) != current_size:
                logger.info(f"Updating stability homeostatic factors for new size: {current_size}")
                
                # Create new homeostatic factors with proper dimensions
                new_factors = np.ones(current_size)
                
                # Copy existing factors where possible
                min_size = min(len(self.stability.homeostatic_factors), current_size)
                new_factors[:min_size] = self.stability.homeostatic_factors[:min_size]
                
                # Update homeostatic factors
                self.stability.homeostatic_factors = new_factors
                
                # Also update target activity count if needed
                if hasattr(self.stability, 'target_count'):
                    self.stability.target_count = int(current_size * self.stability.target_activity)
                    
                # Update thresholds if they exist
                if hasattr(self.stability, 'thresholds') and len(self.stability.thresholds) != current_size:
                    logger.info(f"Updating stability thresholds for new size: {current_size}")
                    
                    # Create new thresholds with proper dimensions
                    new_thresholds = np.zeros(current_size)
                    
                    # Copy existing thresholds where possible
                    min_size = min(len(self.stability.thresholds), current_size)
                    new_thresholds[:min_size] = self.stability.thresholds[:min_size]
                    
                    # Update thresholds
                    self.stability.thresholds = new_thresholds

    def _init_rgb_control(self):
        """Initialize the RGB control weights for direct pixel manipulation"""
        h, w = self.video_params['output_size'][::-1]  # Height and width
        output_size = h * w * 3  # RGB channels for each pixel
        
        # Initialize weights from neural patterns to RGB values with higher variance
        # for more visual interest and exploration
        self.rgb_control_weights = {
            # From multimodal to RGB - use higher variance for more visual variation
            'multimodal': np.random.normal(0, 0.05, (self.multimodal_size, output_size)),
            
            # From visual to RGB - ensure some randomness by default
            'visual': np.random.normal(0, 0.03, (self.visual_output_size, output_size)),
            
            # Optional audio to RGB influence - lower influence by default
            'audio': np.random.normal(0, 0.01, (self.audio_output_size, output_size))
        }
        
        # Add more initial structure to generate distinct patterns
        # Create some gradients and patterns in the initial weights
        for i in range(self.multimodal_size):
            # Create structured patterns in weight space
            if i % 3 == 0:  # Every third neuron gets horizontal stripes
                pattern = np.zeros((h, w, 3))
                for row in range(h):
                    if row % 8 < 4:  # Create stripe pattern
                        pattern[row, :, 0] = 0.1  # Red channel
                        
                self.rgb_control_weights['multimodal'][i] = pattern.reshape(-1) + self.rgb_control_weights['multimodal'][i]
            
            elif i % 3 == 1:  # Every third + 1 neuron gets vertical stripes
                pattern = np.zeros((h, w, 3))
                for col in range(w):
                    if col % 8 < 4:  # Create stripe pattern
                        pattern[:, col, 1] = 0.1  # Green channel
                        
                self.rgb_control_weights['multimodal'][i] = pattern.reshape(-1) + self.rgb_control_weights['multimodal'][i]
            
            else:  # Remaining neurons get diagonal patterns
                pattern = np.zeros((h, w, 3))
                for row in range(h):
                    for col in range(w):
                        if (row + col) % 8 < 4:  # Create diagonal pattern
                            pattern[row, col, 2] = 0.1  # Blue channel
                            
                self.rgb_control_weights['multimodal'][i] = pattern.reshape(-1) + self.rgb_control_weights['multimodal'][i]
        
        # Create a baseline RGB representation
        self.direct_pixel_output = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create separate channel control factors to allow independent channel control
        self.rgb_channel_bias = np.zeros((3, output_size // 3))  # One bias per RGB channel per pixel
        self.rgb_channel_gain = np.ones((3, output_size // 3))   # One gain factor per RGB channel per pixel
        
        # Initialize with some variations to create more independent RGB channels
        for c in range(3):  # For each RGB channel
            # Add some random bias to each channel (centered around different values)
            self.rgb_channel_bias[c] = np.random.normal(0.05 * (c + 1), 0.02, output_size // 3)
            # Add some random gain to each channel (centered around different values)
            self.rgb_channel_gain[c] = np.random.normal(1.0 + 0.1 * c, 0.05, output_size // 3)
        
        logger.info(f"Initialized RGB control for direct pixel manipulation: output shape {self.direct_pixel_output.shape}")

    def set_direct_pixel_control(self, enabled: bool):
        """Enable or disable direct pixel control by the model
        
        Args:
            enabled: Whether to enable direct RGB pixel control
        """
        self.direct_pixel_control = enabled
        logger.info(f"Direct pixel control {'enabled' if enabled else 'disabled'}")
        
    def get_direct_pixel_output(self, output_size: Tuple[int, int]):
        """Generate direct RGB pixel output from the model's internal state
        
        Args:
            output_size: Desired (width, height) of the output image
        
        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        # Update output size if needed
        if output_size != self.video_params['output_size']:
            self.video_params['output_size'] = output_size
            h, w = output_size[::-1]  # Height, width
            self._resize_rgb_control(h, w)
        
        # If we have valid state, generate RGB output
        if self.current_multimodal_state is not None:
            # Generate RGB values from the multimodal representation
            flattened_rgb = self.current_multimodal_state @ self.rgb_control_weights['multimodal']
            
            # Add influence from visual representation if available
            if self.current_visual_encoding is not None:
                flattened_rgb += self.current_visual_encoding @ self.rgb_control_weights['visual']
            
            # Add influence from audio representation if available
            if self.current_audio_encoding is not None:
                flattened_rgb += self.current_audio_encoding @ self.rgb_control_weights['audio']
            
            # Shape into RGB image: (height, width, 3)
            h, w = self.video_params['output_size'][::-1]
            total_pixels = h * w
            
            if self.enhanced_rgb_control:
                # Process each RGB channel separately for more independent control
                rgb_shaped = np.zeros((h, w, 3))
                for c in range(3):  # For each RGB channel
                    # Extract this channel's values, apply channel-specific gain and bias
                    channel_values = flattened_rgb[c::3]  # Every 3rd value starting from channel index
                    channel_values = channel_values * self.rgb_channel_gain[c] + self.rgb_channel_bias[c]
                    
                    # Apply separate sigmoid to each channel for more independent control
                    channel_normalized = 1.0 / (1.0 + np.exp(-channel_values))
                    
                    # Reshape to 2D spatial array and assign to the appropriate color channel
                    rgb_shaped[:, :, c] = channel_normalized.reshape(h, w)
                
                # Scale to 0-255 for RGB output
                self.direct_pixel_output = (rgb_shaped * 255).astype(np.uint8)
            else:
                # Traditional approach: reshape and apply sigmoid to all channels together
                rgb_shaped = flattened_rgb.reshape((h, w, 3))
                rgb_normalized = 1.0 / (1.0 + np.exp(-rgb_shaped))  # Sigmoid
                self.direct_pixel_output = (rgb_normalized * 255).astype(np.uint8)
            
            # Apply any additional transformations from video_params
            if self.video_params.get('grayscale', False):
                # Convert to grayscale but keep 3 channels
                gray = cv2.cvtColor(self.direct_pixel_output, cv2.COLOR_RGB2GRAY)
                self.direct_pixel_output = np.stack([gray, gray, gray], axis=2)
            
            # Apply contrast adjustment
            contrast = self.video_params.get('contrast', 1.0)
            if contrast != 1.0:
                self.direct_pixel_output = np.clip(
                    self.direct_pixel_output * contrast, 0, 255).astype(np.uint8)
        
        return self.direct_pixel_output
    
    def _resize_rgb_control(self, height: int, width: int):
        """Resize the RGB control matrices when the output size changes
        
        Args:
            height: New height
            width: New width
        """
        new_output_size = height * width * 3
        
        # Create new weights with appropriate size
        for modality in self.rgb_control_weights:
            old_weights = self.rgb_control_weights[modality]
            input_size = old_weights.shape[0]
            
            # Initialize new weights
            new_weights = np.random.normal(0, 0.01, (input_size, new_output_size))
            
            # Try to preserve existing patterns by sampling/scaling
            if self.rgb_control_weights[modality].size > 0:
                # Simple approach: initialize with small random values
                self.rgb_control_weights[modality] = new_weights
        
        # Initialize direct pixel output with new size
        self.direct_pixel_output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update channel bias and gain for new size
        total_pixels = height * width
        self.rgb_channel_bias = np.zeros((3, total_pixels))
        self.rgb_channel_gain = np.ones((3, total_pixels))
        
        # Initialize with some variations
        for c in range(3):  # For each RGB channel
            self.rgb_channel_bias[c] = np.random.normal(0.05 * (c + 1), 0.02, total_pixels)
            self.rgb_channel_gain[c] = np.random.normal(1.0 + 0.1 * c, 0.05, total_pixels)
        
        logger.info(f"Resized RGB control matrices to {(height, width)}")
        
    def update_rgb_weights(self, target_image: np.ndarray, learning_rate: Optional[float] = None):
        """Learn to associate the current neural state with a target RGB image
        
        This allows the model to learn to generate specific RGB patterns.
        
        Args:
            target_image: Target RGB image of shape (height, width, 3)
            learning_rate: Learning rate for this update (or use default)
        """
        if not self.rgb_learning_enabled:
            return
            
        if learning_rate is None:
            learning_rate = self.rgb_learning_rate
            
        if self.current_multimodal_state is None:
            logger.warning("Cannot update RGB weights: no multimodal state available")
            return
            
        # Ensure target image has the right shape
        h, w = self.video_params['output_size'][::-1]
        if target_image.shape != (h, w, 3):
            logger.warning(f"Target image shape {target_image.shape} doesn't match expected {(h, w, 3)}")
            target_image = cv2.resize(target_image, (w, h))
            
        # Flatten the target image
        target_flat = target_image.reshape(-1) / 255.0  # Normalize to 0-1
        
        # Get the current output
        current_output = self.get_direct_pixel_output((w, h))
        current_flat = current_output.reshape(-1) / 255.0
        
        # Compute error
        error = target_flat - current_flat
        
        # Update weights using a simple delta rule
        if self.current_multimodal_state is not None:
            delta = learning_rate * np.outer(self.current_multimodal_state, error)
            self.rgb_control_weights['multimodal'] += delta
            
        if self.current_visual_encoding is not None:
            delta = learning_rate * np.outer(self.current_visual_encoding, error)
            self.rgb_control_weights['visual'] += delta
            
        if self.current_audio_encoding is not None:
            delta = learning_rate * 0.5 * np.outer(self.current_audio_encoding, error)  # Less influence from audio
            self.rgb_control_weights['audio'] += delta
            
        logger.debug(f"Updated RGB control weights, error: {np.mean(np.abs(error)):.4f}")

    def process(
        self,
        visual_input: np.ndarray,
        audio_input: np.ndarray,
        learning_enabled: bool = None,
        target_rgb_output: Optional[np.ndarray] = None,  # Optional target RGB output for learning
    ) -> Dict[str, Any]:
        """
        Process synchronized audio-visual input.
        
        Args:
            visual_input: Visual input frame (raw image)
            audio_input: Audio input snippet (raw waveform)
            learning_enabled: Whether to enable learning for this input (overrides global setting)
            target_rgb_output: Optional target RGB output for pixel control learning
            
        Returns:
            Dictionary with processing results
        """
        if not self.system_ready:
            logger.warning("System not fully initialized yet")
            return {"error": "System not fully initialized"}
        
        # Track performance
        start_time = time.time()
        
        # Use global learning setting if not specified
        if learning_enabled is None:
            learning_enabled = self.enable_learning
        
        # Process visual input
        visual_encoding = self.visual_processor.process_frame(visual_input)
        
        # Process audio input
        audio_encoding = self.audio_processor.process_waveform(audio_input)[0]  # Take first frame
        
        # Store current encodings
        self.current_visual_encoding = visual_encoding
        self.current_audio_encoding = audio_encoding
        
        # Create dictionary of modality activities
        modality_activities = {
            "visual": visual_encoding,
            "audio": audio_encoding
        }
        
        # Update multimodal associations
        association_result = self.multimodal_association.update(
            modality_activities=modality_activities,
            learning_enabled=learning_enabled
        )
        
        # Get multimodal state
        multimodal_state = association_result["association_activity"]
        
        # Ensure the multimodal state size matches structural plasticity size
        current_size = self.structural_plasticity.current_size
        if len(multimodal_state) != current_size:
            logger.warning(f"Multimodal state size mismatch during processing: {len(multimodal_state)} != {current_size}")
            
            # Resize to match structural plasticity
            if len(multimodal_state) < current_size:
                # Pad with zeros
                multimodal_state = np.pad(multimodal_state, (0, current_size - len(multimodal_state)))
            else:
                # Truncate to match
                multimodal_state = multimodal_state[:current_size]
            
            logger.info(f"Resized multimodal state to {len(multimodal_state)} during processing")
        
        self.current_multimodal_state = multimodal_state
        
        # Process temporal sequence if needed
        if self.temporal_prediction is not None:
            temporal_result = self.temporal_prediction.update(
                current_state=multimodal_state
            )
        else:
            temporal_result = None
            
        # Apply stability mechanisms if needed
        if self.stability is not None:
            pre_stability = multimodal_state.copy()
            stable_state = self.stability.apply_inhibition(multimodal_state)
            stable_state = self.stability.apply_homeostasis(stable_state)
            stable_state = self.stability.apply_thresholds(stable_state)
            
            # Update current state with stabilized version
            self.current_multimodal_state = stable_state
        
        # Check if it's time to update structural plasticity
        update_structural = (
            self.structural_plasticity is not None and
            self.frames_processed % self.structural_plasticity_interval == 0
        )
        
        if update_structural:
            weights = None
            if self.multimodal_association is not None:
                weights = self.multimodal_association.weights.get("visual", None)
                
            structural_result = self.structural_plasticity.update(
                activity=self.current_multimodal_state,
                weights=weights,
                reconstruction_error=association_result.get("reconstruction_error", None)
            )
            
            # If structure changed, make sure multimodal state is synced
            if structural_result.get("size_changed", False):
                logger.info(f"Structural change detected during update: {structural_result}")
                self._sync_multimodal_state_with_structure()
        else:
            structural_result = None
        
        # Update direct pixel output if enabled
        if self.direct_pixel_control and target_rgb_output is not None:
            self.update_rgb_weights(target_rgb_output)
        
        # Check if it's time for pruning
        if (
            self.frames_processed % self.prune_interval == 0 and
            self.structural_plasticity is not None and
            self.structural_plasticity.enable_connection_pruning
        ):
            # Pruning handled within structural plasticity update
            pass
            
        # Update frame counter
        self.frames_processed += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            "processing_time": processing_time,
            "visual_encoding": visual_encoding,
            "audio_encoding": audio_encoding,
            "multimodal_state": self.current_multimodal_state
        }
        
        # Return the result
        return result
    
    def get_cross_modal_prediction(
        self,
        source_modality: str,
        source_data: np.ndarray,
        target_modality: str
    ) -> Dict[str, Any]:
        """
        Generate cross-modal prediction (e.g., audio->visual or visual->audio).
        
        Args:
            source_modality: Source modality ("visual" or "audio")
            source_data: Input data for source modality
            target_modality: Target modality to predict
            
        Returns:
            Dictionary with prediction results
        """
        if not self.system_ready:
            return {"error": "System not fully initialized"}
        
        # Process source data
        source_encoding = None
        if source_modality == "visual":
            source_encoding = self.visual_processor.process_frame(source_data)
        elif source_modality == "audio":
            source_encoding = self.audio_processor.process_waveform(source_data)[0]  # Take first frame
        else:
            return {"error": f"Unknown source modality: {source_modality}"}
        
        # Generate prediction
        prediction = self.multimodal_association.get_cross_modal_prediction(
            source_modality=source_modality,
            source_activity=source_encoding,
            target_modality=target_modality
        )
        
        # For visualization purposes, decode the prediction if needed
        decoded_prediction = None
        # We would need proper decoders in the respective processors
        
        result = {
            "source_modality": source_modality,
            "target_modality": target_modality,
            "source_encoding": source_encoding,
            "prediction": prediction,
            "decoded_prediction": decoded_prediction
        }
        
        return result
    
    def get_temporal_predictions(
        self,
        steps: int = 3,
        include_decoded: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get temporal predictions for future states.
        
        Args:
            steps: Number of future steps to predict
            include_decoded: Whether to include decoded predictions
            
        Returns:
            Dictionary mapping steps to prediction results
        """
        if not self.system_ready or self.current_multimodal_state is None:
            return {0: {"error": "No current state available"}}
        
        # Get predictions from temporal prediction component
        raw_predictions = self.temporal_prediction.predict_future(
            current_state=self.current_multimodal_state,
            steps=steps,
            include_confidence=True
        )
        
        # Create result dictionary
        results = {}
        
        # Process each prediction
        for step, (prediction, confidence) in raw_predictions.items():
            # Reconstruct modalities from prediction
            visual_reconstruction = self.multimodal_association.get_cross_modal_prediction(
                source_modality="multimodal",
                source_activity=prediction,
                target_modality="visual"
            )
            
            audio_reconstruction = self.multimodal_association.get_cross_modal_prediction(
                source_modality="multimodal",
                source_activity=prediction,
                target_modality="audio"
            )
            
            # Store results (without decoded outputs as they're not implemented)
            results[step] = {
                "multimodal_prediction": prediction,
                "confidence": confidence,
                "visual_reconstruction": visual_reconstruction,
                "audio_reconstruction": audio_reconstruction
            }
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics including pathway states and performance metrics for monitoring.
        
        Returns:
            Dictionary with system statistics
        """
        # Get pathway states required by the monitor
        visual_state = self.visual_processor.get_pathway_state()
        audio_state = self.audio_processor.get_pathway_state()
        multimodal_activity = self.multimodal_association.get_multimodal_activity() if hasattr(self.multimodal_association, 'get_multimodal_activity') else None

        stats = {
            "frames_processed": self.frames_processed,
            "multimodal_size": self.multimodal_size,
            "visual_output_size": self.visual_output_size,
            "audio_output_size": self.audio_output_size,
            "learning_enabled": self.enable_learning,
            # Add pathway states directly for monitor compatibility
            "visual": visual_state,
            "audio": audio_state,
            "multimodal_activity": multimodal_activity,
            # Add system performance metrics
            "metrics": {
                "prediction_error": self.prediction_errors,
                "reconstruction_error": self.reconstruction_errors # Include reconstruction errors too
            },
            "components": {
                "visual_processor": self.visual_processor.get_stats(),
                "audio_processor": self.audio_processor.get_stats(),
                "multimodal_association": self.multimodal_association.get_stats(),
                "temporal_prediction": self.temporal_prediction.get_stats(),
                "stability": {
                    "target_activity": self.stability.target_activity,
                    "current_activity": float(np.mean(self.current_multimodal_state > 0)) 
                    if self.current_multimodal_state is not None else 0.0
                },
                "structural_plasticity": self.structural_plasticity.get_stats()
            }
        }
        
        return stats
    
    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """
        Save system checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save component states
        visual_state = self.visual_processor.get_pathway_state()
        np.savez(os.path.join(checkpoint_path, "visual_processor.npz"), **visual_state)
        
        audio_state = self.audio_processor.get_pathway_state()
        np.savez(os.path.join(checkpoint_path, "audio_processor.npz"), **audio_state)
        
        # Save multimodal association
        association_state = self.multimodal_association.serialize()
        np.savez(
            os.path.join(checkpoint_path, "multimodal_association.npz"),
            **association_state
        )
        
        # Save temporal prediction
        prediction_state = self.temporal_prediction.serialize()
        np.savez(
            os.path.join(checkpoint_path, "temporal_prediction.npz"),
            **prediction_state
        )
        
        # Save stability state (if available)
        stability_state = {
            "target_activity": self.stability.target_activity,
            "homeostatic_rate": self.stability.homeostatic_rate
        }
        
        # Include thresholds if they exist
        if hasattr(self.stability, "thresholds"):
            if isinstance(self.stability.thresholds, np.ndarray):
                stability_state["thresholds"] = self.stability.thresholds.tolist()
            else:
                stability_state["thresholds"] = self.stability.thresholds
        
        np.savez(
            os.path.join(checkpoint_path, "stability.npz"),
            **stability_state
        )
        
        # Save structural plasticity state
        plasticity_state = self.structural_plasticity.serialize()
        np.savez(
            os.path.join(checkpoint_path, "structural_plasticity.npz"),
            **plasticity_state
        )
        
        # Save config
        with open(os.path.join(checkpoint_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        visual_processor_class=None,
        audio_processor_class=None
    ) -> 'SelfOrganizingAVSystem':
        """
        Load system from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            visual_processor_class: Class to use for visual processor
            audio_processor_class: Class to use for audio processor
            
        Returns:
            Loaded system instance
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
        
        # Load config
        config_path = os.path.join(checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found in checkpoint: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Determine processor classes if not provided
        if visual_processor_class is None:
            # Import inside the method to avoid circular imports
            import importlib
            try:
                visual_module = importlib.import_module("self_organizing_av_system.models.visual.processor")
                visual_processor_class = visual_module.VisualProcessor
            except ImportError:
                logger.error("Could not import VisualProcessor, make sure it's accessible in the PYTHONPATH")
                raise
        
        if audio_processor_class is None:
            # Import inside the method to avoid circular imports
            import importlib
            try:
                audio_module = importlib.import_module("self_organizing_av_system.models.audio.processor")
                audio_processor_class = audio_module.AudioProcessor
            except ImportError:
                logger.error("Could not import AudioProcessor, make sure it's accessible in the PYTHONPATH")
                raise
        
        # Load processor configurations
        visual_config = config.get("visual_processor", {})
        audio_config = config.get("audio_processor", {})
        
        # Create processor instances
        visual_processor = visual_processor_class(**visual_config)
        audio_processor = audio_processor_class(**audio_config)
        
        # Create system instance
        system = cls(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config=config
        )
        
        try:
            # Load visual processor state
            visual_processor_path = os.path.join(checkpoint_path, "visual_processor.npz")
            if os.path.exists(visual_processor_path):
                visual_state = dict(np.load(visual_processor_path))
                # Set pathway state directly rather than using load_state
                visual_processor.visual_pathway = visual_state.get("pathway", visual_processor.visual_pathway)
            
            # Load audio processor state
            audio_processor_path = os.path.join(checkpoint_path, "audio_processor.npz")
            if os.path.exists(audio_processor_path):
                audio_state = dict(np.load(audio_processor_path))
                # Set pathway state directly rather than using load_state
                audio_processor.audio_pathway = audio_state.get("pathway", audio_processor.audio_pathway)
            
            # Load component states
            # Multimodal association
            association_path = os.path.join(checkpoint_path, "multimodal_association.npz")
            if os.path.exists(association_path):
                association_data = dict(np.load(association_path))
                system.multimodal_association = MultimodalAssociation.deserialize(association_data)
            
            # Temporal prediction
            prediction_path = os.path.join(checkpoint_path, "temporal_prediction.npz")
            if os.path.exists(prediction_path):
                prediction_data = dict(np.load(prediction_path))
                system.temporal_prediction = TemporalPrediction.deserialize(prediction_data)
            
            # Structural plasticity
            plasticity_path = os.path.join(checkpoint_path, "structural_plasticity.npz")
            if os.path.exists(plasticity_path):
                plasticity_data = dict(np.load(plasticity_path))
                system.structural_plasticity = StructuralPlasticity.deserialize(plasticity_data)
        except Exception as e:
            logger.error(f"Error loading checkpoint components: {e}")
            logger.info("Continuing with partially loaded system")
        
        logger.info(f"Loaded system from checkpoint: {checkpoint_path}")
        return system
    
    def process_av_pair(self, frame: np.ndarray, audio_chunk: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Process a synchronized audio-visual pair.
        
        Args:
            frame: Video frame
            audio_chunk: Audio chunk
            learn: Whether to enable learning
            
        Returns:
            Processing results
        """
        return self.process(frame, audio_chunk, learning_enabled=learn)
    
    def process_video_sequence(
        self, 
        frames: List[np.ndarray], 
        audio_waveform: np.ndarray, 
        sample_rate: int,
        learn: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a sequence of video frames with corresponding audio.
        
        Args:
            frames: List of video frames
            audio_waveform: Audio waveform for the entire sequence
            sample_rate: Audio sample rate
            learn: Whether to enable learning
            
        Returns:
            List of processing results
        """
        results = []
        
        # Process each frame with corresponding audio
        total_frames = len(frames)
        for i, frame in enumerate(frames):
            # Calculate audio segment for this frame
            # This is a simple mapping - in reality we might need more sophisticated syncing
            audio_samples_per_frame = len(audio_waveform) / total_frames
            start_sample = int(i * audio_samples_per_frame)
            end_sample = int((i + 1) * audio_samples_per_frame)
            
            # Get audio chunk for this frame
            audio_chunk = audio_waveform[start_sample:end_sample]
            
            # Process AV pair
            result = self.process_av_pair(frame, audio_chunk, learn=learn)
            results.append(result)
        
        return results
    
    def save_state(self, filepath: str) -> bool:
        """
        Save system state to a file.
        
        Args:
            filepath: Path to save the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory:  # Only create directory if there's a directory part
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Ensured directory exists: {directory}")
                except Exception as dir_error:
                    logger.error(f"Failed to create directory {directory}: {dir_error}")
                    # Try to continue anyway in case directory already exists
            
            # Prepare state dictionary
            state = {
                'frame_count': self.frames_processed,
                'config': self.config,
                'visual_state': self.visual_processor.get_pathway_state(),
                'audio_state': self.audio_processor.get_pathway_state(),
                'multimodal_state': {
                    'weights': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in self.multimodal_association.weights.items()},
                    'biases': self.multimodal_association.biases.tolist() if hasattr(self.multimodal_association, 'biases') else None,
                    'association_size': self.multimodal_size
                },
                'timestamp': time.time()
            }
            
            # Save to file
            import pickle
            logger.info(f"Preparing to save state to {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved system state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save system state to {filepath}: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
            
    def load_state(self, filepath: str) -> bool:
        """
        Load system state from a file.
        
        Args:
            filepath: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"State file not found: {filepath}")
                return False
                
            # Load state
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Update frame count
            self.frames_processed = state.get('frame_count', 0)
            
            # Update config if present
            if 'config' in state:
                self.config.update(state['config'])
            
            # Check if model dimensions match the saved state
            model_compatible = True
            
            # Check visual pathway compatibility
            if 'visual_state' in state:
                try:
                    visual_state = state['visual_state']
                    for i, layer_state in enumerate(visual_state.get('layers', [])):
                        if i < len(self.visual_processor.visual_pathway.layers):
                            current_size = self.visual_processor.visual_pathway.layers[i].layer_size
                            saved_size = len(layer_state.get('activations', []))
                            if current_size != saved_size:
                                logger.warning(f"Visual layer {i} size mismatch: saved={saved_size}, current={current_size}")
                                model_compatible = False
                except Exception as e:
                    logger.error(f"Error checking visual pathway compatibility: {e}")
                    model_compatible = False
            
            # Check audio pathway compatibility
            if 'audio_state' in state:
                try:
                    audio_state = state['audio_state']
                    for i, layer_state in enumerate(audio_state.get('layers', [])):
                        if i < len(self.audio_processor.audio_pathway.layers):
                            current_size = self.audio_processor.audio_pathway.layers[i].layer_size
                            saved_size = len(layer_state.get('activations', []))
                            if current_size != saved_size:
                                logger.warning(f"Audio layer {i} size mismatch: saved={saved_size}, current={current_size}")
                                model_compatible = False
                except Exception as e:
                    logger.error(f"Error checking audio pathway compatibility: {e}")
                    model_compatible = False
            
            # Check multimodal compatibility
            if 'multimodal_state' in state:
                try:
                    mm_state = state['multimodal_state']
                    saved_size = mm_state.get('association_size', 0)
                    if saved_size != self.multimodal_size:
                        logger.warning(f"Multimodal size mismatch: saved={saved_size}, current={self.multimodal_size}")
                        model_compatible = False
                except Exception as e:
                    logger.error(f"Error checking multimodal compatibility: {e}")
                    model_compatible = False
            
            # If model is incompatible, don't load weights and return false
            if not model_compatible:
                logger.warning("Model dimensions don't match checkpoint. Starting with fresh weights.")
                logger.info(f"Note: You can still use the frame count from the checkpoint: {self.frames_processed}")
                return True  # Return true so the system still runs with fresh weights
            
            # Load visual processor state if present
            if 'visual_state' in state:
                try:
                    visual_state = state['visual_state']
                    # Set activations for each layer
                    for i, layer_state in enumerate(visual_state.get('layers', [])):
                        if i < len(self.visual_processor.visual_pathway.layers):
                            layer = self.visual_processor.visual_pathway.layers[i]
                            if 'activations' in layer_state:
                                layer.activations = np.array(layer_state['activations'])
                            if 'weights' in layer_state:
                                for j, neuron in enumerate(layer.neurons):
                                    if j < len(layer_state['weights']):
                                        neuron.weights = np.array(layer_state['weights'][j])
                    logger.info("Restored visual pathway state")
                except Exception as e:
                    logger.error(f"Error restoring visual state: {e}")
            
            # Load audio processor state if present
            if 'audio_state' in state:
                try:
                    audio_state = state['audio_state']
                    # Set activations for each layer
                    for i, layer_state in enumerate(audio_state.get('layers', [])):
                        if i < len(self.audio_processor.audio_pathway.layers):
                            layer = self.audio_processor.audio_pathway.layers[i]
                            if 'activations' in layer_state:
                                layer.activations = np.array(layer_state['activations'])
                            if 'weights' in layer_state:
                                for j, neuron in enumerate(layer.neurons):
                                    if j < len(layer_state['weights']):
                                        neuron.weights = np.array(layer_state['weights'][j])
                    logger.info("Restored audio pathway state")
                except Exception as e:
                    logger.error(f"Error restoring audio state: {e}")
            
            # Load multimodal state if present
            if 'multimodal_state' in state:
                try:
                    mm_state = state['multimodal_state']
                    if 'weights' in mm_state:
                        weights_dict = mm_state['weights']
                        # Convert lists back to numpy arrays
                        for modality, weights in weights_dict.items():
                            if isinstance(weights, list):
                                weights_dict[modality] = np.array(weights)
                        # Update weights in multimodal association
                        self.multimodal_association.weights = weights_dict
                    
                    # Restore biases if present
                    if 'biases' in mm_state and mm_state['biases'] is not None:
                        self.multimodal_association.biases = np.array(mm_state['biases'])
                        
                    logger.info("Restored multimodal association state")
                except Exception as e:
                    logger.error(f"Error restoring multimodal state: {e}")
            
            logger.info(f"Loaded system state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    @property
    def multimodal(self):
        """Provide access to multimodal functions for backward compatibility"""
        return self
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current state of the system for monitoring.
        
        Returns:
            Dictionary with system state
        """
        # Use the get_stats method to gather all necessary information
        stats = self.get_stats()
        
        # Add any additional information needed specifically for visualization
        try:
            # Ensure multimodal_association has forward_weights for visualization
            if hasattr(self.multimodal_association, 'weights'):
                stats['multimodal_weights'] = self.multimodal_association.weights
            elif hasattr(self.multimodal_association, 'forward_weights'):
                stats['multimodal_weights'] = self.multimodal_association.forward_weights
            elif hasattr(self.multimodal_association, 'association_weights'):
                stats['multimodal_weights'] = self.multimodal_association.association_weights
            
            # Add placeholder if no weights are available
            if 'multimodal_weights' not in stats:
                # Create placeholder weights for visualization
                modality_sizes = {
                    "visual": self.visual_output_size,
                    "audio": self.audio_output_size
                }
                
                placeholder_weights = {}
                for modality, size in modality_sizes.items():
                    placeholder_weights[modality] = np.random.rand(size, self.multimodal_size) * 0.1
                
                stats['multimodal_weights'] = placeholder_weights
                
        except Exception as e:
            logger.warning(f"Error adding extra visualization data: {e}")
            
        return stats
    
    def analyze_associations(self) -> Dict[str, List]:
        """
        Analyze cross-modal associations.
        
        Returns:
            Dictionary with association analysis results
        """
        associations = {
            "visual_to_audio": [],
            "audio_to_visual": [],
            "stable_associations": [],
            "statistics": {}
        }
        
        # Get association weights from multimodal layer
        if hasattr(self.multimodal_association, 'weights'):
            weights = self.multimodal_association.weights
            
            # Analyze visual-to-audio associations
            visual_weights = weights.get('visual', None)
            if visual_weights is not None:
                # Find strongest visual-to-multimodal connections
                strong_visual = np.where(np.abs(visual_weights) > 0.5)
                for i, j in zip(strong_visual[0], strong_visual[1]):
                    associations["visual_to_audio"].append({
                        "visual_neuron": int(j),
                        "multimodal_neuron": int(i),
                        "strength": float(visual_weights[i, j])
                    })
            
            # Analyze audio-to-visual associations
            audio_weights = weights.get('audio', None)
            if audio_weights is not None:
                # Find strongest audio-to-multimodal connections
                strong_audio = np.where(np.abs(audio_weights) > 0.5)
                for i, j in zip(strong_audio[0], strong_audio[1]):
                    associations["audio_to_visual"].append({
                        "audio_neuron": int(j),
                        "multimodal_neuron": int(i),
                        "strength": float(audio_weights[i, j])
                    })
            
            # Find stable cross-modal associations
            if visual_weights is not None and audio_weights is not None:
                # Neurons that are strongly connected to both modalities
                visual_strength = np.abs(visual_weights).mean(axis=1)
                audio_strength = np.abs(audio_weights).mean(axis=1)
                
                # Find neurons with strong connections to both
                strong_both = np.where((visual_strength > 0.3) & (audio_strength > 0.3))[0]
                for neuron in strong_both:
                    associations["stable_associations"].append({
                        "multimodal_neuron": int(neuron),
                        "visual_strength": float(visual_strength[neuron]),
                        "audio_strength": float(audio_strength[neuron]),
                        "combined_strength": float(visual_strength[neuron] * audio_strength[neuron])
                    })
                
                # Add statistics
                associations["statistics"] = {
                    "total_visual_connections": int(np.sum(np.abs(visual_weights) > 0.1)),
                    "total_audio_connections": int(np.sum(np.abs(audio_weights) > 0.1)),
                    "strong_associations": len(associations["stable_associations"]),
                    "avg_visual_weight": float(np.mean(np.abs(visual_weights))),
                    "avg_audio_weight": float(np.mean(np.abs(audio_weights)))
                }
        
        return associations

    # --- Methods for GUI Interaction ---
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Provide a summary of the current model architecture for the GUI.
        Returns:
            A dictionary containing key architectural parameters.
        """
        # Get layer sizes from processors
        # The visual_processor and audio_processor may have a layer_sizes attribute
        # or we need to infer them from the pathway
        visual_layers = getattr(self.visual_processor, 'layer_sizes', None)
        if not visual_layers and hasattr(self.visual_processor, 'visual_pathway') and hasattr(self.visual_processor.visual_pathway, 'layers'):
            visual_layers = [layer.layer_size for layer in self.visual_processor.visual_pathway.layers]
            
        audio_layers = getattr(self.audio_processor, 'layer_sizes', None)
        if not audio_layers and hasattr(self.audio_processor, 'audio_pathway') and hasattr(self.audio_processor.audio_pathway, 'layers'):
            audio_layers = [layer.layer_size for layer in self.audio_processor.audio_pathway.layers]
        
        info = {
            "Visual Layers": visual_layers,
            "Audio Layers": audio_layers,
            "Multimodal Size": self.multimodal_size,
            "Learning Rate": self.learning_rate,
            "Learning Rule": self.learning_rule,
            "Pruning Enabled": self.config.get("structural_plasticity", {}).get("enable_connection_pruning", True),
            "Growth Enabled": self.config.get("structural_plasticity", {}).get("enable_neuron_growth", True),
            "Frames Processed": self.frames_processed
        }
        # Add current sizes if structural plasticity is enabled
        if hasattr(self, 'structural_plasticity') and hasattr(self.structural_plasticity, 'current_size'):
             info["Current Multimodal Size"] = self.structural_plasticity.current_size

        # Add connection info if available
        if hasattr(self.multimodal_association, 'get_connection_stats'):
            info["Connections"] = self.multimodal_association.get_connection_stats()
        elif hasattr(self.multimodal_association, 'weights') and isinstance(self.multimodal_association.weights, dict):
            total_connections = 0
            for modality, matrix in self.multimodal_association.weights.items():
                if matrix is not None:
                    total_connections += np.sum(matrix != 0) # Count non-zero weights
            info["Approx. Connections"] = total_connections

        # Add dynamic structural information
        info["Structural Changes"] = {
            "Neurons Added": len(self.structural_plasticity.growth_events) if hasattr(self.structural_plasticity, 'growth_events') else 0,
            "Neurons Pruned": len(self.structural_plasticity.prune_events) if hasattr(self.structural_plasticity, 'prune_events') else 0,
            "Current Connections": self._get_connection_count(),
            "Recent Plasticity Events": self.structural_plasticity.recent_events if hasattr(self.structural_plasticity, 'recent_events') else []
        }
        
        # Add layer-wise plasticity metrics
        try:
            info["Visual Plasticity"] = self.visual_processor.get_plasticity_stats() if hasattr(self.visual_processor, 'get_plasticity_stats') else {}
        except AttributeError:
            info["Visual Plasticity"] = {}
            
        try:
            info["Audio Plasticity"] = self.audio_processor.get_plasticity_stats() if hasattr(self.audio_processor, 'get_plasticity_stats') else {}
        except AttributeError:
            info["Audio Plasticity"] = {}
        
        # Add RGB control information
        info["RGB Control"] = {
            "Enabled": self.direct_pixel_control,
            "Learning Enabled": self.rgb_learning_enabled,
            "Learning Rate": self.rgb_learning_rate,
            "Output Size": self.video_params['output_size']
        }
        
        # Add stats about the RGB weights if enabled
        if self.direct_pixel_control and hasattr(self, 'rgb_control_weights'):
            rgb_weight_stats = {}
            for modality, weights in self.rgb_control_weights.items():
                if weights is not None:
                    rgb_weight_stats[f"{modality}_mean"] = float(np.mean(weights))
                    rgb_weight_stats[f"{modality}_std"] = float(np.std(weights))
                    rgb_weight_stats[f"{modality}_nonzero"] = float(np.count_nonzero(weights)) / weights.size
            info["RGB Control"]["Weights"] = rgb_weight_stats
        
        return info

    def update_video_params(self, params):
        """Update video processing parameters from GUI"""
        self.video_params.update(params)
        
    def get_video_panel_output(self, params=None):
        """Get video output based on current mode (direct control or neural visualization)"""
        # Update params if provided
        if params is not None:
            self.update_video_params(params)
            
        # If direct pixel control is enabled, use that
        if self.direct_pixel_control:
            return self.get_direct_pixel_output(self.video_params['output_size'])
            
        # Otherwise, use the traditional neural visualization
        # Get raw neural map from visual processor
        neural_map = self.current_visual_encoding

        # Get structural plasticity visualization
        structure_viz = self._get_structural_visualization()
        
        # Combine into RGB output (neural activations in R, structure in G, empty B)
        video_output = np.zeros((*self.video_params['output_size'][::-1], 3), dtype=np.uint8)
        
        # Normalize and scale neural activations to red channel
        if neural_map is not None:
            neural_norm = (neural_map - neural_map.min()) / (neural_map.max() - neural_map.min() + 1e-8)
            resized_neural = cv2.resize(neural_norm, self.video_params['output_size'])
            video_output[..., 0] = (resized_neural * 255).astype(np.uint8)
        
        # Add structural visualization to green channel
        if structure_viz is not None:
            resized_structure = cv2.resize(structure_viz, self.video_params['output_size'])
            video_output[..., 1] = (resized_structure * 255).astype(np.uint8)
        
        return video_output

    def _get_structural_visualization(self):
        """Create visualization of model structure"""
        viz = np.zeros((100, 100))  # Base size will be resized
        
        # Visualize layer sizes
        layer_sizes = [
            self.visual_processor.visual_pathway.layers[-1].layer_size,
            self.audio_processor.audio_pathway.layers[-1].layer_size,
            self.multimodal_size
        ]
        
        # Create simple bar chart
        max_size = max(layer_sizes)
        viz = np.zeros((50, len(layer_sizes)*20))
        for i, size in enumerate(layer_sizes):
            height = int(40 * (size/max_size))
            viz[40-height:40, i*20:(i+1)*20] = 1.0
        
        # Add connection density
        if hasattr(self.multimodal_association, 'weights'):
            conn_density = np.mean(np.abs(self.multimodal_association.weights['visual']))
            viz[45:50, :int(conn_density*100)] = 0.5
        
        return viz 

    def learn_from_rgb_feedback(self, feedback_signal: float, learning_rate: Optional[float] = None):
        """Learn to adjust RGB output based on external feedback signal.
        
        This allows the system to learn from reinforcement signals to improve its
        RGB pixel control capability.
        
        Args:
            feedback_signal: A scalar feedback value between -1.0 (negative) and 1.0 (positive)
            learning_rate: Optional custom learning rate (defaults to rgb_learning_rate)
        """
        if not self.rgb_learning_enabled or not self.direct_pixel_control:
            return
            
        if learning_rate is None:
            learning_rate = self.rgb_learning_rate
        
        # Ensure feedback is in valid range
        feedback_signal = np.clip(feedback_signal, -1.0, 1.0)
        
        if self.current_multimodal_state is None:
            logger.warning("Cannot learn from feedback: no multimodal state available")
            return
            
        # Get current RGB output
        current_output = self.get_direct_pixel_output(self.video_params['output_size'])
        h, w = self.video_params['output_size'][::-1]
        total_pixels = h * w
        
        if self.enhanced_rgb_control:
            # Enhanced feedback approach with independent channel control
            
            # Calculate how much to adjust based on the feedback signal
            # Positive feedback reinforces current output, negative tries to move away from it
            adjustment_factor = feedback_signal * learning_rate
            
            # Apply feedback to weights - this reinforces or weakens the current patterns
            for modality in self.rgb_control_weights:
                if modality == 'multimodal' and self.current_multimodal_state is not None:
                    # For multimodal, use stronger adjustment since it has primary control
                    weight_change = adjustment_factor * 1.2 * np.outer(
                        self.current_multimodal_state, 
                        current_output.reshape(-1) / 255.0  # Normalize to 0-1
                    )
                    self.rgb_control_weights[modality] += weight_change
                    
                elif modality == 'visual' and self.current_visual_encoding is not None:
                    # For visual, use normal adjustment
                    weight_change = adjustment_factor * np.outer(
                        self.current_visual_encoding,
                        current_output.reshape(-1) / 255.0
                    )
                    self.rgb_control_weights[modality] += weight_change
                    
                elif modality == 'audio' and self.current_audio_encoding is not None:
                    # For audio, use weaker adjustment (less direct influence)
                    weight_change = adjustment_factor * 0.8 * np.outer(
                        self.current_audio_encoding,
                        current_output.reshape(-1) / 255.0
                    )
                    self.rgb_control_weights[modality] += weight_change
            
            # Apply feedback to the channel-specific controls for more precise adjustment
            current_flat = current_output.reshape(-1) / 255.0  # Flatten and normalize
            
            # Different adjustment for each channel - allows for interesting color shifts
            for c in range(3):
                # Get just this channel's values
                channel_values = current_flat[c::3]  # Every 3rd value starting from channel index
                
                # Different adjustment strength for each channel
                channel_adjustment = adjustment_factor * (1.0 + 0.2 * c)
                
                # For positive feedback, increase the gain for active pixels and decrease bias
                # For negative feedback, do the opposite
                self.rgb_channel_gain[c] += channel_adjustment * 0.1 * channel_values
                self.rgb_channel_bias[c] += channel_adjustment * 0.05 * (1.0 - channel_values)
                
                # Keep gain in reasonable range
                self.rgb_channel_gain[c] = np.clip(self.rgb_channel_gain[c], 0.2, 3.0)
            
            logger.debug(f"Applied enhanced RGB feedback with signal: {feedback_signal:.2f}")
            
        else:
            # Original approach - create target images
            
            # Modify weights based on feedback direction
            if feedback_signal > 0:
                # Positive feedback: Reinforce current output
                # Create a target slightly more extreme version of current output
                enhanced_output = current_output.copy()
                # Make bright areas brighter and dark areas darker
                bright_mask = current_output > 128
                enhanced_output[bright_mask] = np.clip(current_output[bright_mask] * 1.1, 0, 255)
                dark_mask = current_output <= 128
                enhanced_output[dark_mask] = np.clip(current_output[dark_mask] * 0.9, 0, 255)
                
                # Learn towards the enhanced version
                self.update_rgb_weights(enhanced_output, learning_rate * feedback_signal)
                
            elif feedback_signal < 0:
                # Negative feedback: Move away from current output
                # Create a more muted/neutral version as target
                muted_output = current_output.copy()
                # Make bright areas darker and dark areas brighter (move toward middle gray)
                bright_mask = current_output > 128
                muted_output[bright_mask] = np.clip(current_output[bright_mask] * 0.8 + 25, 0, 255)
                dark_mask = current_output <= 128
                muted_output[dark_mask] = np.clip(current_output[dark_mask] * 0.8 + 25, 0, 255)
                
                # Learn towards the muted version with negative feedback strength
                self.update_rgb_weights(muted_output, learning_rate * abs(feedback_signal))
            
            logger.debug(f"Applied standard RGB feedback with signal {feedback_signal:.2f}")

    def apply_rgb_mutation(self, mutation_rate: float = 0.1):
        """Apply random mutations to RGB control weights to encourage exploration.
        
        Args:
            mutation_rate: Rate of mutation (0.0-1.0)
        """
        if not self.rgb_learning_enabled or not self.direct_pixel_control:
            return
            
        # Clip mutation rate to valid range
        mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
        
        # Apply small random changes to weights
        for modality in self.rgb_control_weights:
            # Get random noise with same shape as weights
            noise = np.random.normal(0, 0.01, self.rgb_control_weights[modality].shape)
            
            # Apply noise scaled by mutation rate
            self.rgb_control_weights[modality] += noise * mutation_rate
        
        # Apply mutations to channel-specific controls if enhanced RGB is enabled
        if self.enhanced_rgb_control:
            for c in range(3):
                # Apply different mutation rates to different channels for more variety
                channel_mutation = mutation_rate * (0.8 + 0.4 * c / 3.0)
                
                # Mutate bias (more aggressive for better visible changes)
                bias_noise = np.random.normal(0, 0.03, self.rgb_channel_bias[c].shape)
                self.rgb_channel_bias[c] += bias_noise * channel_mutation
                
                # Mutate gain (more subtle)
                gain_noise = np.random.normal(0, 0.02, self.rgb_channel_gain[c].shape)
                self.rgb_channel_gain[c] += gain_noise * channel_mutation
                # Ensure gain stays positive
                self.rgb_channel_gain[c] = np.maximum(0.2, self.rgb_channel_gain[c])
            
        logger.debug(f"Applied RGB weight mutation with rate {mutation_rate:.2f}")
        
        # Get new output after mutation
        return self.get_direct_pixel_output(self.video_params['output_size'])

    def _get_connection_count(self) -> int:
        """Get the total number of connections in the multimodal association network"""
        total_connections = 0
        
        # Try the get_connection_stats method first
        if hasattr(self.multimodal_association, 'get_connection_stats'):
            try:
                return self.multimodal_association.get_connection_stats()['total']
            except (KeyError, AttributeError, TypeError):
                # Fall through to manual counting if method fails
                pass
                
        # Manual counting of non-zero weights
        if hasattr(self.multimodal_association, 'weights') and isinstance(self.multimodal_association.weights, dict):
            for modality, matrix in self.multimodal_association.weights.items():
                if matrix is not None:
                    total_connections += np.sum(matrix != 0)
        
        return total_connections 