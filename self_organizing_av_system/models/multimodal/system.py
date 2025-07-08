import numpy as np
import time
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any

from core.multimodal_association import MultimodalAssociation
from models.visual.processor import VisualProcessor
from models.audio.processor import AudioProcessor


class SelfOrganizingAVSystem:
    """
    The complete self-organizing audio-visual learning system.
    
    This system integrates visual and auditory pathways with multimodal
    association, implementing the biologically inspired architecture for
    autonomous learning from synchronized audio-visual inputs.
    """
    
    def __init__(
        self,
        visual_processor: Optional[VisualProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        multimodal_size: int = 100,
        prune_interval: int = 1000,
        structural_plasticity_interval: int = 5000,
        learning_rate: float = 0.01,
        learning_rule: str = 'oja'
    ):
        """
        Initialize the complete audio-visual learning system.
        
        Args:
            visual_processor: Visual pathway processor (or None to create default)
            audio_processor: Audio pathway processor (or None to create default)
            multimodal_size: Size of the multimodal association layer
            prune_interval: Frequency of synaptic pruning (in frames)
            structural_plasticity_interval: Frequency of structural updates (in frames)
            learning_rate: Global learning rate
            learning_rule: Default learning rule ('hebbian', 'oja', or 'stdp')
        """
        self.multimodal_size = multimodal_size
        self.prune_interval = prune_interval
        self.structural_plasticity_interval = structural_plasticity_interval
        self.learning_rate = learning_rate
        self.learning_rule = learning_rule
        
        # Initialize processors if not provided
        if visual_processor is None:
            self.visual_processor = VisualProcessor()
        else:
            self.visual_processor = visual_processor
        
        if audio_processor is None:
            self.audio_processor = AudioProcessor()
        else:
            self.audio_processor = audio_processor
        
        # Create multimodal association
        # Get output sizes from the top layers of each pathway
        visual_output_size = self.visual_processor.visual_pathway.layers[-1].layer_size
        audio_output_size = self.audio_processor.audio_pathway.layers[-1].layer_size
        
        modality_sizes = {
            'visual': visual_output_size,
            'audio': audio_output_size
        }
        
        self.multimodal = MultimodalAssociation(
            modality_sizes=modality_sizes,
            association_size=multimodal_size,
            learning_rate=learning_rate * 0.5,  # Lower learning rate for associations
            association_threshold=0.1,
            use_sparse_coding=True
        )
        
        # System state
        self.frame_count = 0
        self.time_step = 0
        self.is_learning = True
        self.recent_prediction_errors = []
        self.start_time = time.time()
        self.processing_times = []
        
        # Monitoring metrics
        self.metrics = {
            'prediction_error': [],
            'structural_changes': [],
            'pruning_stats': [],
            'cross_modal_strength': []
        }
        
        # Logger setup
        self.logger = logging.getLogger('AVSystem')
        self.logger.setLevel(logging.INFO)
    
    def process_av_pair(
        self,
        video_frame: np.ndarray,
        audio_chunk: np.ndarray,
        learn: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process a synchronized audio-visual pair through the system.
        
        Args:
            video_frame: RGB video frame
            audio_chunk: Raw audio samples
            learn: Whether to apply learning
            
        Returns:
            Dictionary with activation patterns of different components
        """
        start_time = time.time()
        
        # Process video frame
        visual_activations = self.visual_processor.process_frame(
            video_frame, time_step=self.time_step
        )
        
        # Process audio chunk
        audio_activations = self.audio_processor.process_audio_chunk(
            audio_chunk, time_step=self.time_step
        )
        
        # Process through multimodal layer
        # First, prepare modality activities dictionary
        modality_activities = {
            'visual': visual_activations,
            'audio': audio_activations
        }
        
        # Update multimodal associations
        multimodal_result = self.multimodal.update(
            modality_activities=modality_activities,
            learning_enabled=learn and self.is_learning
        )
        multimodal_activations = multimodal_result['association_activity']
        
        # Apply learning if enabled
        if learn and self.is_learning:
            self.visual_processor.learn(self.learning_rule)
            self.audio_processor.learn(self.learning_rule)
            
            # Periodic maintenance operations
            self._apply_periodic_maintenance()
        
        # Increment counters
        self.frame_count += 1
        self.time_step += 1
        
        # Record processing time
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        
        # Return current activation state
        return {
            'visual': visual_activations,
            'audio': audio_activations,
            'multimodal': multimodal_activations
        }
    
    def process_video_sequence(
        self,
        video_frames: List[np.ndarray],
        audio_waveform: np.ndarray,
        sample_rate: int,
        learn: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        Process a complete audio-visual sequence.
        
        Args:
            video_frames: List of video frames
            audio_waveform: Complete audio waveform
            sample_rate: Audio sample rate
            learn: Whether to apply learning
            
        Returns:
            List of activation state dictionaries, one per frame
        """
        # Extract audio processing parameters
        window_size = self.audio_processor.window_size
        hop_length = self.audio_processor.hop_length
        
        # Process full audio waveform
        audio_activations = self.audio_processor.process_waveform(
            audio_waveform, time_step=self.time_step
        )
        
        # Match video frames to audio segments
        # Simple approach: assume frames and spectrogram frames are pre-aligned
        # For a real system, this would need proper audio-video synchronization
        n_frames = len(video_frames)
        n_audio_frames = len(audio_activations)
        
        if n_frames != n_audio_frames:
            self.logger.warning(
                f"Video frames ({n_frames}) don't match audio frames ({n_audio_frames}). " +
                "Using shorter length and skipping excess."
            )
        
        # Process frames and corresponding audio
        combined_length = min(n_frames, n_audio_frames)
        activation_sequence = []
        
        for i in range(combined_length):
            # Set the current audio activations (already processed)
            self.audio_processor.audio_pathway.layers[-1].activations = audio_activations[i]
            
            # Process the corresponding video frame
            visual_activations = self.visual_processor.process_frame(
                video_frames[i], time_step=self.time_step
            )
            
            # Process through multimodal layer
            # Prepare modality activities dictionary
            modality_activities = {
                'visual': visual_activations,
                'audio': audio_activations[i]
            }
            
            # Update multimodal associations
            multimodal_result = self.multimodal.update(
                modality_activities=modality_activities,
                learning_enabled=learn and self.is_learning
            )
            multimodal_activations = multimodal_result['association_activity']
            
            # Apply learning if enabled
            if learn and self.is_learning:
                self.visual_processor.learn(self.learning_rule)
                # Note: Audio already processed, don't learn twice
                
                # Periodic maintenance operations
                self._apply_periodic_maintenance()
            
            # Record activations
            activation_sequence.append({
                'visual': visual_activations,
                'audio': audio_activations[i],
                'multimodal': multimodal_activations,
                'time_step': self.time_step
            })
            
            # Increment time step
            self.time_step += 1
        
        # Update frame count
        self.frame_count += combined_length
        
        return activation_sequence
    
    def predict_next_av_pair(self) -> Dict[str, np.ndarray]:
        """
        Predict the next frame's activations based on current state.
        
        Returns:
            Dictionary with predicted activation patterns
        """
        # Get predictions from each pathway
        visual_prediction = self.visual_processor.visual_pathway.predict_next()[0]
        audio_prediction = self.audio_processor.audio_pathway.predict_next()[0]
        
        return {
            'visual_prediction': visual_prediction,
            'audio_prediction': audio_prediction
        }
    
    def generate_from_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Generate expected visual activations from audio input only.
        
        This simulates cross-modal generation, like imagining visual content
        when hearing a sound.
        
        Args:
            audio_chunk: Audio input
            
        Returns:
            Predicted visual activations
        """
        # Process audio through its pathway
        audio_activations = self.audio_processor.process_audio_chunk(
            audio_chunk, time_step=None
        )
        
        # Get current visual activations (to be influenced)
        current_visual = np.zeros_like(self.visual_processor.visual_pathway.layers[-1].activations)
        self.visual_processor.visual_pathway.layers[-1].activations = current_visual
        
        # Run multimodal processing - cross-modal weights will influence visual
        modality_activities = {
            'audio': audio_activations,
            'visual': current_visual
        }
        self.multimodal.update(modality_activities=modality_activities, learning_enabled=False)
        
        # Return the influenced visual activations
        return self.visual_processor.visual_pathway.layers[-1].activations.copy()
    
    def generate_from_visual(self, video_frame: np.ndarray) -> np.ndarray:
        """
        Generate expected audio activations from visual input only.
        
        This simulates cross-modal generation, like imagining a sound
        when seeing a visual pattern.
        
        Args:
            video_frame: Visual input
            
        Returns:
            Predicted audio activations
        """
        # Process visual through its pathway
        visual_activations = self.visual_processor.process_frame(
            video_frame, time_step=None
        )
        
        # Get current audio activations (to be influenced)
        current_audio = np.zeros_like(self.audio_processor.audio_pathway.layers[-1].activations)
        self.audio_processor.audio_pathway.layers[-1].activations = current_audio
        
        # Run multimodal processing - cross-modal weights will influence audio
        modality_activities = {
            'visual': visual_activations,
            'audio': current_audio
        }
        self.multimodal.update(modality_activities=modality_activities, learning_enabled=False)
        
        # Return the influenced audio activations
        return self.audio_processor.audio_pathway.layers[-1].activations.copy()
    
    def _apply_periodic_maintenance(self) -> None:
        """Apply periodic maintenance operations like pruning and structural plasticity."""
        # Prune weak connections
        if self.frame_count % self.prune_interval == 0:
            pruning_stats = self._prune_connections()
            self.metrics['pruning_stats'].append((self.frame_count, pruning_stats))
            self.logger.info(f"Applied pruning at frame {self.frame_count}")
        
        # Apply structural plasticity
        if self.frame_count % self.structural_plasticity_interval == 0:
            structural_changes = self._apply_structural_plasticity()
            self.metrics['structural_changes'].append((self.frame_count, structural_changes))
            self.logger.info(f"Applied structural plasticity at frame {self.frame_count}")
        
        # Update cross-modal strength metrics
        if self.frame_count % 1000 == 0:
            strengths = self._get_cross_modal_strength()
            self.metrics['cross_modal_strength'].append((self.frame_count, strengths))
    
    def _prune_connections(self, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Prune weak connections throughout the network.
        
        Args:
            threshold: Pruning threshold
            
        Returns:
            Dictionary with pruning statistics
        """
        # Prune visual pathway
        visual_stats = self.visual_processor.visual_pathway.prune_pathway(threshold)
        
        # Prune audio pathway
        audio_stats = self.audio_processor.audio_pathway.prune_pathway(threshold)
        
        # Prune multimodal connections
        # The MultimodalAssociation class doesn't have a prune_connections method
        # We'll track this as not pruned for now
        multimodal_stats = {'pruned': 0}
        
        return {
            'visual': visual_stats,
            'audio': audio_stats,
            'multimodal': multimodal_stats
        }
    
    def _apply_structural_plasticity(self) -> Dict[str, Any]:
        """
        Apply structural plasticity through neuron addition and replacement.
        
        Returns:
            Dictionary with structural change statistics
        """
        # Add neurons where needed
        visual_additions = self.visual_processor.visual_pathway.add_neurons_where_needed()
        audio_additions = self.audio_processor.audio_pathway.add_neurons_where_needed()
        
        # Replace dead neurons
        visual_replacements = self.visual_processor.visual_pathway.replace_dead_neurons()
        audio_replacements = self.audio_processor.audio_pathway.replace_dead_neurons()
        
        return {
            'visual_additions': visual_additions,
            'audio_additions': audio_additions,
            'visual_replacements': visual_replacements,
            'audio_replacements': audio_replacements
        }
    
    def _get_cross_modal_strength(self) -> Dict[str, float]:
        """
        Calculate overall strength of cross-modal associations.
        
        Returns:
            Dictionary with cross-modal strength metrics
        """
        # Get connection strengths from forward weights
        connections = self.multimodal.weights
        
        strengths = {}
        for key, matrix in connections.items():
            # Calculate average absolute weight
            avg_strength = np.mean(np.abs(matrix))
            # Calculate sparsity (fraction of non-zero weights)
            sparsity = np.mean(matrix != 0)
            
            strengths[f"{key}_strength"] = float(avg_strength)
            strengths[f"{key}_sparsity"] = float(sparsity)
        
        return strengths
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current state of the entire system.
        
        Returns:
            Dictionary with comprehensive system state
        """
        visual_state = self.visual_processor.get_pathway_state()
        audio_state = self.audio_processor.get_pathway_state()
        
        multimodal_activity = self.multimodal.get_multimodal_activity()
        # Get multimodal statistics
        associations = self.multimodal.get_stats()
        
        # Calculate average processing time
        avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
        
        return {
            'frame_count': self.frame_count,
            'time_step': self.time_step,
            'run_time': time.time() - self.start_time,
            'avg_processing_time': avg_time,
            'is_learning': self.is_learning,
            'visual': visual_state,
            'audio': audio_state,
            'multimodal_activity': multimodal_activity,
            'associations': associations,
            'metrics': self.metrics
        }
    
    def save_state(self, filename: str) -> bool:
        """
        Save the system state to a file.
        
        Args:
            filename: Path to save the state file
            
        Returns:
            Whether saving was successful
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Extract serializable state
            state = {
                # System general state
                'frame_count': self.frame_count,
                'time_step': self.time_step,
                'multimodal_size': self.multimodal_size,
                'prune_interval': self.prune_interval,
                'structural_plasticity_interval': self.structural_plasticity_interval,
                'learning_rate': self.learning_rate,
                'learning_rule': self.learning_rule,
                'is_learning': self.is_learning,
                'metrics': self.metrics,
                'recent_prediction_errors': self.recent_prediction_errors,
                
                # Visual pathway state
                'visual_processor': {
                    # Pathway configuration
                    'input_width': self.visual_processor.input_width,
                    'input_height': self.visual_processor.input_height,
                    'use_grayscale': self.visual_processor.use_grayscale,
                    'patch_size': self.visual_processor.patch_size,
                    'stride': self.visual_processor.stride,
                    'contrast_normalize': self.visual_processor.contrast_normalize,
                    
                    # Layers state
                    'pathway': self._serialize_pathway(self.visual_processor.visual_pathway)
                },
                
                # Audio pathway state
                'audio_processor': {
                    # Pathway configuration
                    'sample_rate': self.audio_processor.sample_rate,
                    'window_size': self.audio_processor.window_size,
                    'hop_length': self.audio_processor.hop_length,
                    'n_mels': self.audio_processor.n_mels,
                    'min_freq': self.audio_processor.min_freq,
                    'max_freq': self.audio_processor.max_freq,
                    'normalize': self.audio_processor.normalize,
                    
                    # Layers state
                    'pathway': self._serialize_pathway(self.audio_processor.audio_pathway)
                },
                
                # Multimodal associations state
                'multimodal': self.multimodal.serialize()
            }
            
            # Save to file using pickle (handles numpy arrays)
            with open(filename, 'wb') as f:
                pickle.dump(state, f, protocol=4)  # Protocol 4 for compatibility
            
            self.logger.info(f"Saved system state to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state to {filename}: {e}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load the system state from a file.
        
        Args:
            filename: Path to the state file
            
        Returns:
            Whether loading was successful
        """
        try:
            if not os.path.exists(filename):
                self.logger.error(f"State file not found: {filename}")
                return False
            
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            # Update system state variables
            self.frame_count = state['frame_count']
            self.time_step = state['time_step']
            self.multimodal_size = state['multimodal_size']
            self.prune_interval = state['prune_interval']
            self.structural_plasticity_interval = state['structural_plasticity_interval']
            self.learning_rate = state['learning_rate']
            self.learning_rule = state['learning_rule']
            self.is_learning = state['is_learning']
            self.metrics = state['metrics']
            self.recent_prediction_errors = state['recent_prediction_errors']
            
            # Restore visual pathway state
            self._restore_pathway(self.visual_processor.visual_pathway, state['visual_processor']['pathway'])
            
            # Restore audio pathway state
            self._restore_pathway(self.audio_processor.audio_pathway, state['audio_processor']['pathway'])
            
            # Restore multimodal state
            self.multimodal = MultimodalAssociation.deserialize(state['multimodal'])
            
            self.logger.info(f"Loaded system state from {filename}")
            self.logger.info(f"Resumed at frame {self.frame_count}, time step {self.time_step}")
            
            # Reset start time to current time
            self.start_time = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state from {filename}: {e}")
            return False
    
    def _serialize_pathway(self, pathway) -> Dict:
        """Helper to serialize a neural pathway."""
        return {
            'name': pathway.name,
            'input_size': pathway.input_size,
            'num_layers': pathway.num_layers,
            'layers': [self._serialize_layer(layer) for layer in pathway.layers],
            'error_history': pathway.error_history
        }
    
    def _serialize_layer(self, layer) -> Dict:
        """Helper to serialize a neural layer."""
        return {
            'name': layer.name,
            'input_size': layer.input_size,
            'layer_size': layer.layer_size,
            'learning_rate': layer.learning_rate,
            'target_activation': layer.target_activation,
            'homeostatic_factor': layer.homeostatic_factor,
            'k_winners': layer.k_winners,
            'activations': layer.activations,
            'activations_raw': layer.activations_raw,
            'weights': [neuron.weights for neuron in layer.neurons],
            'thresholds': [neuron.threshold for neuron in layer.neurons],
            'recent_mean_activations': [neuron.recent_mean_activation for neuron in layer.neurons],
            'recurrent_weights': layer.recurrent_weights if hasattr(layer, 'recurrent_weights') else None,
            'mean_activation_history': layer.mean_activation_history,
            'sparsity_history': layer.sparsity_history
        }
    
    def _restore_pathway(self, pathway, state) -> None:
        """Helper to restore a neural pathway from state."""
        pathway.name = state['name']
        pathway.input_size = state['input_size']
        pathway.num_layers = state['num_layers']
        pathway.error_history = state['error_history']
        
        for i, layer_state in enumerate(state['layers']):
            self._restore_layer(pathway.layers[i], layer_state)
    
    def _restore_layer(self, layer, state) -> None:
        """Helper to restore a neural layer from state."""
        layer.name = state['name']
        layer.input_size = state['input_size']
        layer.layer_size = state['layer_size']
        layer.learning_rate = state['learning_rate']
        layer.target_activation = state['target_activation']
        layer.homeostatic_factor = state['homeostatic_factor']
        layer.k_winners = state['k_winners']
        layer.activations = state['activations']
        layer.activations_raw = state['activations_raw']
        layer.mean_activation_history = state['mean_activation_history']
        layer.sparsity_history = state['sparsity_history']
        
        # Restore neuron weights
        for i, weights in enumerate(state['weights']):
            if i < len(layer.neurons):
                layer.neurons[i].weights = weights
        
        # Restore neuron thresholds
        for i, threshold in enumerate(state['thresholds']):
            if i < len(layer.neurons):
                layer.neurons[i].threshold = threshold
        
        # Restore neuron activation stats
        for i, mean_act in enumerate(state['recent_mean_activations']):
            if i < len(layer.neurons):
                layer.neurons[i].recent_mean_activation = mean_act
        
        # Restore recurrent weights if present
        if state['recurrent_weights'] is not None and hasattr(layer, 'recurrent_weights'):
            layer.recurrent_weights = state['recurrent_weights']
    
    def __repr__(self) -> str:
        return (f"SelfOrganizingAVSystem(frames={self.frame_count}, " +
                f"associations={self.multimodal_size})") 