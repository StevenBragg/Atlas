import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import logging

# Import directly from the pathway module instead of through core
from core.pathway import NeuralPathway


class VisualProcessor:
    """
    Processes video frames and feeds them through a visual neural pathway.
    
    This module handles preprocessing of raw video frames into suitable inputs
    for the self-organizing neural network.
    """
    
    def __init__(
        self,
        input_width: int = 64,
        input_height: int = 64,
        use_grayscale: bool = True,
        patch_size: int = 8,
        stride: int = 4,
        contrast_normalize: bool = True,
        layer_sizes: Optional[List[int]] = None
    ):
        """
        Initialize the visual processor.
        
        Args:
            input_width: Width to resize input frames to
            input_height: Height to resize input frames to
            use_grayscale: Whether to convert frames to grayscale
            patch_size: Size of patches for local feature extraction
            stride: Stride for patch extraction
            contrast_normalize: Whether to apply contrast normalization
            layer_sizes: List of neuron counts for each neural layer
        """
        self.input_width = input_width
        self.input_height = input_height
        self.use_grayscale = use_grayscale
        self.patch_size = patch_size
        self.stride = stride
        self.contrast_normalize = contrast_normalize
        
        # Calculate derived parameters
        self.num_channels = 1 if use_grayscale else 3
        self.patches_w = (input_width - patch_size) // stride + 1
        self.patches_h = (input_height - patch_size) // stride + 1
        self.num_patches = self.patches_w * self.patches_h
        self.patch_dim = patch_size * patch_size * self.num_channels
        
        # Prepare patch extraction
        self.patch_coords = self._precompute_patch_coordinates()
        
        # Configure network layers if not specified
        if layer_sizes is None:
            # Set up a standard hierarchy for visual processing
            layer_sizes = [
                200,  # First-level features (like edge detectors)
                100,  # Mid-level features (shape components)
                50    # High-level features (object/scene features)
            ]
        
        # Create the neural pathway
        self.visual_pathway = NeuralPathway(
            name="visual",
            input_size=self.patch_dim,
            layer_sizes=layer_sizes,
            use_recurrent=False
        )
        
        # Tracking statistics
        self.frame_count = 0
        self.current_frame = None
        self.current_patches = None
        self.previous_frame = None
        self.motion_features = None
    
    def process_frame(self, frame: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Process a video frame through the visual pathway.
        
        Args:
            frame: RGB video frame
            time_step: Current simulation time step
            
        Returns:
            Activation vector of the top layer of the visual pathway
        """
        # Save the previous frame
        self.previous_frame = self.current_frame.copy() if self.current_frame is not None else None
        
        # Update current frame
        self.current_frame = self._preprocess_frame(frame)
        
        # Extract patches
        self.current_patches = self._extract_patches(self.current_frame)
        
        # Calculate basic motion features if we have a previous frame
        if self.previous_frame is not None:
            self.motion_features = self._compute_motion_features(self.previous_frame, self.current_frame)
        
        # Process all patches through the pathway
        all_patch_activations = []
        
        for patch in self.current_patches:
            # Normalize patch to unit norm
            patch_normalized = patch / (np.linalg.norm(patch) + 1e-10)
            
            # Process through visual pathway
            activations = self.visual_pathway.process(patch_normalized, time_step)
            all_patch_activations.append(activations)
        
        # Increment frame count
        self.frame_count += 1
        
        # Return the average top-level activations across all patches
        return np.mean(all_patch_activations, axis=0)
    
    def learn(self, learning_rule: str = 'oja') -> None:
        """
        Apply learning to the visual pathway.
        
        Args:
            learning_rule: Learning rule to use ('hebbian', 'oja', or 'stdp')
        """
        self.visual_pathway.learn(learning_rule)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for input to the neural network.
        
        Args:
            frame: Raw RGB video frame
            
        Returns:
            Preprocessed frame
        """
        # Resize to input dimensions
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert to grayscale if specified
        if self.use_grayscale:
            if len(resized.shape) == 3:
                processed = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            else:
                processed = resized  # Already grayscale
        else:
            processed = resized
        
        # Normalize pixel values to [0, 1]
        processed = processed.astype(np.float32) / 255.0
        
        # Apply contrast normalization if specified
        if self.contrast_normalize:
            # Local contrast normalization
            mean = np.mean(processed)
            std = np.std(processed)
            if std > 0:
                processed = (processed - mean) / std
                # Rescale to [0, 1]
                processed = (processed - np.min(processed)) / (np.max(processed) - np.min(processed) + 1e-10)
        
        return processed
    
    def _precompute_patch_coordinates(self) -> List[Tuple[int, int]]:
        """
        Precompute the coordinates for patch extraction.
        
        Returns:
            List of (row, col) coordinates for top-left corner of each patch
        """
        coords = []
        for i in range(0, self.input_height - self.patch_size + 1, self.stride):
            for j in range(0, self.input_width - self.patch_size + 1, self.stride):
                coords.append((i, j))
        return coords
    
    def _extract_patches(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract patches from a preprocessed frame.
        
        Args:
            frame: Preprocessed video frame
            
        Returns:
            List of flattened patch vectors
        """
        patches = []
        
        # Handle both grayscale and color frames
        if len(frame.shape) == 2:  # Grayscale
            for i, j in self.patch_coords:
                patch = frame[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch.flatten())
        else:  # Color
            for i, j in self.patch_coords:
                patch = frame[i:i+self.patch_size, j:j+self.patch_size, :]
                patches.append(patch.flatten())
        
        return patches
    
    def _compute_motion_features(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute simple motion features between frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            Motion feature map
        """
        # Simple frame difference as motion detection
        diff = np.abs(curr_frame - prev_frame)
        
        # For color frames, take max across channels
        if len(diff.shape) == 3:
            diff = np.max(diff, axis=2)
        
        return diff
    
    def get_receptive_fields(self, layer_idx: int = 0) -> np.ndarray:
        """
        Get receptive fields of neurons in the specified layer.
        
        Args:
            layer_idx: Index of the layer to get receptive fields from
            
        Returns:
            Matrix of receptive fields, one per row
        """
        return self.visual_pathway.layers[layer_idx].get_all_receptive_fields()
    
    def visualize_receptive_field(self, layer_idx: int, neuron_idx: int) -> np.ndarray:
        """
        Create a visual representation of a neuron's receptive field.
        
        Args:
            layer_idx: Index of the layer
            neuron_idx: Index of the neuron within the layer
            
        Returns:
            Image representation of the receptive field
        """
        # Check if indices are valid
        if layer_idx < 0 or layer_idx >= len(self.visual_pathway.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer = self.visual_pathway.layers[layer_idx]
        if neuron_idx < 0 or neuron_idx >= layer.layer_size:
            raise ValueError(f"Invalid neuron index: {neuron_idx}")
        
        # Get the receptive field
        rf = layer.neurons[neuron_idx].get_receptive_field()
        
        # For layer 0, we can visualize directly as a patch
        if layer_idx == 0:
            # Reshape to patch dimensions
            if self.use_grayscale:
                rf_image = rf.reshape(self.patch_size, self.patch_size)
            else:
                rf_image = rf.reshape(self.patch_size, self.patch_size, self.num_channels)
            
            # Normalize for visualization
            rf_image = (rf_image - np.min(rf_image)) / (np.max(rf_image) - np.min(rf_image) + 1e-10)
            
            return rf_image
        else:
            # For higher layers, use activation maximization approach
            # Create a synthetic input that maximally activates this neuron
            
            # Get the pathway up to the specified layer
            pathway_layers = self.visual_pathway.layers[:layer_idx + 1]
            
            # Start with random input
            rf_input = np.random.randn(self.input_width, self.input_height, self.num_channels) * 0.1
            
            # Gradient ascent to maximize the neuron's activation
            learning_rate = 0.1
            iterations = 50
            
            for _ in range(iterations):
                # Forward pass through layers
                current = rf_input.reshape(-1)
                for i, layer in enumerate(pathway_layers):
                    weights = layer.get_weight_matrix()
                    current = np.dot(weights, current)
                    # Apply activation (ReLU)
                    current = np.maximum(0, current)
                
                # Get activation of target neuron
                if neuron_idx < len(current):
                    activation = current[neuron_idx]
                    
                    # Approximate gradient via finite differences
                    epsilon = 0.001
                    grad = np.zeros_like(rf_input)
                    
                    for i in range(rf_input.size):
                        # Perturb input
                        rf_input_flat = rf_input.flatten()
                        rf_input_flat[i] += epsilon
                        rf_input_plus = rf_input_flat.reshape(rf_input.shape)
                        
                        # Forward pass with perturbation
                        current_plus = rf_input_plus.reshape(-1)
                        for layer in pathway_layers:
                            weights = layer.get_weight_matrix()
                            current_plus = np.dot(weights, current_plus)
                            current_plus = np.maximum(0, current_plus)
                        
                        # Compute gradient
                        if neuron_idx < len(current_plus):
                            grad_flat = grad.flatten()
                            grad_flat[i] = (current_plus[neuron_idx] - activation) / epsilon
                    
                    # Update input
                    rf_input += learning_rate * grad
                    
                    # Constrain to valid range
                    rf_input = np.clip(rf_input, 0, 1)
            
            # Convert to displayable format
            if self.use_grayscale:
                rf_image = (rf_input * 255).astype(np.uint8)
            else:
                rf_image = (rf_input * 255).astype(np.uint8)
            
            # Resize for better visualization
            rf_image = cv2.resize(rf_image, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            return rf_image
    
    def get_pathway_state(self) -> Dict:
        """
        Get the current state of the visual pathway.
        
        Returns:
            Dictionary with pathway state information
        """
        return self.visual_pathway.get_pathway_state()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the visual processor.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            "frames_processed": self.frame_count,
            "input_dimensions": f"{self.input_width}x{self.input_height}",
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "layer_sizes": [layer.layer_size for layer in self.visual_pathway.layers],
            "use_grayscale": self.use_grayscale
        }
    
    def __repr__(self) -> str:
        return f"VisualProcessor(input={self.input_width}x{self.input_height}, patches={self.num_patches})" 