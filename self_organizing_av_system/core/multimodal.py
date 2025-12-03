from typing import List, Dict, Tuple, Optional, Union, Any
import logging

from .backend import xp, to_cpu
from .neuron import Neuron
from .layer import NeuralLayer
from .pathway import NeuralPathway


class LegacyMultimodalAssociation:
    """
    LEGACY IMPLEMENTATION - Kept for reference purposes only.
    
    This is an older implementation of the multimodal association component that
    works with the NeuralPathway, NeuralLayer, and Neuron classes. It is not compatible
    with the current system architecture which uses the newer MultimodalAssociation
    class from multimodal_association.py.
    
    DO NOT USE THIS CLASS in the current system implementation.
    
    Connects multiple sensory pathways to form cross-modal associations.
    
    This implements the cross-modal binding layer that discovers correlations
    between different sensory modalities and forms associations.
    """
    
    def __init__(
        self,
        pathways: List[NeuralPathway],
        association_layer_size: int = 100,
        association_sparsity: float = 0.1,
        correlation_threshold: float = 0.3,
        learning_rate: float = 0.005,
        reentrant_connections: bool = True,
        bidirectional_association: bool = True
    ):
        """
        Initialize the multimodal association component.
        
        Args:
            pathways: List of sensory pathways to connect
            association_layer_size: Number of neurons in the multimodal layer
            association_sparsity: Target sparsity for multimodal layer
            correlation_threshold: Threshold for forming cross-modal links
            learning_rate: Learning rate for association weights
            reentrant_connections: Whether to use reentrant processing
            bidirectional_association: Whether associations are bidirectional
        """
        self.pathways = pathways
        self.num_pathways = len(pathways)
        self.correlation_threshold = correlation_threshold
        self.learning_rate = learning_rate
        self.association_sparsity = association_sparsity
        self.reentrant_connections = reentrant_connections
        self.bidirectional_association = bidirectional_association
        
        # Extract top-level features from each pathway
        self.pathway_output_sizes = [pathway.layers[-1].layer_size for pathway in pathways]
        self.pathway_names = [pathway.name for pathway in pathways]
        
        # Create a multimodal layer that receives inputs from all pathways
        total_input_size = sum(self.pathway_output_sizes)
        k_winners = max(1, int(association_layer_size * association_sparsity))
        
        # Association layer connects all pathways
        self.association_layer = NeuralLayer(
            input_size=total_input_size,
            layer_size=association_layer_size,
            name="multimodal_association",
            learning_rate=learning_rate,
            k_winners=k_winners
        )
        
        # Track where each pathway's inputs start in the concatenated input
        self.pathway_input_start_indices = []
        start_idx = 0
        for size in self.pathway_output_sizes:
            self.pathway_input_start_indices.append(start_idx)
            start_idx += size
        
        # Direct cross-modal connections between pathways
        # For each pair of pathways, create a connection matrix
        self.cross_modal_connections = {}
        for i in range(self.num_pathways):
            for j in range(self.num_pathways):
                if i != j:  # Skip self-connections
                    key = f"{self.pathway_names[i]}_to_{self.pathway_names[j]}"
                    # Initially very small random weights
                    self.cross_modal_connections[key] = xp.random.normal(
                        0, 0.01, 
                        (self.pathway_output_sizes[i], self.pathway_output_sizes[j])
                    )
        
        # For tracking statistics on connections
        self.connection_statistics = {
            'connection_count': {},
            'connection_strength': {},
            'recent_activations': {}
        }
        for i in range(self.num_pathways):
            for j in range(self.num_pathways):
                if i != j:
                    key = f"{self.pathway_names[i]}_to_{self.pathway_names[j]}"
                    self.connection_statistics['connection_count'][key] = 0
                    self.connection_statistics['connection_strength'][key] = 0.0
                    self.connection_statistics['recent_activations'][key] = []
        
        # Temporal context for prediction
        self.last_activations = {}
        for i in range(self.num_pathways):
            self.last_activations[self.pathway_names[i]] = None
        
        # Initialize multimodal recurrent connections for sequence learning
        if reentrant_connections:
            self.association_layer.init_recurrent_connections()
        
        # Logger setup
        self.logger = logging.getLogger('LegacyMultimodalAssociation')
    
    def process(self, time_step: Optional[int] = None) -> xp.ndarray:
        """
        Process top-layer activations from all pathways through the multimodal layer.
        
        Args:
            time_step: Current simulation time step
            
        Returns:
            Activations of the multimodal layer
        """
        # Collect activations from each pathway's top layer
        pathway_activations = []
        for pathway in self.pathways:
            # Get the top layer activations
            top_layer_activations = pathway.layers[-1].activations
            pathway_activations.append(top_layer_activations)
            
            # Store for temporal processing
            self.last_activations[pathway.name] = top_layer_activations.copy()
        
        # Process direct cross-modal connections first (predictive activations)
        self._process_cross_modal_connections()
        
        # Concatenate all pathway activations for the multimodal layer
        multimodal_input = xp.concatenate(pathway_activations)
        
        # Process through multimodal association layer
        multimodal_activations = self.association_layer.activate(multimodal_input, time_step)
        
        # If reentrant connections are enabled, update pathway activations
        # This implements top-down influence from multimodal to unimodal
        if self.reentrant_connections:
            self._apply_reentrant_processing(multimodal_activations)
        
        return multimodal_activations
    
    def _process_cross_modal_connections(self) -> None:
        """
        Apply direct cross-modal connections between pathways.
        
        This implements the multimodal prediction: activity in one pathway
        can drive activity in another pathway through learned associations.
        """
        # Process each pair of pathways
        for i in range(self.num_pathways):
            for j in range(self.num_pathways):
                if i == j:  # Skip self-connections
                    continue
                
                # Get pathway names
                source_name = self.pathway_names[i]
                target_name = self.pathway_names[j]
                
                # Get activations
                source_activations = self.last_activations[source_name]
                if source_activations is None:
                    continue
                
                # Get connection matrix
                connection_key = f"{source_name}_to_{target_name}"
                connection_matrix = self.cross_modal_connections[connection_key]
                
                # Calculate cross-modal influence
                if target_name in self.last_activations and self.last_activations[target_name] is not None:
                    # Apply influence to target pathway's activations
                    # This is a simplified form of predictive coding
                    target_influence = xp.dot(source_activations, connection_matrix)
                    
                    # Update target pathway's layer directly
                    target_pathway = self.pathways[j]
                    target_layer = target_pathway.layers[-1]
                    
                    # Blend with existing activations - this is an expectation signal
                    # In a more sophisticated model, this would be an explicit prediction
                    # compared against actual input, but here we just blend
                    if hasattr(target_layer, 'prediction_signal'):
                        # If layer has explicit support, use it
                        target_layer.prediction_signal = target_influence
                    else:
                        # Otherwise blend with current activations
                        # Small weight to avoid dominating actual sensory input
                        influence_weight = 0.2
                        target_layer.activations = (
                            (1.0 - influence_weight) * target_layer.activations + 
                            influence_weight * target_influence
                        )
                    
                    # Track statistics
                    if len(self.connection_statistics['recent_activations'][connection_key]) > 10:
                        self.connection_statistics['recent_activations'][connection_key].pop(0)
                    self.connection_statistics['recent_activations'][connection_key].append(
                        xp.mean(xp.abs(target_influence))
                    )
    
    def _apply_reentrant_processing(self, multimodal_activations: xp.ndarray) -> None:
        """
        Apply top-down influence from multimodal layer to individual pathways.
        
        Args:
            multimodal_activations: Current activations of the multimodal layer
        """
        # The multimodal layer has weights connecting to all pathways.
        # We can use these to send top-down signals.
        # This is a form of expectation or attention signal.
        
        # Get weights from multimodal layer
        multimodal_weights = xp.array([n.weights for n in self.association_layer.neurons])
        
        # For each pathway, calculate the influence
        for i, pathway in enumerate(self.pathways):
            # Extract weights connecting to this pathway
            start_idx = self.pathway_input_start_indices[i]
            end_idx = start_idx + self.pathway_output_sizes[i]
            
            # Weights from multimodal to this pathway's section
            pathway_weights = multimodal_weights[:, start_idx:end_idx]
            
            # Calculate top-down influence (transpose to get right dimensions)
            top_down_signal = xp.dot(multimodal_activations, pathway_weights)
            
            # Apply to pathway's top layer
            top_layer = pathway.layers[-1]
            
            # Blend with existing activations (weak influence)
            # In a more sophisticated model, this would be part of predictive coding
            top_down_weight = 0.15  # Weaker than direct cross-modal
            
            # Apply the top-down influence
            if hasattr(top_layer, 'top_down_signal'):
                # If layer has explicit support, use it
                top_layer.top_down_signal = top_down_signal
            else:
                # Otherwise blend with current activations
                top_layer.activations = (
                    (1.0 - top_down_weight) * top_layer.activations + 
                    top_down_weight * top_down_signal
                )
    
    def learn(self, learning_rule: str = 'oja') -> None:
        """
        Apply learning to the multimodal association layer and cross-connections.
        
        Args:
            learning_rule: Learning rule to use ('hebbian', 'oja', or 'stdp')
        """
        # Collect current inputs from all pathways
        pathway_activations = []
        for pathway in self.pathways:
            top_activations = pathway.layers[-1].activations
            pathway_activations.append(top_activations)
        
        # Concatenated input
        multimodal_input = xp.concatenate(pathway_activations)
        
        # Learn in the association layer
        self.association_layer.learn(multimodal_input, learning_rule)
        
        # Learn direct cross-modal connections between pathways
        self._learn_cross_modal_connections()
        
        # Update recurrent connections for temporal learning
        if self.reentrant_connections:
            self.association_layer.update_recurrent_connections()
    
    def _learn_cross_modal_connections(self) -> None:
        """
        Update cross-modal connection weights based on Hebbian coincidence.
        
        This implements cross-modal binding: neurons that fire together
        wire together across modalities.
        """
        # Process each pair of pathways
        for i in range(self.num_pathways):
            for j in range(self.num_pathways):
                if i == j:  # Skip self-connections
                    continue
                
                # Get pathway names and activations
                source_name = self.pathway_names[i]
                target_name = self.pathway_names[j]
                
                source_activations = self.last_activations[source_name]
                target_activations = self.last_activations[target_name]
                
                if source_activations is None or target_activations is None:
                    continue
                
                # Get connection matrix
                connection_key = f"{source_name}_to_{target_name}"
                connection_matrix = self.cross_modal_connections[connection_key]
                
                # Reshape for outer product
                source_act = source_activations.reshape(-1, 1)  # Column vector
                target_act = target_activations.reshape(1, -1)  # Row vector
                
                # Hebbian update (outer product)
                # When source and target are both active, strengthen connection
                update = self.learning_rate * xp.outer(source_act, target_act)
                
                # Apply Oja-like normalization to prevent weight explosion
                # We subtract a small fraction of the current weights scaled by activity
                # This is a form of weight decay proportional to the output
                normalization = self.learning_rate * 0.1 * (
                    connection_matrix * xp.mean(source_act) * xp.mean(target_act)
                )
                
                # Update connection matrix
                connection_matrix += update - normalization
                
                # Apply soft constraints to keep weights in reasonable range
                # Clip to prevent extreme values
                xp.clip(connection_matrix, -1.0, 1.0, out=connection_matrix)
                
                # Update statistics
                # Count non-zero connections
                connection_count = xp.sum(xp.abs(connection_matrix) > 0.05)
                avg_strength = xp.mean(xp.abs(connection_matrix))
                
                self.connection_statistics['connection_count'][connection_key] = connection_count
                self.connection_statistics['connection_strength'][connection_key] = avg_strength
    
    def check_association_exists(
        self,
        source_pathway: str,
        source_idx: int,
        target_pathway: str,
        target_idx: int
    ) -> bool:
        """
        Check if a direct association exists between neurons in different pathways.
        
        Args:
            source_pathway: Name of source pathway
            source_idx: Index of source neuron
            target_pathway: Name of target pathway
            target_idx: Index of target neuron
            
        Returns:
            Whether the association exists
        """
        # Validate inputs
        if source_pathway == target_pathway:
            return False  # No self-connections
            
        if source_pathway not in self.pathway_names or target_pathway not in self.pathway_names:
            return False  # Invalid pathway names
        
        # Get connection key
        connection_key = f"{source_pathway}_to_{target_pathway}"
        if connection_key not in self.cross_modal_connections:
            return False
        
        # Get connection matrix
        connection_matrix = self.cross_modal_connections[connection_key]
        
        # Check if connection weight is significant
        return xp.abs(connection_matrix[source_idx, target_idx]) > 0.05
    
    def create_association(
        self,
        source_pathway: str,
        source_idx: int,
        target_pathway: str,
        target_idx: int,
        initial_weight: float = 0.1
    ) -> bool:
        """
        Create a new association between neurons in different pathways.
        
        Args:
            source_pathway: Name of source pathway
            source_idx: Index of source neuron
            target_pathway: Name of target pathway
            target_idx: Index of target neuron
            initial_weight: Initial connection weight
            
        Returns:
            Whether the association was created
        """
        # Validate inputs
        if source_pathway == target_pathway:
            return False  # No self-connections
            
        if source_pathway not in self.pathway_names or target_pathway not in self.pathway_names:
            return False  # Invalid pathway names
        
        # Get pathway indices
        source_idx_pathway = self.pathway_names.index(source_pathway)
        target_idx_pathway = self.pathway_names.index(target_pathway)
        
        # Check index bounds
        source_size = self.pathway_output_sizes[source_idx_pathway]
        target_size = self.pathway_output_sizes[target_idx_pathway]
        
        if source_idx < 0 or source_idx >= source_size:
            return False  # Invalid source index
            
        if target_idx < 0 or target_idx >= target_size:
            return False  # Invalid target index
        
        # Get connection key
        forward_key = f"{source_pathway}_to_{target_pathway}"
        
        # Set the connection weight
        self.cross_modal_connections[forward_key][source_idx, target_idx] = initial_weight
        
        # If bidirectional, set reverse connection too
        if self.bidirectional_association:
            reverse_key = f"{target_pathway}_to_{source_pathway}"
            self.cross_modal_connections[reverse_key][target_idx, source_idx] = initial_weight
            
        self.logger.debug(
            f"Created association from {source_pathway}[{source_idx}] to "
            f"{target_pathway}[{target_idx}] with weight {initial_weight:.3f}"
        )
        
        return True
    
    def get_connection_strengths(self) -> Dict[str, xp.ndarray]:
        """
        Get the current strength of all connection matrices.
        
        Returns:
            Dictionary of connection matrices
        """
        return self.cross_modal_connections
    
    def get_multimodal_activity(self) -> Dict[str, Any]:
        """
        Get the current activity state of the multimodal layer.
        
        Returns:
            Dictionary with multimodal activity statistics
        """
        multimodal_act = self.association_layer.activations
        
        # Calculate statistics
        sparsity = xp.mean(multimodal_act > 0)
        avg_activation = xp.mean(multimodal_act)
        active_count = xp.sum(multimodal_act > 0)
        
        return {
            'activations': multimodal_act,
            'sparsity': float(sparsity),
            'avg_activation': float(avg_activation),
            'active_count': int(active_count)
        }
    
    def analyze_associations(self) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Analyze the strength of associations between modalities.
        
        Returns:
            Dictionary mapping pathway pairs to lists of strong associations
        """
        associations = {}
        
        # Process each pair of pathways
        for i in range(self.num_pathways):
            for j in range(self.num_pathways):
                if i == j:  # Skip self-connections
                    continue
                
                # Get pathway names
                source_name = self.pathway_names[i]
                target_name = self.pathway_names[j]
                
                # Get connection matrix
                key = f"{source_name}_to_{target_name}"
                matrix = self.cross_modal_connections[key]
                
                # Find strong associations
                threshold = 0.1  # Minimum weight to consider
                strong_connections = []
                
                for source_idx in range(matrix.shape[0]):
                    for target_idx in range(matrix.shape[1]):
                        weight = matrix[source_idx, target_idx]
                        if abs(weight) > threshold:
                            strong_connections.append((source_idx, target_idx, float(weight)))
                
                # Sort by strength
                strong_connections.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Store up to 20 strongest connections
                associations[key] = strong_connections[:20]
        
        return associations
    
    def prune_connections(self, threshold: float = 0.01) -> Dict[str, int]:
        """
        Prune weak connections across all cross-modal matrices.
        
        Args:
            threshold: Weight threshold for pruning
            
        Returns:
            Dictionary with pruning statistics per connection
        """
        pruning_stats = {}
        
        # Process each cross-modal connection
        for key, matrix in self.cross_modal_connections.items():
            # Find weak connections
            weak_mask = xp.abs(matrix) < threshold
            pruned_count = xp.sum(weak_mask)
            
            # Zero out weak connections
            if pruned_count > 0:
                matrix[weak_mask] = 0.0
                self.logger.debug(f"Pruned {pruned_count} weak connections in {key}")
            
            # Update pruning statistics
            pruning_stats[key] = int(pruned_count)
        
        # Prune multimodal layer connections
        multimodal_pruned = self.association_layer.prune_connections(threshold)
        pruning_stats['multimodal_layer'] = multimodal_pruned
        
        return pruning_stats
    
    def predict_cross_modal(
        self, 
        source_pathway: str, 
        source_activations: xp.ndarray
    ) -> Dict[str, xp.ndarray]:
        """
        Predict activations in other modalities given source activations.
        
        Args:
            source_pathway: Name of the source pathway
            source_activations: Activations in the source pathway
            
        Returns:
            Dictionary mapping target pathway names to predicted activations
        """
        if source_pathway not in self.pathway_names:
            return {}
            
        predictions = {}
        
        # Get source index
        source_idx = self.pathway_names.index(source_pathway)
        
        # Predict each target pathway
        for i, target_pathway in enumerate(self.pathway_names):
            if target_pathway == source_pathway:
                continue  # Skip self-prediction
                
            # Get connection key
            key = f"{source_pathway}_to_{target_pathway}"
            if key not in self.cross_modal_connections:
                continue
                
            # Get connection matrix
            matrix = self.cross_modal_connections[key]
            
            # Calculate prediction
            predicted_activations = xp.dot(source_activations, matrix)
            
            # Add to results
            predictions[target_pathway] = predicted_activations
        
        return predictions
    
    def __repr__(self) -> str:
        pathway_str = ", ".join(self.pathway_names)
        return f"LegacyMultimodalAssociation(pathways=[{pathway_str}], size={self.association_layer.layer_size})" 