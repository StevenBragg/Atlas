"""
Advanced visualization and analysis tools for neural representations.

This module provides tools to analyze and visualize the learned
representations, receptive fields, and cross-modal associations.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import os

from ..core import NeuralLayer, NeuralPathway
from ..models.multimodal.system import SelfOrganizingAVSystem


class RepresentationAnalyzer:
    """
    Analyze and visualize learned representations and neural properties.
    """
    
    def __init__(
        self,
        system: SelfOrganizingAVSystem,
        output_dir: str = "analysis",
    ):
        """
        Initialize representation analyzer.
        
        Args:
            system: The SelfOrganizingAVSystem to analyze
            output_dir: Directory to save visualizations
        """
        self.system = system
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger('RepresentationAnalyzer')
    
    def visualize_receptive_fields(self, pathway_name: str, layer_idx: int = 0) -> str:
        """
        Visualize receptive fields of neurons in a specific layer.
        
        Args:
            pathway_name: Name of the pathway ('visual' or 'audio')
            layer_idx: Index of the layer to visualize
            
        Returns:
            Path to saved visualization file
        """
        # Get the appropriate pathway
        pathway = None
        if pathway_name == 'visual':
            pathway = self.system.visual_processor.visual_pathway
        elif pathway_name == 'audio':
            pathway = self.system.audio_processor.audio_pathway
        else:
            self.logger.error(f"Unknown pathway name: {pathway_name}")
            return ""
        
        # Check layer index
        if layer_idx < 0 or layer_idx >= len(pathway.layers):
            self.logger.error(f"Invalid layer index {layer_idx} for pathway {pathway_name}")
            return ""
        
        # Get the layer
        layer = pathway.layers[layer_idx]
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=min(10, layer.layer_size // 5 + 1), 
            ncols=min(5, layer.layer_size),
            figsize=(15, 10)
        )
        axes = axes.flatten()
        
        # Plot receptive fields
        for i, neuron in enumerate(layer.neurons):
            if i >= len(axes):
                break
                
            weights = neuron.weights
            
            # Different visualization depending on pathway
            if pathway_name == 'visual':
                # For visual pathway, try to reshape for 2D visualization
                # This assumes weights connect to an image patch or similar
                if pathway.input_size == self.system.visual_processor.input_width * self.system.visual_processor.input_height:
                    # First layer might connect directly to image
                    width = self.system.visual_processor.input_width
                    height = self.system.visual_processor.input_height
                    weights_2d = weights.reshape(height, width)
                else:
                    # Otherwise, try to make a square-ish shape
                    size = int(np.sqrt(len(weights)))
                    weights_2d = weights[:size*size].reshape(size, size)
                
                # Plot as image
                im = axes[i].imshow(weights_2d, cmap='viridis')
                axes[i].set_title(f"Neuron {i}")
                axes[i].axis('off')
                
            elif pathway_name == 'audio':
                # For audio pathway, show as frequency response
                axes[i].plot(weights)
                axes[i].set_title(f"Neuron {i}")
                if i % 5 == 0:  # Add y-label for leftmost plots
                    axes[i].set_ylabel("Weight")
                if i >= len(axes) - 5:  # Add x-label for bottom plots
                    axes[i].set_xlabel("Input Index")
        
        # Hide unused axes
        for i in range(layer.layer_size, len(axes)):
            axes[i].axis('off')
        
        # Add colorbar for visual pathway
        if pathway_name == 'visual':
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{pathway_name}_layer{layer_idx}_receptive_fields.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved receptive field visualization to {output_path}")
        return output_path
    
    def visualize_cross_modal_associations(self) -> str:
        """
        Visualize cross-modal associations between pathways.
        
        Returns:
            Path to saved visualization file
        """
        # Get connection matrices from multimodal component
        connection_strengths = self.system.multimodal.get_connection_strengths()
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=len(connection_strengths), 
            ncols=1,
            figsize=(10, 8 * len(connection_strengths))
        )
        
        # Handle single connection case
        if len(connection_strengths) == 1:
            axes = [axes]
        
        # Plot each connection matrix
        for i, (key, matrix) in enumerate(connection_strengths.items()):
            im = axes[i].imshow(matrix, cmap='coolwarm', aspect='auto')
            axes[i].set_title(f"Cross-Modal Connections: {key}")
            axes[i].set_xlabel("Target Neuron Index")
            axes[i].set_ylabel("Source Neuron Index")
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"cross_modal_associations.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved cross-modal association visualization to {output_path}")
        return output_path
    
    def visualize_layer_activities(self, pathway_name: str) -> str:
        """
        Visualize activations of all layers in a pathway.
        
        Args:
            pathway_name: Name of the pathway ('visual' or 'audio')
            
        Returns:
            Path to saved visualization file
        """
        # Get the appropriate pathway
        pathway = None
        if pathway_name == 'visual':
            pathway = self.system.visual_processor.visual_pathway
        elif pathway_name == 'audio':
            pathway = self.system.audio_processor.audio_pathway
        else:
            self.logger.error(f"Unknown pathway name: {pathway_name}")
            return ""
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=pathway.num_layers, 
            ncols=1,
            figsize=(10, 3 * pathway.num_layers)
        )
        
        # Handle single layer case
        if pathway.num_layers == 1:
            axes = [axes]
        
        # Plot each layer's activations
        for i, layer in enumerate(pathway.layers):
            activations = layer.activations
            
            # Plot as bar chart
            axes[i].bar(range(len(activations)), activations)
            axes[i].set_title(f"Layer {i} Activations")
            axes[i].set_xlabel("Neuron Index")
            axes[i].set_ylabel("Activation")
            
            # Add sparsity information
            sparsity = np.mean(activations > 0)
            axes[i].text(
                0.02, 0.95, 
                f"Sparsity: {sparsity:.3f}", 
                transform=axes[i].transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{pathway_name}_layer_activities.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved layer activations visualization to {output_path}")
        return output_path
    
    def analyze_pathway_diversity(self, pathway_name: str) -> Dict[str, Any]:
        """
        Calculate diversity metrics for neurons in a pathway.
        
        Args:
            pathway_name: Name of the pathway ('visual' or 'audio')
            
        Returns:
            Dictionary with diversity metrics
        """
        # Get the appropriate pathway
        pathway = None
        if pathway_name == 'visual':
            pathway = self.system.visual_processor.visual_pathway
        elif pathway_name == 'audio':
            pathway = self.system.audio_processor.audio_pathway
        else:
            self.logger.error(f"Unknown pathway name: {pathway_name}")
            return {}
        
        # Calculate diversity metrics for each layer
        metrics = {}
        
        for i, layer in enumerate(pathway.layers):
            layer_metrics = {}
            
            # Get all weight vectors (receptive fields)
            weight_vectors = np.array([neuron.weights for neuron in layer.neurons])
            
            # Calculate weight correlation matrix
            correlations = np.corrcoef(weight_vectors)
            np.fill_diagonal(correlations, 0)  # Remove self-correlations
            
            # Calculate metrics
            mean_correlation = np.mean(np.abs(correlations))
            max_correlation = np.max(np.abs(correlations))
            
            # Calculate weight variance (average distance from mean weight)
            mean_weights = np.mean(weight_vectors, axis=0)
            weight_variance = np.mean(np.mean((weight_vectors - mean_weights)**2, axis=1))
            
            # Calculate activation sparsity from history
            if hasattr(layer, 'mean_activation_history') and len(layer.mean_activation_history) > 0:
                activation_sparsity = np.mean(layer.sparsity_history[-min(len(layer.sparsity_history), 100):])
            else:
                activation_sparsity = np.mean(layer.activations > 0)
            
            # Store metrics
            layer_metrics['mean_weight_correlation'] = float(mean_correlation)
            layer_metrics['max_weight_correlation'] = float(max_correlation)
            layer_metrics['weight_variance'] = float(weight_variance)
            layer_metrics['activation_sparsity'] = float(activation_sparsity)
            
            # Calculate effective dimensionality if there are enough neurons
            if layer.layer_size > 5:
                # Use eigenvalues of the covariance matrix as an estimate
                cov = np.cov(weight_vectors)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
                
                # Effective dimensionality formula
                sum_eig = np.sum(eigenvalues)
                if sum_eig > 0:
                    eff_dim = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
                    layer_metrics['effective_dimensionality'] = float(eff_dim)
                else:
                    layer_metrics['effective_dimensionality'] = 0.0
            
            metrics[f"layer_{i}"] = layer_metrics
        
        return metrics
    
    def visualize_prediction_performance(self) -> str:
        """
        Visualize temporal prediction performance.
        
        Returns:
            Path to saved visualization file
        """
        # Get prediction error history from system metrics
        error_history = self.system.metrics.get('prediction_error', [])
        
        if not error_history:
            self.logger.warning("No prediction error history available")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot error over time
        frames = [x[0] for x in error_history]
        errors = [x[1] for x in error_history]
        
        ax.plot(frames, errors)
        ax.set_title("Prediction Error Over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean Squared Error")
        
        # Add smoothed trend line if there are enough points
        if len(frames) > 10:
            window_size = min(len(frames) // 5, 50)
            smoothed = np.convolve(errors, np.ones(window_size)/window_size, mode='valid')
            smoothed_frames = frames[window_size-1:]
            ax.plot(smoothed_frames, smoothed, 'r-', linewidth=2, label='Trend')
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            "prediction_performance.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved prediction performance visualization to {output_path}")
        return output_path
    
    def visualize_network_growth(self) -> str:
        """
        Visualize network growth and structural changes over time.
        
        Returns:
            Path to saved visualization file
        """
        # Get structural changes from system metrics
        structural_changes = self.system.metrics.get('structural_changes', [])
        pruning_stats = self.system.metrics.get('pruning_stats', [])
        
        if not structural_changes and not pruning_stats:
            self.logger.warning("No structural change history available")
            return ""
        
        # Create figure
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
        
        # Plot neuron additions over time
        if structural_changes:
            frames = [x[0] for x in structural_changes]
            
            # Extract additions per pathway
            pathways = set()
            for _, changes in structural_changes:
                pathways.update(changes.keys())
            
            for pathway in pathways:
                additions = [
                    sum(change[1].get(pathway, {}).values())
                    for _, change in structural_changes
                ]
                axes[0].plot(frames, additions, label=f'{pathway} neurons')
            
            axes[0].set_title("Neuron Additions Over Time")
            axes[0].set_ylabel("Neurons Added")
            axes[0].legend()
        
        # Plot pruning stats over time
        if pruning_stats:
            frames = [x[0] for x in pruning_stats]
            
            # Extract pruning counts per pathway
            pathways = set()
            for _, stats in pruning_stats:
                pathways.update(stats.keys())
            
            for pathway in pathways:
                prunes = []
                for _, stats in pruning_stats:
                    if pathway in stats:
                        prunes.append(sum(stats[pathway].values()))
                    else:
                        prunes.append(0)
                
                axes[1].plot(frames, prunes, label=f'{pathway} pruned')
            
            axes[1].set_title("Connection Pruning Over Time")
            axes[1].set_xlabel("Frame")
            axes[1].set_ylabel("Connections Pruned")
            axes[1].legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            "network_growth.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved network growth visualization to {output_path}")
        return output_path
    
    def generate_representation_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the system's representations.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'frame_count': self.system.frame_count,
            'time_step': self.system.time_step,
            'visual': self.analyze_pathway_diversity('visual'),
            'audio': self.analyze_pathway_diversity('audio'),
            'cross_modal': {},
        }
        
        # Add multimodal statistics
        multimodal_activity = self.system.multimodal.get_multimodal_activity()
        associations = self.system.multimodal.analyze_associations()
        
        summary['multimodal'] = {
            'activity': {
                'sparsity': multimodal_activity['sparsity'],
                'avg_activation': multimodal_activity['avg_activation'],
                'active_count': multimodal_activity['active_count']
            },
            'associations': {
                key: len(assocs) for key, assocs in associations.items()
            }
        }
        
        # Generate visualizations
        visualizations = {
            'visual_receptive_fields': self.visualize_receptive_fields('visual', 0),
            'audio_receptive_fields': self.visualize_receptive_fields('audio', 0),
            'cross_modal_associations': self.visualize_cross_modal_associations(),
            'visual_activities': self.visualize_layer_activities('visual'),
            'audio_activities': self.visualize_layer_activities('audio'),
            'prediction_performance': self.visualize_prediction_performance(),
            'network_growth': self.visualize_network_growth()
        }
        
        summary['visualizations'] = visualizations
        
        # Save summary to file
        summary_path = os.path.join(
            self.output_dir, 
            f"representation_summary_{self.system.frame_count}.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(f"Representation Summary at Frame {self.system.frame_count}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Visual Pathway:\n")
            for layer, metrics in summary['visual'].items():
                f.write(f"  {layer}:\n")
                for k, v in metrics.items():
                    f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
            
            f.write(f"Audio Pathway:\n")
            for layer, metrics in summary['audio'].items():
                f.write(f"  {layer}:\n")
                for k, v in metrics.items():
                    f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
            
            f.write(f"Multimodal Layer:\n")
            for k, v in summary['multimodal']['activity'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            f.write(f"Cross-Modal Associations:\n")
            for k, v in summary['multimodal']['associations'].items():
                f.write(f"  {k}: {v} strong associations\n")
            f.write("\n")
            
            f.write(f"Visualizations saved to:\n")
            for k, v in visualizations.items():
                if v:
                    f.write(f"  {k}: {v}\n")
        
        self.logger.info(f"Saved representation summary to {summary_path}")
        
        return summary
    
    def visualize_receptive_field_distribution(self, pathway_name: str, layer_idx: int = 0) -> str:
        """
        Visualize the distribution of receptive field properties.
        
        Args:
            pathway_name: Name of the pathway ('visual' or 'audio')
            layer_idx: Index of the layer to visualize
            
        Returns:
            Path to saved visualization file
        """
        # Get the appropriate pathway
        pathway = None
        if pathway_name == 'visual':
            pathway = self.system.visual_processor.visual_pathway
        elif pathway_name == 'audio':
            pathway = self.system.audio_processor.audio_pathway
        else:
            self.logger.error(f"Unknown pathway name: {pathway_name}")
            return ""
        
        # Check layer index
        if layer_idx < 0 or layer_idx >= len(pathway.layers):
            self.logger.error(f"Invalid layer index {layer_idx} for pathway {pathway_name}")
            return ""
        
        # Get the layer
        layer = pathway.layers[layer_idx]
        
        # Extract weight statistics
        receptive_fields = np.array([neuron.weights for neuron in layer.neurons])
        
        # Calculate some properties
        mean_magnitudes = np.mean(np.abs(receptive_fields), axis=1)
        max_values = np.max(receptive_fields, axis=1)
        min_values = np.min(receptive_fields, axis=1)
        ranges = max_values - min_values
        
        # Create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        
        # Plot distributions
        axes[0, 0].hist(mean_magnitudes, bins=20)
        axes[0, 0].set_title('Mean Absolute Weight')
        axes[0, 0].set_xlabel('Magnitude')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].hist(max_values, bins=20)
        axes[0, 1].set_title('Maximum Weight')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Count')
        
        axes[1, 0].hist(min_values, bins=20)
        axes[1, 0].set_title('Minimum Weight')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].hist(ranges, bins=20)
        axes[1, 1].set_title('Weight Range')
        axes[1, 1].set_xlabel('Range')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{pathway_name}_layer{layer_idx}_receptive_field_distribution.png"
        )
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved receptive field distribution to {output_path}")
        return output_path 