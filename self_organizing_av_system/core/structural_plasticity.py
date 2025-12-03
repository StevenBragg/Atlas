"""
Structural plasticity mechanisms for the self-organizing AV system.

This implements dynamic neuron addition, synapse sprouting, and pruning
as described in the architecture document.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable
import time
from collections import defaultdict
from enum import Enum

from .backend import xp, to_cpu
from .neuron import Neuron
from .layer import NeuralLayer
from .pathway import NeuralPathway

logger = logging.getLogger(__name__)


class GrowthStrategy(Enum):
    """Strategies for growing new neurons and connections"""
    ACTIVITY_BASED = "activity_based"  # Add neurons when high activity variance is detected
    NOVELTY_BASED = "novelty_based"    # Add neurons when novel patterns are detected
    ERROR_BASED = "error_based"        # Add neurons when reconstruction error is high
    HYBRID = "hybrid"                  # Combination of multiple strategies


class PruningStrategy(Enum):
    """Strategies for pruning neurons and connections"""
    WEIGHT_BASED = "weight_based"      # Prune connections with weak weights
    ACTIVITY_BASED = "activity_based"  # Prune neurons with consistently low activity
    CORRELATION_BASED = "correlation_based"  # Prune redundant neurons with high correlation
    UTILITY_BASED = "utility_based"    # Prune based on utility/contribution metrics


class PlasticityMode(Enum):
    """Modes for structural plasticity"""
    STABLE = "stable"               # No structural changes
    GROWING = "growing"             # Add neurons/connections to adapt to novel patterns
    PRUNING = "pruning"             # Remove weak/unused connections
    ADAPTIVE = "adaptive"           # Automatic mode switching based on error signals
    SPROUTING = "sprouting"         # Create new lateral connections between co-active neurons


class StructuralPlasticity:
    """
    Manages structural plasticity for neural pathways through dynamic addition and 
    pruning of neurons and connections.
    
    This class implements mechanisms for:
    
    1. Dynamic growth of neurons in response to novel patterns
    2. Sprouting of connections between modalities
    3. Pruning of weak or redundant connections
    4. Monitoring and analysis of structural changes
    
    The module works with neural pathways to analyze activity patterns and
    adaptation needs, modifying the network structure to improve representation.
    """
    
    def __init__(
        self,
        initial_size: int,
        max_size: int = None,
        min_size: int = 10,
        growth_threshold: float = 0.8,
        growth_rate: float = 0.1,
        prune_threshold: float = 0.01,
        utility_window: int = 100,
        growth_strategy: Union[str, GrowthStrategy] = GrowthStrategy.HYBRID,
        pruning_strategy: Union[str, PruningStrategy] = PruningStrategy.WEIGHT_BASED,
        growth_cooldown: int = 50,
        check_interval: int = 100,
        novelty_threshold: float = 0.3,
        redundancy_threshold: float = 0.85,
        max_growth_per_step: int = 5,
        max_prune_per_step: int = 3,
        enable_consolidation: bool = True,
        min_age_for_pruning: int = 100,
        structural_plasticity_mode: Union[str, PlasticityMode] = PlasticityMode.ADAPTIVE,
        max_fan_in: int = None,
        max_fan_out: int = None,
        weight_init_mean: float = 0.0,
        weight_init_std: float = 0.1,
        growth_increment: int = 1,
        update_interval: int = 10,
        consolidation_threshold: float = 0.5,
        enable_neuron_growth: bool = True,
        enable_connection_pruning: bool = True,
        enable_connection_sprouting: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize structural plasticity.
        
        Args:
            initial_size: Initial size of the neural representation
            max_size: Maximum allowed representation size
            min_size: Minimum allowed representation size
            growth_threshold: Threshold for activity to trigger growth
            growth_rate: Rate at which to grow relative to current size
            prune_threshold: Threshold for pruning weak connections
            utility_window: Number of updates to consider for utility calculations
            growth_strategy: Strategy for growing new neurons
            pruning_strategy: Strategy for pruning neurons and connections
            growth_cooldown: Minimum updates between growth events
            check_interval: Interval for checking structural changes
            novelty_threshold: Threshold for detecting novel patterns
            redundancy_threshold: Correlation threshold for detecting redundancy
            max_growth_per_step: Maximum neurons to add per update
            max_prune_per_step: Maximum neurons to prune per update
            enable_consolidation: Whether to enable consolidation of representations
            min_age_for_pruning: Minimum age (updates) before a connection can be pruned
            structural_plasticity_mode: Mode of structural plasticity
            max_fan_in: Maximum incoming connections per neuron (None for unlimited)
            max_fan_out: Maximum outgoing connections per neuron (None for unlimited)
            weight_init_mean: Mean for initializing new weights
            weight_init_std: Standard deviation for initializing new weights
            growth_increment: Number of neurons to add at each growth step
            update_interval: Interval (in update steps) between structural changes
            consolidation_threshold: Threshold for connection consolidation
            enable_neuron_growth: Whether to enable neuron growth
            enable_connection_pruning: Whether to enable connection pruning
            enable_connection_sprouting: Whether to enable connection sprouting
            random_seed: Random seed for reproducibility
        """
        self.current_size = initial_size
        self.max_size = max_size if max_size is not None else initial_size * 5
        self.min_size = min_size
        self.growth_threshold = growth_threshold
        self.growth_rate = growth_rate
        self.prune_threshold = prune_threshold
        self.utility_window = utility_window
        
        # Convert string to enum if needed
        if isinstance(growth_strategy, str):
            growth_strategy = GrowthStrategy(growth_strategy)
        if isinstance(pruning_strategy, str):
            pruning_strategy = PruningStrategy(pruning_strategy)
            
        self.growth_strategy = growth_strategy
        self.pruning_strategy = pruning_strategy
        self.growth_cooldown = growth_cooldown
        self.check_interval = check_interval
        self.novelty_threshold = novelty_threshold
        self.redundancy_threshold = redundancy_threshold
        self.max_growth_per_step = max_growth_per_step
        self.max_prune_per_step = max_prune_per_step
        self.enable_consolidation = enable_consolidation
        
        # Internal state tracking
        self.update_count = 0
        self.last_growth_update = -growth_cooldown  # Allow growth immediately
        self.last_prune_update = 0
        
        # Activity history for utility calculations
        self.activity_history = []
        self.utility_scores = xp.ones(initial_size)  # Initialize with uniform utility
        self.activation_frequency = xp.zeros(initial_size)
        self.reconstruction_errors = []
        
        # Growth and pruning history
        self.growth_events = []
        self.prune_events = []
        
        # Novelty detection
        self.prototype_patterns = []
        self.pattern_counts = []
        
        # New parameters
        self.min_age_for_pruning = min_age_for_pruning
        self.structural_plasticity_mode = structural_plasticity_mode
        self.max_fan_in = max_fan_in
        self.max_fan_out = max_fan_out
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.growth_increment = growth_increment
        self.update_interval = update_interval
        self.consolidation_threshold = consolidation_threshold
        self.enable_neuron_growth = enable_neuron_growth
        self.enable_connection_pruning = enable_connection_pruning
        self.enable_connection_sprouting = enable_connection_sprouting
        
        # Set random seed if provided
        if random_seed is not None:
            xp.random.seed(random_seed)
        
        logger.info(f"Initialized structural plasticity: size={initial_size}, max={max_size}, "
                   f"growth_strategy={growth_strategy.value}, pruning_strategy={pruning_strategy.value}")
    
    def update(
        self,
        activity: xp.ndarray,
        weights: Optional[xp.ndarray] = None,
        reconstruction_error: Optional[float] = None,
        force_check: bool = False
    ) -> Dict[str, Any]:
        """
        Update structural plasticity based on current activity.
        
        Args:
            activity: Current neural activity pattern
            weights: Current weight matrix (if available)
            reconstruction_error: Current reconstruction error (if available)
            force_check: Force structural check regardless of interval
            
        Returns:
            Dictionary with structural change decisions
        """
        # Verify input dimensions
        assert len(activity) == self.current_size, f"Activity size mismatch: {len(activity)} != {self.current_size}"
        
        # Update activity history
        self._update_activity_history(activity)
        
        # Update reconstruction error history
        if reconstruction_error is not None:
            self.reconstruction_errors.append(reconstruction_error)
            if len(self.reconstruction_errors) > self.utility_window:
                self.reconstruction_errors = self.reconstruction_errors[-self.utility_window:]
        
        # Calculate utility scores based on activity history
        self._update_utility_scores(activity)
        
        # Check for novelty in activity pattern
        is_novel = self._check_novelty(activity)
        
        # Determine if structural check is needed
        check_needed = (
            (self.update_count % self.check_interval == 0) or 
            force_check or 
            (is_novel and (self.update_count - self.last_growth_update >= self.growth_cooldown))
        )
        
        result = {
            'size_changed': False,
            'grown': 0,
            'pruned': 0,
            'is_novel': is_novel,
            'current_size': self.current_size,
            'consolidated': False
        }
        
        # Perform structural check if needed
        if check_needed:
            # Evaluate growth needs
            growth_neurons = self._evaluate_growth_needs(activity, weights, reconstruction_error)
            
            # Evaluate pruning needs
            prune_indices = self._evaluate_pruning_needs(weights)
            
            # Apply changes
            if growth_neurons > 0:
                result['grown'] = growth_neurons
                result['size_changed'] = True
                self.current_size += growth_neurons
                self.utility_scores = xp.append(self.utility_scores, xp.ones(growth_neurons))
                self.activation_frequency = xp.append(self.activation_frequency, xp.zeros(growth_neurons))
                
                # Record growth event
                self.growth_events.append({
                    'update': self.update_count,
                    'grown': growth_neurons,
                    'new_size': self.current_size,
                    'strategy': self.growth_strategy.value,
                    'reason': 'novelty' if is_novel else 'utility'
                })
                
                self.last_growth_update = self.update_count
                logger.info(f"Grown {growth_neurons} neurons, new size: {self.current_size}")
            
            if prune_indices:
                result['pruned'] = len(prune_indices)
                result['prune_indices'] = prune_indices
                result['size_changed'] = True
                
                # Record prune event
                self.prune_events.append({
                    'update': self.update_count,
                    'pruned': len(prune_indices),
                    'new_size': self.current_size - len(prune_indices),
                    'strategy': self.pruning_strategy.value
                })
                
                self.current_size -= len(prune_indices)
                self.last_prune_update = self.update_count
                logger.info(f"Pruned {len(prune_indices)} neurons, new size: {self.current_size}")
            
            # Consolidate representations if enabled and no structural changes
            if self.enable_consolidation and not result['size_changed'] and self.update_count % (self.check_interval * 5) == 0:
                result['consolidated'] = self._consolidate_representations(weights)
        
        self.update_count += 1
        return result
    
    def _update_activity_history(self, activity: xp.ndarray) -> None:
        """
        Update the activity history with current activity.
        
        Args:
            activity: Current neural activity pattern
        """
        self.activity_history.append(activity.copy())
        if len(self.activity_history) > self.utility_window:
            self.activity_history = self.activity_history[-self.utility_window:]
        
        # Update activation frequency
        self.activation_frequency = xp.mean([a > 0.1 for a in self.activity_history], axis=0)
    
    def _update_utility_scores(self, activity: xp.ndarray) -> None:
        """
        Update utility scores for all neurons.
        
        Args:
            activity: Current neural activity pattern
        """
        # Skip if not enough history
        if len(self.activity_history) < 5:
            return
        
        # Convert history to array for calculations
        history_array = xp.array(self.activity_history)
        
        # Get mean activity per neuron
        mean_activity = xp.mean(history_array, axis=0)
        
        # Get variance of activity per neuron
        var_activity = xp.var(history_array, axis=0)
        
        # Calculate activation sparsity (% of time neuron is active)
        sparsity = xp.mean(history_array > 0.1, axis=0)
        
        # Calculate utility as combination of variance and activation frequency
        # Neurons with high variance and moderate activation are most useful
        self.utility_scores = (var_activity + 0.1) * (4 * sparsity * (1 - sparsity) + 0.1)
        
        # Normalize utility scores
        self.utility_scores = self.utility_scores / (xp.mean(self.utility_scores) + 1e-10)
    
    def _check_novelty(self, activity: xp.ndarray) -> bool:
        """
        Check if current activity pattern represents a novel input.
        
        Args:
            activity: Current neural activity pattern
            
        Returns:
            Boolean indicating novelty
        """
        # Skip if activity is too sparse
        if xp.sum(activity > 0.1) < 2:
            return False
        
        # If no prototypes exist, add the first and return True
        if not self.prototype_patterns:
            if xp.max(activity) > 0.5:  # Only add as prototype if activity is significant
                self.prototype_patterns.append(activity.copy())
                self.pattern_counts.append(1)
            return True
        
        # Calculate similarity to existing prototypes
        max_similarity = 0
        max_idx = 0
        
        # Normalize activity for similarity comparison
        norm_activity = activity / (xp.linalg.norm(activity) + 1e-10)
        
        # Find most similar prototype
        for i, prototype in enumerate(self.prototype_patterns):
            norm_prototype = prototype / (xp.linalg.norm(prototype) + 1e-10)
            similarity = xp.dot(norm_activity, norm_prototype)
            if similarity > max_similarity:
                max_similarity = similarity
                max_idx = i
        
        # If similarity is below threshold, this is a novel pattern
        if max_similarity < self.novelty_threshold and xp.max(activity) > 0.5:
            # Add as new prototype if we haven't reached the limit
            if len(self.prototype_patterns) < 50:  # Limit number of prototypes
                self.prototype_patterns.append(activity.copy())
                self.pattern_counts.append(1)
            return True
        else:
            # Increment pattern count
            if len(self.pattern_counts) > max_idx:
                self.pattern_counts[max_idx] += 1
            return False
    
    def _evaluate_growth_needs(
        self, 
        activity: xp.ndarray,
        weights: Optional[xp.ndarray] = None,
        reconstruction_error: Optional[float] = None
    ) -> int:
        """
        Evaluate need for neural growth based on current activity.
        
        Args:
            activity: Current neural activity pattern
            weights: Current weight matrix (if available)
            reconstruction_error: Current reconstruction error (if available)
            
        Returns:
            Number of neurons to add
        """
        # Enforce maximum size limit
        if self.current_size >= self.max_size:
            return 0
        
        # Enforce growth cooldown
        if (self.update_count - self.last_growth_update) < self.growth_cooldown:
            return 0
        
        # Calculate growth based on selected strategy
        if self.growth_strategy == GrowthStrategy.ACTIVITY_BASED:
            # Grow based on high activity levels and concentration
            if len(self.activity_history) < 10:
                return 0
                
            # Check if activity is concentrated in few neurons
            history_array = xp.array(self.activity_history[-10:])
            mean_activity = xp.mean(history_array, axis=0)
            
            # Calculate concentration of activity (Gini-like coefficient)
            sorted_activity = xp.sort(mean_activity)
            cumsum = xp.cumsum(sorted_activity)
            concentration = xp.sum(cumsum) / (self.current_size * xp.sum(mean_activity))
            
            # Grow if activity is high and concentrated
            if concentration > self.growth_threshold and xp.mean(mean_activity) > 0.3:
                growth_size = int(xp.ceil(self.current_size * self.growth_rate))
                return min(growth_size, self.max_growth_per_step)
                
        elif self.growth_strategy == GrowthStrategy.NOVELTY_BASED:
            # Check if we're seeing consistently novel patterns
            if len(self.prototype_patterns) > self.current_size * 0.5:
                growth_size = int(xp.ceil(self.current_size * self.growth_rate))
                return min(growth_size, self.max_growth_per_step)
                
        elif self.growth_strategy == GrowthStrategy.ERROR_BASED:
            # Grow based on high reconstruction error
            if not self.reconstruction_errors or reconstruction_error is None:
                return 0
                
            # Compare current error to recent average
            recent_error_avg = xp.mean(self.reconstruction_errors[-10:])
            
            # Grow if error is consistently high
            if reconstruction_error > recent_error_avg * 1.2 and recent_error_avg > 0.2:
                growth_size = int(xp.ceil(self.current_size * self.growth_rate))
                return min(growth_size, self.max_growth_per_step)
                
        elif self.growth_strategy == GrowthStrategy.HYBRID:
            # Combine multiple growth criteria
            growth_signals = 0
            
            # Activity-based signal
            if len(self.activity_history) >= 10:
                history_array = xp.array(self.activity_history[-10:])
                mean_activity = xp.mean(history_array, axis=0)
                sorted_activity = xp.sort(mean_activity)
                cumsum = xp.cumsum(sorted_activity)
                concentration = xp.sum(cumsum) / (self.current_size * xp.sum(mean_activity) + 1e-10)
                
                if concentration > self.growth_threshold and xp.mean(mean_activity) > 0.3:
                    growth_signals += 1
            
            # Novelty-based signal
            if len(self.prototype_patterns) > self.current_size * 0.3:
                growth_signals += 1
            
            # Error-based signal
            if (self.reconstruction_errors and reconstruction_error is not None and 
                reconstruction_error > xp.mean(self.reconstruction_errors) * 1.2):
                growth_signals += 1
            
            # Grow if multiple signals agree
            if growth_signals >= 2:
                growth_size = int(xp.ceil(self.current_size * self.growth_rate))
                return min(growth_size, self.max_growth_per_step)
        
        return 0
    
    def _evaluate_pruning_needs(self, weights: Optional[xp.ndarray] = None) -> List[int]:
        """
        Evaluate need for pruning based on utility scores.
        
        Args:
            weights: Current weight matrix (if available)
            
        Returns:
            List of indices to prune
        """
        # Enforce minimum size limit
        if self.current_size <= self.min_size:
            return []
        
        # Skip if not enough history
        if len(self.activity_history) < self.utility_window // 2:
            return []
        
        prune_indices = []
        
        if self.pruning_strategy == PruningStrategy.WEIGHT_BASED:
            # Prune based on weight magnitudes (need weight matrix)
            if weights is None:
                return []
                
            # Get row weights (for outgoing connections)
            if len(weights.shape) == 2:
                row_norms = xp.sum(xp.abs(weights), axis=1)
                
                # Identify candidates with low weights
                candidates = xp.where(row_norms < self.prune_threshold * xp.mean(row_norms))[0]
                
                # Select up to max_prune_per_step of the weakest neurons
                if len(candidates) > 0:
                    # Sort by increasing strength
                    sorted_candidates = candidates[xp.argsort(row_norms[candidates])]
                    # Take the weakest ones
                    prune_indices = sorted_candidates[:min(len(sorted_candidates), self.max_prune_per_step)]
        
        elif self.pruning_strategy == PruningStrategy.ACTIVITY_BASED:
            # Prune neurons with consistently low activity
            
            # Calculate average activation
            history_array = xp.array(self.activity_history)
            mean_activity = xp.mean(history_array, axis=0)
            
            # Identify candidates with very low activity
            candidates = xp.where(mean_activity < self.prune_threshold * xp.mean(mean_activity))[0]
            
            # Ensure we don't prune too many at once
            if len(candidates) > 0:
                sorted_candidates = candidates[xp.argsort(mean_activity[candidates])]
                prune_indices = sorted_candidates[:min(len(sorted_candidates), self.max_prune_per_step)]
        
        elif self.pruning_strategy == PruningStrategy.CORRELATION_BASED:
            # Prune redundant neurons with high correlation
            if len(self.activity_history) < 10:
                return []
                
            # Calculate correlation between neuron activities
            history_array = xp.array(self.activity_history[-30:])
            
            # Skip if variance is too low
            if xp.mean(xp.var(history_array, axis=0)) < 0.01:
                return []
            
            # Calculate correlation matrix between neurons
            if self.current_size > 1:  # Need at least 2 neurons for correlation
                # This is computationally expensive, so do it less frequently
                if self.update_count % (self.check_interval * 2) != 0:
                    return []
                    
                # Calculate correlation
                correlations = xp.corrcoef(history_array.T)
                xp.fill_diagonal(correlations, 0)  # Exclude self-correlations
                
                # Find pairs with correlation above threshold
                high_corr_pairs = xp.argwhere(correlations > self.redundancy_threshold)
                
                # Identify redundant neurons (appear in multiple pairs)
                if len(high_corr_pairs) > 0:
                    # Count occurrences of each neuron in high correlation pairs
                    neuron_counts = xp.bincount(high_corr_pairs.flatten(), minlength=self.current_size)
                    
                    # Find neurons involved in multiple high correlations
                    redundant_candidates = xp.where(neuron_counts > 1)[0]
                    
                    # For each candidate, choose the one with lower utility
                    for idx in redundant_candidates:
                        # Find its partners
                        partners = xp.where(correlations[idx] > self.redundancy_threshold)[0]
                        
                        # Compare utility scores
                        if xp.any(self.utility_scores[partners] > self.utility_scores[idx]):
                            if idx not in prune_indices and len(prune_indices) < self.max_prune_per_step:
                                prune_indices.append(idx)
        
        elif self.pruning_strategy == PruningStrategy.UTILITY_BASED:
            # Prune based on comprehensive utility metric
            
            # Identify candidates with low utility
            sorted_utility = xp.argsort(self.utility_scores)
            
            # Consider lowest 10% as candidates
            n_candidates = max(1, int(0.1 * self.current_size))
            candidates = sorted_utility[:n_candidates]
            
            # Prune only if utility is very low compared to average
            prune_threshold = xp.mean(self.utility_scores) * self.prune_threshold
            final_candidates = [idx for idx in candidates if self.utility_scores[idx] < prune_threshold]
            
            # Limit number pruned per step
            prune_indices = final_candidates[:min(len(final_candidates), self.max_prune_per_step)]
        
        # Convert to list of integers
        return sorted([int(i) for i in prune_indices])
    
    def _consolidate_representations(self, weights: Optional[xp.ndarray] = None) -> bool:
        """
        Consolidate neural representations by reinforcing useful pattern separations.
        
        Args:
            weights: Current weight matrix (if available)
            
        Returns:
            Whether consolidation was performed
        """
        # Skip if not enough data or weights not provided
        if weights is None or len(self.activity_history) < 10:
            return False
            
        # This would involve reinforcing useful weights and neurons
        # For example, increasing the contrast between high and low utility neurons
        
        # Simple implementation: track that we performed consolidation
        return True
    
    def resize(self, new_size: int) -> Dict[str, Any]:
        """
        Explicitly resize the representation.
        
        Args:
            new_size: New size for the representation
            
        Returns:
            Dictionary with resize information
        """
        # Validate new size
        if new_size < self.min_size:
            new_size = self.min_size
            logger.warning(f"Requested size {new_size} below minimum, using {self.min_size}")
        
        if new_size > self.max_size:
            new_size = self.max_size
            logger.warning(f"Requested size {new_size} above maximum, using {self.max_size}")
        
        # Calculate growth or shrink
        size_diff = new_size - self.current_size
        
        result = {
            'previous_size': self.current_size,
            'new_size': new_size,
            'change': size_diff
        }
        
        # Update internal state
        if size_diff != 0:
            self.current_size = new_size
            
            # Resize utility scores and activation frequency
            if size_diff > 0:
                # Growing
                self.utility_scores = xp.append(self.utility_scores, xp.ones(size_diff))
                self.activation_frequency = xp.append(self.activation_frequency, xp.zeros(size_diff))
                
                # Record growth event
                self.growth_events.append({
                    'update': self.update_count,
                    'grown': size_diff,
                    'new_size': self.current_size,
                    'strategy': 'manual',
                    'reason': 'explicit_resize'
                })
                
                logger.info(f"Explicitly grown by {size_diff} neurons, new size: {self.current_size}")
                
            else:
                # Shrinking - keep neurons with highest utility
                sorted_utility = xp.argsort(self.utility_scores)[::-1]  # Sort by descending utility
                keep_indices = sorted_utility[:new_size]
                
                # Resize arrays
                self.utility_scores = self.utility_scores[keep_indices]
                self.activation_frequency = self.activation_frequency[keep_indices]
                
                # Record prune event
                self.prune_events.append({
                    'update': self.update_count,
                    'pruned': -size_diff,
                    'new_size': self.current_size,
                    'strategy': 'manual',
                    'reason': 'explicit_resize'
                })
                
                # Return indices that were pruned
                pruned_indices = sorted_utility[new_size:]
                result['pruned_indices'] = [int(i) for i in pruned_indices]
                
                logger.info(f"Explicitly shrunk by {-size_diff} neurons, new size: {self.current_size}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current structural state.
        
        Returns:
            Dictionary with structural statistics
        """
        # Calculate stats on utility distribution
        utility_mean = xp.mean(self.utility_scores) if len(self.utility_scores) > 0 else 0
        utility_std = xp.std(self.utility_scores) if len(self.utility_scores) > 0 else 0
        utility_min = xp.min(self.utility_scores) if len(self.utility_scores) > 0 else 0
        utility_max = xp.max(self.utility_scores) if len(self.utility_scores) > 0 else 0
        
        # Summary of growth and pruning history
        total_grown = sum(event['grown'] for event in self.growth_events)
        total_pruned = sum(event['pruned'] for event in self.prune_events)
        
        # Recent growth and pruning
        recent_grown = sum(event['grown'] for event in self.growth_events[-5:]) if self.growth_events else 0
        recent_pruned = sum(event['pruned'] for event in self.prune_events[-5:]) if self.prune_events else 0
        
        # Calculate stability metric: 1 - (recent_changes / current_size)
        stability = 1.0 - (recent_grown + recent_pruned) / (self.current_size + 1e-10)
        
        return {
            'current_size': self.current_size,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'update_count': self.update_count,
            'total_grown': int(total_grown),
            'total_pruned': int(total_pruned),
            'net_growth': int(total_grown - total_pruned),
            'utility_mean': float(utility_mean),
            'utility_std': float(utility_std),
            'utility_min': float(utility_min),
            'utility_max': float(utility_max),
            'stability': float(stability),
            'prototype_count': len(self.prototype_patterns),
            'growth_strategy': self.growth_strategy.value,
            'pruning_strategy': self.pruning_strategy.value,
            'recent_grown': int(recent_grown),
            'recent_pruned': int(recent_pruned)
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the structural plasticity state for saving.
        
        Returns:
            Dictionary with serialized state
        """
        return {
            'current_size': self.current_size,
            'max_size': self.max_size,
            'min_size': self.min_size,
            'growth_threshold': self.growth_threshold,
            'growth_rate': self.growth_rate,
            'prune_threshold': self.prune_threshold,
            'utility_window': self.utility_window,
            'growth_strategy': self.growth_strategy.value,
            'pruning_strategy': self.pruning_strategy.value,
            'growth_cooldown': self.growth_cooldown,
            'check_interval': self.check_interval,
            'novelty_threshold': self.novelty_threshold,
            'redundancy_threshold': self.redundancy_threshold,
            'enable_consolidation': self.enable_consolidation,
            'update_count': self.update_count,
            'last_growth_update': self.last_growth_update,
            'last_prune_update': self.last_prune_update,
            'utility_scores': self.utility_scores.tolist(),
            'activation_frequency': self.activation_frequency.tolist(),
            'growth_events': self.growth_events[-10:] if self.growth_events else [],
            'prune_events': self.prune_events[-10:] if self.prune_events else [],
            'prototype_count': len(self.prototype_patterns),
            'min_age_for_pruning': self.min_age_for_pruning,
            'structural_plasticity_mode': self.structural_plasticity_mode.value,
            'max_fan_in': self.max_fan_in,
            'max_fan_out': self.max_fan_out,
            'weight_init_mean': self.weight_init_mean,
            'weight_init_std': self.weight_init_std,
            'growth_increment': self.growth_increment,
            'update_interval': self.update_interval,
            'consolidation_threshold': self.consolidation_threshold,
            'enable_neuron_growth': self.enable_neuron_growth,
            'enable_connection_pruning': self.enable_connection_pruning,
            'enable_connection_sprouting': self.enable_connection_sprouting
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'StructuralPlasticity':
        """
        Create a structural plasticity instance from serialized data.
        
        Args:
            data: Dictionary with serialized state
            
        Returns:
            StructuralPlasticity instance
        """
        instance = cls(
            initial_size=data['current_size'],
            max_size=data['max_size'],
            min_size=data['min_size'],
            growth_threshold=data['growth_threshold'],
            growth_rate=data['growth_rate'],
            prune_threshold=data['prune_threshold'],
            utility_window=data['utility_window'],
            growth_strategy=data['growth_strategy'],
            pruning_strategy=data['pruning_strategy'],
            growth_cooldown=data['growth_cooldown'],
            check_interval=data['check_interval'],
            novelty_threshold=data['novelty_threshold'],
            redundancy_threshold=data['redundancy_threshold'],
            max_growth_per_step=data['max_growth_per_step'],
            max_prune_per_step=data['max_prune_per_step'],
            enable_consolidation=data['enable_consolidation'],
            min_age_for_pruning=data['min_age_for_pruning'],
            structural_plasticity_mode=data['structural_plasticity_mode'],
            max_fan_in=data['max_fan_in'],
            max_fan_out=data['max_fan_out'],
            weight_init_mean=data['weight_init_mean'],
            weight_init_std=data['weight_init_std'],
            growth_increment=data['growth_increment'],
            update_interval=data['update_interval'],
            consolidation_threshold=data['consolidation_threshold'],
            enable_neuron_growth=data['enable_neuron_growth'],
            enable_connection_pruning=data['enable_connection_pruning'],
            enable_connection_sprouting=data['enable_connection_sprouting']
        )
        
        # Restore additional state
        instance.update_count = data['update_count']
        instance.last_growth_update = data['last_growth_update']
        instance.last_prune_update = data['last_prune_update']
        instance.utility_scores = xp.array(data['utility_scores'])
        instance.activation_frequency = xp.array(data['activation_frequency'])
        instance.growth_events = data.get('growth_events', [])
        instance.prune_events = data.get('prune_events', [])
        
        return instance 