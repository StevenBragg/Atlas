#!/usr/bin/env python3
"""
Recursive Self-Improvement System for ATLAS Superintelligence

This module implements recursive self-improvement capabilities that enable:
1. Architecture search - Finding optimal network configurations
2. Hyperparameter optimization - Self-tuning of learning parameters
3. Performance monitoring - Tracking capabilities across dimensions
4. Capability expansion - Growing new abilities
5. Safe self-modification - Controlled, reversible changes

Core Principles:
- No backpropagation - uses evolutionary and local search methods
- Safety-first - All changes are reversible and monitored
- Gradual improvement - Small, verified steps
- Modular enhancement - Component-level optimization
- Maintains core principles while enhancing capabilities
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque
import logging
import time
import copy

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    """Types of self-improvement"""
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    ALGORITHM = "algorithm"
    CAPABILITY = "capability"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"


class SafetyLevel(Enum):
    """Safety levels for modifications"""
    MINIMAL = "minimal"  # Very safe, small changes
    MODERATE = "moderate"  # Reasonable changes with monitoring
    AGGRESSIVE = "aggressive"  # Larger changes, more risk


class ModificationStatus(Enum):
    """Status of a modification attempt"""
    PENDING = "pending"
    TESTING = "testing"
    APPLIED = "applied"
    REVERTED = "reverted"
    FAILED = "failed"


@dataclass
class PerformanceMetric:
    """A performance metric being tracked"""
    name: str
    current_value: float
    baseline_value: float
    target_value: float
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    improvement_rate: float = 0.0

    def update(self, new_value: float):
        """Update metric with new value"""
        self.history.append({
            'value': new_value,
            'timestamp': time.time()
        })
        old_value = self.current_value
        self.current_value = new_value

        # Compute improvement rate
        if len(self.history) > 1:
            self.improvement_rate = (new_value - old_value) / (abs(old_value) + 1e-8)

    def get_trend(self, window: int = 10) -> float:
        """Get trend over recent history"""
        if len(self.history) < 2:
            return 0.0

        recent = list(self.history)[-window:]
        if len(recent) < 2:
            return 0.0

        values = [h['value'] for h in recent]
        trend = (values[-1] - values[0]) / (len(values) - 1)
        return trend


@dataclass
class Modification:
    """A proposed or applied modification"""
    modification_id: str
    improvement_type: ImprovementType
    description: str
    parameters: Dict[str, Any]
    original_values: Dict[str, Any]
    new_values: Dict[str, Any]
    status: ModificationStatus = ModificationStatus.PENDING
    expected_improvement: float = 0.0
    actual_improvement: float = 0.0
    safety_level: SafetyLevel = SafetyLevel.MINIMAL
    creation_time: float = field(default_factory=time.time)
    application_time: Optional[float] = None
    reversion_time: Optional[float] = None


@dataclass
class Capability:
    """A capability that can be enhanced"""
    name: str
    level: float  # 0-1 capability level
    dependencies: List[str]  # Other capabilities this depends on
    enhancement_history: List[Dict[str, Any]] = field(default_factory=list)


class RecursiveSelfImprovement:
    """
    System for recursive self-improvement with safety guarantees.
    """

    def __init__(
        self,
        num_hyperparameters: int = 20,
        max_modifications: int = 100,
        safety_level: SafetyLevel = SafetyLevel.MODERATE,
        improvement_threshold: float = 0.05,
        reversion_threshold: float = -0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the self-improvement system.

        Args:
            num_hyperparameters: Number of hyperparameters to optimize
            max_modifications: Maximum modifications to track
            safety_level: Default safety level for modifications
            improvement_threshold: Minimum improvement to keep change
            reversion_threshold: Maximum degradation before reverting
            random_seed: Random seed for reproducibility
        """
        self.num_hyperparameters = num_hyperparameters
        self.max_modifications = max_modifications
        self.safety_level = safety_level
        self.improvement_threshold = improvement_threshold
        self.reversion_threshold = reversion_threshold

        if random_seed is not None:
            np.random.seed(random_seed)

        # Hyperparameter management
        self.hyperparameters: Dict[str, float] = {}
        self.hyperparameter_ranges: Dict[str, Tuple[float, float]] = {}
        self._initialize_hyperparameters()

        # Performance tracking
        self.metrics: Dict[str, PerformanceMetric] = {}
        self._initialize_metrics()

        # Modification history
        self.modifications: Dict[str, Modification] = {}
        self.modification_counter = 0
        self.pending_modifications: List[str] = []

        # Capability tracking
        self.capabilities: Dict[str, Capability] = {}
        self._initialize_capabilities()

        # Search state for architecture/algorithm search
        self.search_population: List[Dict[str, Any]] = []
        self.best_configuration: Optional[Dict[str, Any]] = None
        self.generation = 0

        # Safety monitoring
        self.safety_violations = 0
        self.reverted_modifications = 0
        self.successful_improvements = 0

        # Checkpoints for reversion
        self.checkpoints: deque = deque(maxlen=10)
        self._create_checkpoint("initial")

    def _initialize_hyperparameters(self):
        """Initialize hyperparameters with defaults and ranges"""
        defaults = {
            'learning_rate': (0.001, 0.0001, 0.1),  # (default, min, max)
            'momentum': (0.9, 0.0, 0.99),
            'weight_decay': (0.0001, 0.0, 0.01),
            'batch_size': (32, 1, 256),
            'hidden_dim': (64, 16, 512),
            'num_layers': (3, 1, 10),
            'dropout': (0.1, 0.0, 0.5),
            'temperature': (1.0, 0.1, 10.0),
            'exploration_rate': (0.1, 0.0, 1.0),
            'discount_factor': (0.99, 0.9, 0.999),
            'plasticity_rate': (0.01, 0.001, 0.1),
            'hebbian_rate': (0.01, 0.001, 0.1),
            'stdp_window': (20, 5, 100),
            'homeostatic_rate': (0.001, 0.0001, 0.01),
            'growth_threshold': (0.8, 0.5, 0.99),
            'pruning_threshold': (0.1, 0.01, 0.3),
            'attention_heads': (4, 1, 16),
            'memory_capacity': (1000, 100, 10000),
            'consolidation_rate': (0.1, 0.01, 0.5),
            'creativity_temperature': (1.0, 0.5, 2.0)
        }

        for name, (default, min_val, max_val) in defaults.items():
            self.hyperparameters[name] = default
            self.hyperparameter_ranges[name] = (min_val, max_val)

    def _initialize_metrics(self):
        """Initialize performance metrics"""
        metric_configs = {
            'prediction_accuracy': (0.5, 0.5, 0.95),  # (current, baseline, target)
            'learning_speed': (1.0, 1.0, 2.0),
            'memory_retention': (0.5, 0.5, 0.9),
            'generalization': (0.5, 0.5, 0.85),
            'creativity_score': (0.3, 0.3, 0.8),
            'reasoning_depth': (0.4, 0.4, 0.9),
            'efficiency': (0.5, 0.5, 0.9),
            'robustness': (0.5, 0.5, 0.95),
            'adaptability': (0.5, 0.5, 0.9),
            'overall_capability': (0.4, 0.4, 0.95)
        }

        for name, (current, baseline, target) in metric_configs.items():
            self.metrics[name] = PerformanceMetric(
                name=name,
                current_value=current,
                baseline_value=baseline,
                target_value=target
            )

    def _initialize_capabilities(self):
        """Initialize capability tracking"""
        capability_configs = {
            'perception': (0.6, []),
            'memory': (0.5, []),
            'reasoning': (0.4, ['memory']),
            'planning': (0.3, ['reasoning', 'memory']),
            'learning': (0.5, []),
            'creativity': (0.3, ['memory', 'reasoning']),
            'language': (0.2, ['memory', 'reasoning']),
            'social': (0.2, ['reasoning', 'memory']),
            'metacognition': (0.3, ['reasoning']),
            'self_improvement': (0.4, ['metacognition', 'learning'])
        }

        for name, (level, deps) in capability_configs.items():
            self.capabilities[name] = Capability(
                name=name,
                level=level,
                dependencies=deps
            )

    def _create_checkpoint(self, name: str):
        """Create a checkpoint of current state"""
        checkpoint = {
            'name': name,
            'timestamp': time.time(),
            'hyperparameters': copy.deepcopy(self.hyperparameters),
            'metrics': {n: m.current_value for n, m in self.metrics.items()},
            'capabilities': {n: c.level for n, c in self.capabilities.items()}
        }
        self.checkpoints.append(checkpoint)

    def propose_improvement(
        self,
        improvement_type: ImprovementType,
        target: str,
        strategy: str = "gradient_free"
    ) -> Optional[Modification]:
        """
        Propose an improvement to the system.

        Args:
            improvement_type: Type of improvement
            target: Target parameter/capability to improve
            strategy: Optimization strategy

        Returns:
            Proposed modification or None
        """
        if improvement_type == ImprovementType.HYPERPARAMETER:
            return self._propose_hyperparameter_change(target, strategy)
        elif improvement_type == ImprovementType.CAPABILITY:
            return self._propose_capability_enhancement(target)
        elif improvement_type == ImprovementType.ARCHITECTURE:
            return self._propose_architecture_change(target)
        else:
            return None

    def _propose_hyperparameter_change(
        self,
        param_name: str,
        strategy: str
    ) -> Optional[Modification]:
        """Propose a hyperparameter change"""
        if param_name not in self.hyperparameters:
            return None

        current_value = self.hyperparameters[param_name]
        min_val, max_val = self.hyperparameter_ranges[param_name]

        if strategy == "gradient_free":
            # Random perturbation scaled by safety level
            if self.safety_level == SafetyLevel.MINIMAL:
                scale = 0.05
            elif self.safety_level == SafetyLevel.MODERATE:
                scale = 0.1
            else:
                scale = 0.2

            delta = np.random.randn() * scale * (max_val - min_val)
            new_value = np.clip(current_value + delta, min_val, max_val)

        elif strategy == "evolutionary":
            # Sample from distribution around current best
            new_value = np.random.uniform(
                max(min_val, current_value - 0.1 * (max_val - min_val)),
                min(max_val, current_value + 0.1 * (max_val - min_val))
            )

        else:
            # Default: small random change
            new_value = current_value * (1 + 0.1 * np.random.randn())
            new_value = np.clip(new_value, min_val, max_val)

        mod_id = f"mod_{self.modification_counter}"
        self.modification_counter += 1

        modification = Modification(
            modification_id=mod_id,
            improvement_type=ImprovementType.HYPERPARAMETER,
            description=f"Adjust {param_name} from {current_value:.4f} to {new_value:.4f}",
            parameters={'param_name': param_name},
            original_values={param_name: current_value},
            new_values={param_name: new_value},
            safety_level=self.safety_level,
            expected_improvement=self._estimate_improvement(param_name, new_value)
        )

        self.modifications[mod_id] = modification
        self.pending_modifications.append(mod_id)

        return modification

    def _propose_capability_enhancement(self, capability_name: str) -> Optional[Modification]:
        """Propose a capability enhancement"""
        if capability_name not in self.capabilities:
            return None

        capability = self.capabilities[capability_name]

        # Check dependencies
        for dep in capability.dependencies:
            if dep in self.capabilities:
                dep_level = self.capabilities[dep].level
                if dep_level < 0.5:
                    # Dependency too weak, enhance it first
                    return self._propose_capability_enhancement(dep)

        # Propose level increase
        current_level = capability.level
        if self.safety_level == SafetyLevel.MINIMAL:
            increase = 0.05
        elif self.safety_level == SafetyLevel.MODERATE:
            increase = 0.1
        else:
            increase = 0.15

        new_level = min(1.0, current_level + increase)

        mod_id = f"mod_{self.modification_counter}"
        self.modification_counter += 1

        modification = Modification(
            modification_id=mod_id,
            improvement_type=ImprovementType.CAPABILITY,
            description=f"Enhance {capability_name} from {current_level:.2f} to {new_level:.2f}",
            parameters={'capability_name': capability_name},
            original_values={capability_name: current_level},
            new_values={capability_name: new_level},
            safety_level=self.safety_level,
            expected_improvement=increase
        )

        self.modifications[mod_id] = modification
        self.pending_modifications.append(mod_id)

        return modification

    def _propose_architecture_change(self, target: str) -> Optional[Modification]:
        """Propose an architecture change"""
        # Architecture changes are represented as hyperparameter configs
        architecture_params = ['hidden_dim', 'num_layers', 'attention_heads']

        changes = {}
        original = {}

        for param in architecture_params:
            if param in self.hyperparameters:
                original[param] = self.hyperparameters[param]
                min_val, max_val = self.hyperparameter_ranges[param]

                # Propose small change
                delta = np.random.choice([-1, 0, 1]) * (max_val - min_val) * 0.05
                changes[param] = np.clip(original[param] + delta, min_val, max_val)

        mod_id = f"mod_{self.modification_counter}"
        self.modification_counter += 1

        modification = Modification(
            modification_id=mod_id,
            improvement_type=ImprovementType.ARCHITECTURE,
            description=f"Architecture modification for {target}",
            parameters={'target': target},
            original_values=original,
            new_values=changes,
            safety_level=SafetyLevel.MODERATE,  # Architecture changes need caution
            expected_improvement=0.0  # Unknown
        )

        self.modifications[mod_id] = modification
        self.pending_modifications.append(mod_id)

        return modification

    def _estimate_improvement(self, param_name: str, new_value: float) -> float:
        """Estimate expected improvement from a change"""
        # Simple heuristic based on historical performance
        current = self.hyperparameters[param_name]

        # Check if similar changes worked before
        similar_mods = [m for m in self.modifications.values()
                       if m.status == ModificationStatus.APPLIED
                       and m.parameters.get('param_name') == param_name]

        if similar_mods:
            avg_improvement = np.mean([m.actual_improvement for m in similar_mods])
            return avg_improvement
        else:
            return 0.0  # Unknown

    def apply_modification(
        self,
        modification_id: str,
        test_duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply a proposed modification.

        Args:
            modification_id: ID of modification to apply
            test_duration: Time to test before finalizing

        Returns:
            Result of application
        """
        if modification_id not in self.modifications:
            return {'success': False, 'reason': 'Modification not found'}

        mod = self.modifications[modification_id]

        if mod.status != ModificationStatus.PENDING:
            return {'success': False, 'reason': f'Modification status is {mod.status.value}'}

        # Create checkpoint before modification
        self._create_checkpoint(f"before_{modification_id}")

        # Apply the modification
        mod.status = ModificationStatus.TESTING
        mod.application_time = time.time()

        if mod.improvement_type == ImprovementType.HYPERPARAMETER:
            param_name = mod.parameters['param_name']
            self.hyperparameters[param_name] = mod.new_values[param_name]

        elif mod.improvement_type == ImprovementType.CAPABILITY:
            cap_name = mod.parameters['capability_name']
            self.capabilities[cap_name].level = mod.new_values[cap_name]

        elif mod.improvement_type == ImprovementType.ARCHITECTURE:
            for param, value in mod.new_values.items():
                self.hyperparameters[param] = value

        # Remove from pending
        if modification_id in self.pending_modifications:
            self.pending_modifications.remove(modification_id)

        return {
            'success': True,
            'modification_id': modification_id,
            'status': 'testing'
        }

    def evaluate_modification(
        self,
        modification_id: str,
        performance_delta: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate a modification based on performance changes.

        Args:
            modification_id: ID of modification to evaluate
            performance_delta: Change in performance metrics

        Returns:
            Evaluation result with keep/revert decision
        """
        if modification_id not in self.modifications:
            return {'success': False, 'reason': 'Modification not found'}

        mod = self.modifications[modification_id]

        # Compute overall improvement
        improvements = list(performance_delta.values())
        overall_improvement = np.mean(improvements) if improvements else 0.0

        mod.actual_improvement = overall_improvement

        # Update metrics
        for metric_name, delta in performance_delta.items():
            if metric_name in self.metrics:
                new_value = self.metrics[metric_name].current_value + delta
                self.metrics[metric_name].update(new_value)

        # Decision based on thresholds
        if overall_improvement >= self.improvement_threshold:
            mod.status = ModificationStatus.APPLIED
            self.successful_improvements += 1
            decision = 'keep'
        elif overall_improvement <= self.reversion_threshold:
            self.revert_modification(modification_id)
            decision = 'revert'
        else:
            mod.status = ModificationStatus.APPLIED  # Keep marginal improvements
            decision = 'keep_marginal'

        return {
            'success': True,
            'modification_id': modification_id,
            'overall_improvement': overall_improvement,
            'decision': decision,
            'status': mod.status.value
        }

    def revert_modification(self, modification_id: str) -> Dict[str, Any]:
        """Revert a modification to original values"""
        if modification_id not in self.modifications:
            return {'success': False, 'reason': 'Modification not found'}

        mod = self.modifications[modification_id]

        # Restore original values
        if mod.improvement_type == ImprovementType.HYPERPARAMETER:
            param_name = mod.parameters['param_name']
            self.hyperparameters[param_name] = mod.original_values[param_name]

        elif mod.improvement_type == ImprovementType.CAPABILITY:
            cap_name = mod.parameters['capability_name']
            self.capabilities[cap_name].level = mod.original_values[cap_name]

        elif mod.improvement_type == ImprovementType.ARCHITECTURE:
            for param, value in mod.original_values.items():
                self.hyperparameters[param] = value

        mod.status = ModificationStatus.REVERTED
        mod.reversion_time = time.time()
        self.reverted_modifications += 1

        return {
            'success': True,
            'modification_id': modification_id,
            'status': 'reverted'
        }

    def run_optimization_cycle(
        self,
        n_proposals: int = 5,
        evaluation_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Run a full optimization cycle.

        Args:
            n_proposals: Number of improvements to propose
            evaluation_fn: Optional function to evaluate changes

        Returns:
            Cycle results
        """
        results = {
            'proposals': [],
            'applied': [],
            'improved': [],
            'reverted': []
        }

        # Generate proposals
        targets = list(self.hyperparameters.keys())[:n_proposals]

        for target in targets:
            mod = self.propose_improvement(
                ImprovementType.HYPERPARAMETER,
                target,
                "gradient_free"
            )
            if mod:
                results['proposals'].append(mod.modification_id)

        # Apply proposals
        for mod_id in results['proposals']:
            apply_result = self.apply_modification(mod_id)
            if apply_result['success']:
                results['applied'].append(mod_id)

                # Evaluate (using dummy evaluation if no function provided)
                if evaluation_fn:
                    config = {'hyperparameters': self.hyperparameters}
                    perf_delta = evaluation_fn(config)
                else:
                    # Simulated evaluation
                    perf_delta = {
                        'overall_capability': np.random.randn() * 0.1
                    }

                eval_result = self.evaluate_modification(mod_id, perf_delta)

                if eval_result['decision'] == 'keep':
                    results['improved'].append(mod_id)
                elif eval_result['decision'] == 'revert':
                    results['reverted'].append(mod_id)

        self.generation += 1

        return results

    def evolutionary_search(
        self,
        population_size: int = 10,
        generations: int = 5,
        evaluation_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    ) -> Dict[str, Any]:
        """
        Run evolutionary search for optimal configuration.

        Args:
            population_size: Size of population
            generations: Number of generations
            evaluation_fn: Fitness evaluation function

        Returns:
            Search results
        """
        # Initialize population
        if not self.search_population:
            for _ in range(population_size):
                individual = {}
                for param, (min_val, max_val) in self.hyperparameter_ranges.items():
                    individual[param] = np.random.uniform(min_val, max_val)
                self.search_population.append(individual)

        best_fitness = float('-inf')
        best_individual = None

        for gen in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in self.search_population:
                if evaluation_fn:
                    fitness = evaluation_fn(individual)
                else:
                    # Simulated fitness
                    fitness = np.random.rand()

                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                i, j = np.random.choice(len(self.search_population), 2, replace=False)
                winner = i if fitness_scores[i] > fitness_scores[j] else j
                selected.append(self.search_population[winner].copy())

            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, len(selected) - 1)]

                # Uniform crossover
                child1, child2 = {}, {}
                for param in parent1:
                    if np.random.rand() < 0.5:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]

                # Mutation
                for child in [child1, child2]:
                    for param in child:
                        if np.random.rand() < 0.1:  # Mutation rate
                            min_val, max_val = self.hyperparameter_ranges[param]
                            child[param] = np.clip(
                                child[param] + np.random.randn() * 0.1 * (max_val - min_val),
                                min_val, max_val
                            )

                new_population.extend([child1, child2])

            self.search_population = new_population[:population_size]
            self.generation += 1

        self.best_configuration = best_individual

        return {
            'generations': generations,
            'best_fitness': best_fitness,
            'best_configuration': best_individual,
            'final_population_size': len(self.search_population)
        }

    def apply_best_configuration(self) -> Dict[str, Any]:
        """Apply the best found configuration"""
        if not self.best_configuration:
            return {'success': False, 'reason': 'No best configuration found'}

        self._create_checkpoint("before_best_config")

        for param, value in self.best_configuration.items():
            if param in self.hyperparameters:
                self.hyperparameters[param] = value

        return {
            'success': True,
            'applied_config': self.best_configuration
        }

    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for what to improve next"""
        recommendations = []

        # Check metrics below target
        for name, metric in self.metrics.items():
            gap = metric.target_value - metric.current_value
            if gap > 0.1:
                recommendations.append({
                    'type': 'metric',
                    'target': name,
                    'current': metric.current_value,
                    'target_value': metric.target_value,
                    'gap': gap,
                    'priority': gap
                })

        # Check capabilities with weak dependencies
        for name, capability in self.capabilities.items():
            weak_deps = [dep for dep in capability.dependencies
                        if dep in self.capabilities
                        and self.capabilities[dep].level < 0.5]
            if weak_deps:
                recommendations.append({
                    'type': 'capability_dependency',
                    'target': name,
                    'weak_dependencies': weak_deps,
                    'priority': 0.5
                })

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return recommendations[:5]

    def get_stats(self) -> Dict[str, Any]:
        """Get self-improvement statistics"""
        return {
            'generation': self.generation,
            'total_modifications': len(self.modifications),
            'pending_modifications': len(self.pending_modifications),
            'successful_improvements': self.successful_improvements,
            'reverted_modifications': self.reverted_modifications,
            'safety_violations': self.safety_violations,
            'safety_level': self.safety_level.value,
            'checkpoints': len(self.checkpoints),
            'current_metrics': {n: m.current_value for n, m in self.metrics.items()},
            'current_capabilities': {n: c.level for n, c in self.capabilities.items()},
            'best_configuration': self.best_configuration
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize self-improvement system to dictionary"""
        return {
            'num_hyperparameters': self.num_hyperparameters,
            'max_modifications': self.max_modifications,
            'safety_level': self.safety_level.value,
            'improvement_threshold': self.improvement_threshold,
            'reversion_threshold': self.reversion_threshold,
            'hyperparameters': self.hyperparameters,
            'hyperparameter_ranges': self.hyperparameter_ranges,
            'metrics': {
                n: {
                    'current_value': m.current_value,
                    'baseline_value': m.baseline_value,
                    'target_value': m.target_value,
                    'improvement_rate': m.improvement_rate
                }
                for n, m in self.metrics.items()
            },
            'capabilities': {
                n: {
                    'level': c.level,
                    'dependencies': c.dependencies
                }
                for n, c in self.capabilities.items()
            },
            'modifications': {
                mid: {
                    'modification_id': m.modification_id,
                    'improvement_type': m.improvement_type.value,
                    'description': m.description,
                    'status': m.status.value,
                    'actual_improvement': m.actual_improvement
                }
                for mid, m in self.modifications.items()
            },
            'generation': self.generation,
            'best_configuration': self.best_configuration,
            'stats': self.get_stats()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'RecursiveSelfImprovement':
        """Deserialize self-improvement system from dictionary"""
        system = cls(
            num_hyperparameters=data['num_hyperparameters'],
            max_modifications=data['max_modifications'],
            safety_level=SafetyLevel(data['safety_level']),
            improvement_threshold=data['improvement_threshold'],
            reversion_threshold=data['reversion_threshold']
        )

        system.hyperparameters = data['hyperparameters']
        system.hyperparameter_ranges = data['hyperparameter_ranges']
        system.generation = data['generation']
        system.best_configuration = data['best_configuration']

        for n, mdata in data.get('metrics', {}).items():
            if n in system.metrics:
                system.metrics[n].current_value = mdata['current_value']
                system.metrics[n].baseline_value = mdata['baseline_value']
                system.metrics[n].target_value = mdata['target_value']
                system.metrics[n].improvement_rate = mdata['improvement_rate']

        for n, cdata in data.get('capabilities', {}).items():
            if n in system.capabilities:
                system.capabilities[n].level = cdata['level']

        return system
