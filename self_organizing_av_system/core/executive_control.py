#!/usr/bin/env python3
"""
Executive Control System for ATLAS Superintelligence

This module implements prefrontal cortex-inspired executive control that enables:
1. Inhibitory control - Suppressing inappropriate responses
2. Task switching - Context-aware switching between tasks
3. Working memory manipulation - Active maintenance and updating
4. Decision making under uncertainty
5. Metacognitive monitoring - Knowing what you know

Core Principles:
- No backpropagation - uses local Hebbian/STDP learning
- Biologically plausible - inspired by prefrontal cortex
- Hierarchical control - Top-down modulation of lower processes
- Adaptive - learns from experience and feedback
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class ControlSignal(Enum):
    """Types of executive control signals"""
    INHIBIT = "inhibit"
    ACTIVATE = "activate"
    SWITCH = "switch"
    MAINTAIN = "maintain"
    UPDATE = "update"
    GATE = "gate"


class CognitiveState(Enum):
    """States of cognitive processing"""
    FOCUSED = "focused"
    SWITCHING = "switching"
    IDLE = "idle"
    OVERLOADED = "overloaded"
    RECOVERY = "recovery"


class UncertaintyType(Enum):
    """Types of uncertainty in decision making"""
    EPISTEMIC = "epistemic"  # Uncertainty due to lack of knowledge
    ALEATORIC = "aleatoric"  # Inherent randomness
    COMPUTATIONAL = "computational"  # Due to limited processing


@dataclass
class Task:
    """Representation of a cognitive task"""
    task_id: str
    name: str
    priority: float
    context: np.ndarray  # Task context embedding
    requirements: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    completion_progress: float = 0.0
    estimated_effort: float = 1.0
    actual_effort: float = 0.0

    def update_progress(self, delta: float = 0.1):
        """Update task completion progress"""
        self.completion_progress = min(1.0, self.completion_progress + delta)
        self.actual_effort += delta


@dataclass
class InhibitionUnit:
    """Unit for inhibitory control"""
    target_id: str
    inhibition_strength: float
    decay_rate: float = 0.1
    active: bool = True
    creation_time: float = field(default_factory=time.time)

    def decay(self):
        """Apply decay to inhibition strength"""
        self.inhibition_strength *= (1 - self.decay_rate)
        if self.inhibition_strength < 0.01:
            self.active = False


@dataclass
class MetacognitiveAssessment:
    """Assessment of own cognitive state"""
    confidence: float
    uncertainty_type: UncertaintyType
    knowledge_state: Dict[str, float]  # What do I know about each domain
    processing_load: float
    time_pressure: float
    recommendations: List[str]


class ExecutiveController:
    """
    Prefrontal cortex-inspired executive control system.
    Manages attention, inhibition, task switching, and metacognition.
    """

    def __init__(
        self,
        control_dim: int = 64,
        max_tasks: int = 10,
        inhibition_threshold: float = 0.5,
        switch_cost: float = 0.2,
        learning_rate: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the executive controller.

        Args:
            control_dim: Dimension of control signals
            max_tasks: Maximum concurrent tasks
            inhibition_threshold: Threshold for triggering inhibition
            switch_cost: Cognitive cost of task switching
            learning_rate: Learning rate for adaptation
            random_seed: Random seed for reproducibility
        """
        self.control_dim = control_dim
        self.max_tasks = max_tasks
        self.inhibition_threshold = inhibition_threshold
        self.switch_cost = switch_cost
        self.learning_rate = learning_rate

        if random_seed is not None:
            np.random.seed(random_seed)

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []
        self.task_counter = 0

        # Cognitive state
        self.cognitive_state = CognitiveState.IDLE
        self.processing_load = 0.0
        self.fatigue = 0.0

        # Inhibition system
        self.inhibitions: Dict[str, InhibitionUnit] = {}
        self.global_inhibition = 0.0  # General suppression level

        # Control signal generation (learned weights)
        self.control_weights = np.random.randn(control_dim, control_dim) * 0.1
        self.control_bias = np.zeros(control_dim)

        # Context representation for task switching
        self.context_buffer = np.zeros(control_dim)
        self.context_history: deque = deque(maxlen=100)

        # Decision making under uncertainty
        self.uncertainty_estimates: Dict[str, float] = {}
        self.confidence_calibration = 1.0  # How well-calibrated is confidence

        # Metacognitive monitoring
        self.knowledge_states: Dict[str, float] = {}  # Domain -> knowledge level
        self.performance_history: deque = deque(maxlen=1000)
        self.prediction_accuracy: float = 0.5

        # Statistics
        self.total_decisions = 0
        self.successful_inhibitions = 0
        self.task_switches = 0
        self.total_processing_time = 0.0

    def inhibit(
        self,
        target_id: str,
        strength: float = 1.0,
        duration: Optional[float] = None
    ) -> bool:
        """
        Apply inhibitory control to a target.

        Args:
            target_id: ID of the target to inhibit
            strength: Strength of inhibition (0-1)
            duration: Optional duration; if None, decays naturally

        Returns:
            True if inhibition was applied
        """
        if strength < self.inhibition_threshold:
            return False

        inhibition = InhibitionUnit(
            target_id=target_id,
            inhibition_strength=min(1.0, strength),
            decay_rate=0.1 if duration is None else 0.0
        )

        self.inhibitions[target_id] = inhibition
        self.successful_inhibitions += 1

        logger.debug(f"Applied inhibition to {target_id} with strength {strength}")

        return True

    def release_inhibition(self, target_id: str) -> bool:
        """Release inhibition on a target"""
        if target_id in self.inhibitions:
            del self.inhibitions[target_id]
            return True
        return False

    def get_inhibition_level(self, target_id: str) -> float:
        """Get current inhibition level for a target"""
        if target_id in self.inhibitions:
            return self.inhibitions[target_id].inhibition_strength
        return 0.0

    def switch_task(
        self,
        new_task_id: str,
        preserve_context: bool = True
    ) -> Dict[str, Any]:
        """
        Switch to a new task with context preservation.

        Args:
            new_task_id: ID of task to switch to
            preserve_context: Whether to preserve current context

        Returns:
            Dictionary with switch results
        """
        if new_task_id not in self.tasks:
            return {'success': False, 'reason': 'Task not found'}

        result = {
            'success': True,
            'previous_task': self.current_task,
            'new_task': new_task_id,
            'switch_cost': self.switch_cost
        }

        # Save current context if task is active
        if self.current_task and preserve_context:
            self.task_stack.append(self.current_task)
            self.context_history.append(self.context_buffer.copy())

        # Apply switch cost (temporary processing reduction)
        self.processing_load += self.switch_cost
        self.cognitive_state = CognitiveState.SWITCHING

        # Load new task context
        new_task = self.tasks[new_task_id]
        self.context_buffer = new_task.context.copy()
        self.current_task = new_task_id

        self.task_switches += 1

        # Transition back to focused after switching
        self.cognitive_state = CognitiveState.FOCUSED

        logger.debug(f"Switched to task {new_task_id}")

        return result

    def add_task(
        self,
        name: str,
        priority: float,
        context: Optional[np.ndarray] = None,
        requirements: Optional[Dict[str, float]] = None
    ) -> str:
        """Add a new task to the executive control system"""
        if len(self.tasks) >= self.max_tasks:
            # Remove lowest priority completed task
            completed = [(tid, t) for tid, t in self.tasks.items()
                        if t.completion_progress >= 1.0]
            if completed:
                completed.sort(key=lambda x: x[1].priority)
                del self.tasks[completed[0][0]]
            else:
                # Remove lowest priority incomplete task
                lowest = min(self.tasks.items(), key=lambda x: x[1].priority)
                del self.tasks[lowest[0]]

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        if context is None:
            context = np.random.randn(self.control_dim)
            context /= np.linalg.norm(context) + 1e-8

        task = Task(
            task_id=task_id,
            name=name,
            priority=priority,
            context=context,
            requirements=requirements or {}
        )

        self.tasks[task_id] = task

        # If no current task, switch to this one
        if self.current_task is None:
            self.current_task = task_id
            self.cognitive_state = CognitiveState.FOCUSED

        return task_id

    def complete_task(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Mark a task as complete and potentially resume previous task"""
        if task_id is None:
            task_id = self.current_task

        if task_id not in self.tasks:
            return {'success': False, 'reason': 'Task not found'}

        task = self.tasks[task_id]
        task.completion_progress = 1.0

        result = {
            'success': True,
            'task_id': task_id,
            'actual_effort': task.actual_effort
        }

        # If completing current task, resume previous or go idle
        if task_id == self.current_task:
            if self.task_stack:
                prev_task = self.task_stack.pop()
                self.switch_task(prev_task, preserve_context=False)

                # Restore context
                if self.context_history:
                    self.context_buffer = self.context_history.pop()

                result['resumed_task'] = prev_task
            else:
                self.current_task = None
                self.cognitive_state = CognitiveState.IDLE

        return result

    def generate_control_signal(
        self,
        input_state: np.ndarray,
        signal_type: ControlSignal
    ) -> np.ndarray:
        """
        Generate a control signal based on input state and desired type.

        Uses learned weights for context-appropriate control.
        """
        input_state = np.asarray(input_state).flatten()[:self.control_dim]
        if len(input_state) < self.control_dim:
            input_state = np.pad(input_state, (0, self.control_dim - len(input_state)))

        # Base control signal from learned weights
        control = self.control_weights @ input_state + self.control_bias

        # Modulate based on signal type
        if signal_type == ControlSignal.INHIBIT:
            # Strong suppression
            control = -np.abs(control)
        elif signal_type == ControlSignal.ACTIVATE:
            # Strong activation
            control = np.abs(control)
        elif signal_type == ControlSignal.GATE:
            # Sigmoid gating
            control = 1 / (1 + np.exp(-control))
        elif signal_type == ControlSignal.MAINTAIN:
            # Add context buffer
            control = 0.5 * control + 0.5 * self.context_buffer

        # Apply current inhibition level
        control *= (1 - self.global_inhibition)

        return control

    def decide_under_uncertainty(
        self,
        options: List[Dict[str, Any]],
        context: Optional[np.ndarray] = None
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make a decision under uncertainty.

        Args:
            options: List of options with 'value', 'risk', and 'embedding'
            context: Optional context for decision

        Returns:
            Tuple of (chosen_index, confidence, decision_info)
        """
        if not options:
            return -1, 0.0, {'error': 'No options provided'}

        self.total_decisions += 1

        # Compute expected utility for each option
        utilities = []
        uncertainties = []

        for opt in options:
            value = opt.get('value', 0.0)
            risk = opt.get('risk', 0.0)
            embedding = opt.get('embedding', np.zeros(self.control_dim))

            embedding = np.asarray(embedding).flatten()[:self.control_dim]
            if len(embedding) < self.control_dim:
                embedding = np.pad(embedding, (0, self.control_dim - len(embedding)))

            # Context-dependent value adjustment
            if context is not None:
                context = np.asarray(context).flatten()[:self.control_dim]
                if len(context) < self.control_dim:
                    context = np.pad(context, (0, self.control_dim - len(context)))
                context_match = np.dot(embedding, context)
                value *= (1 + 0.2 * context_match)

            # Compute utility with risk adjustment
            # Higher fatigue -> more risk-averse
            risk_adjustment = risk * (1 + self.fatigue)
            utility = value - risk_adjustment

            utilities.append(utility)

            # Estimate uncertainty
            epistemic = 1 - self.knowledge_states.get('decision_domain', 0.5)
            aleatoric = risk
            uncertainty = np.sqrt(epistemic**2 + aleatoric**2)
            uncertainties.append(uncertainty)

        # Softmax selection with temperature based on uncertainty
        temperature = 0.5 + np.mean(uncertainties)
        utilities_arr = np.array(utilities)
        probs = np.exp(utilities_arr / temperature)
        probs /= np.sum(probs) + 1e-8

        # Choose based on probabilities (could be max for exploitation)
        chosen = np.argmax(probs)  # Exploit for now

        # Confidence is inverse of uncertainty
        confidence = 1 / (1 + uncertainties[chosen])
        confidence *= self.confidence_calibration

        decision_info = {
            'utilities': utilities,
            'uncertainties': uncertainties,
            'probabilities': probs.tolist(),
            'temperature': temperature,
            'epistemic_uncertainty': 1 - self.knowledge_states.get('decision_domain', 0.5),
            'fatigue_level': self.fatigue
        }

        return chosen, confidence, decision_info

    def update_from_feedback(
        self,
        decision_id: int,
        outcome: float,
        expected: float
    ):
        """
        Update decision-making based on feedback.
        Uses local Hebbian learning.
        """
        prediction_error = outcome - expected

        # Update confidence calibration
        if abs(prediction_error) > 0.5:
            # Over or under confident
            self.confidence_calibration *= 0.95
        else:
            self.confidence_calibration = min(1.0, self.confidence_calibration * 1.01)

        # Update prediction accuracy
        accuracy_update = 1 if abs(prediction_error) < 0.3 else 0
        self.prediction_accuracy = (0.99 * self.prediction_accuracy +
                                    0.01 * accuracy_update)

        # Store in performance history
        self.performance_history.append({
            'decision_id': decision_id,
            'prediction_error': prediction_error,
            'timestamp': time.time()
        })

        # Hebbian update of control weights if we have context
        if len(self.context_history) > 0:
            recent_context = self.context_history[-1]
            # Δw = η * error * context
            delta_w = self.learning_rate * prediction_error * np.outer(
                np.ones(self.control_dim), recent_context
            )
            self.control_weights += delta_w
            self.control_weights /= np.linalg.norm(self.control_weights) + 1e-8

    def assess_metacognition(self) -> MetacognitiveAssessment:
        """
        Perform metacognitive assessment of own cognitive state.
        """
        # Assess confidence based on recent performance
        if len(self.performance_history) > 0:
            recent = list(self.performance_history)[-20:]
            errors = [abs(p['prediction_error']) for p in recent]
            confidence = 1 - np.mean(errors)
        else:
            confidence = 0.5  # Default uncertainty

        # Determine primary uncertainty type
        if self.knowledge_states:
            avg_knowledge = np.mean(list(self.knowledge_states.values()))
            if avg_knowledge < 0.5:
                uncertainty_type = UncertaintyType.EPISTEMIC
            elif self.processing_load > 0.7:
                uncertainty_type = UncertaintyType.COMPUTATIONAL
            else:
                uncertainty_type = UncertaintyType.ALEATORIC
        else:
            uncertainty_type = UncertaintyType.EPISTEMIC

        # Generate recommendations
        recommendations = []

        if self.processing_load > 0.8:
            recommendations.append("Reduce cognitive load by completing or deferring tasks")

        if self.fatigue > 0.7:
            recommendations.append("Rest recommended to restore cognitive capacity")

        if confidence < 0.4:
            recommendations.append("Gather more information before making decisions")

        if len(self.tasks) > 0.8 * self.max_tasks:
            recommendations.append("Prioritize and complete pending tasks")

        return MetacognitiveAssessment(
            confidence=confidence,
            uncertainty_type=uncertainty_type,
            knowledge_state=self.knowledge_states.copy(),
            processing_load=self.processing_load,
            time_pressure=len(self.tasks) / self.max_tasks,
            recommendations=recommendations
        )

    def update_knowledge_state(self, domain: str, level: float):
        """Update knowledge level for a domain"""
        current = self.knowledge_states.get(domain, 0.5)
        # Gradual update with Hebbian-like rule
        self.knowledge_states[domain] = 0.9 * current + 0.1 * level

    def step(self, dt: float = 1.0):
        """
        Update executive control state over time.
        """
        # Update inhibitions
        for inh_id in list(self.inhibitions.keys()):
            inh = self.inhibitions[inh_id]
            inh.decay()
            if not inh.active:
                del self.inhibitions[inh_id]

        # Decay global inhibition
        self.global_inhibition *= 0.95

        # Update processing load
        self.processing_load *= 0.99
        if self.current_task and self.current_task in self.tasks:
            task = self.tasks[self.current_task]
            task.update_progress(0.1 * dt)
            self.processing_load = min(1.0, self.processing_load +
                                       task.requirements.get('cpu', 0.1))

        # Update fatigue
        if self.cognitive_state == CognitiveState.FOCUSED:
            self.fatigue = min(1.0, self.fatigue + 0.01 * dt)
        elif self.cognitive_state == CognitiveState.IDLE:
            self.fatigue = max(0.0, self.fatigue - 0.05 * dt)

        # Update cognitive state based on load and fatigue
        if self.processing_load > 0.9:
            self.cognitive_state = CognitiveState.OVERLOADED
        elif self.fatigue > 0.9:
            self.cognitive_state = CognitiveState.RECOVERY

        self.total_processing_time += dt

    def get_stats(self) -> Dict[str, Any]:
        """Get executive control statistics"""
        return {
            'total_tasks': len(self.tasks),
            'current_task': self.current_task,
            'cognitive_state': self.cognitive_state.value,
            'processing_load': self.processing_load,
            'fatigue': self.fatigue,
            'active_inhibitions': len(self.inhibitions),
            'global_inhibition': self.global_inhibition,
            'total_decisions': self.total_decisions,
            'successful_inhibitions': self.successful_inhibitions,
            'task_switches': self.task_switches,
            'prediction_accuracy': self.prediction_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'knowledge_domains': len(self.knowledge_states),
            'total_processing_time': self.total_processing_time
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize executive controller to dictionary"""
        return {
            'control_dim': self.control_dim,
            'max_tasks': self.max_tasks,
            'inhibition_threshold': self.inhibition_threshold,
            'switch_cost': self.switch_cost,
            'learning_rate': self.learning_rate,
            'tasks': {
                tid: {
                    'task_id': t.task_id,
                    'name': t.name,
                    'priority': t.priority,
                    'context': t.context.tolist(),
                    'requirements': t.requirements,
                    'completion_progress': t.completion_progress
                }
                for tid, t in self.tasks.items()
            },
            'current_task': self.current_task,
            'cognitive_state': self.cognitive_state.value,
            'processing_load': self.processing_load,
            'fatigue': self.fatigue,
            'control_weights': self.control_weights.tolist(),
            'control_bias': self.control_bias.tolist(),
            'knowledge_states': self.knowledge_states,
            'stats': self.get_stats()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ExecutiveController':
        """Deserialize executive controller from dictionary"""
        controller = cls(
            control_dim=data['control_dim'],
            max_tasks=data['max_tasks'],
            inhibition_threshold=data['inhibition_threshold'],
            switch_cost=data['switch_cost'],
            learning_rate=data['learning_rate']
        )

        controller.control_weights = np.array(data['control_weights'])
        controller.control_bias = np.array(data['control_bias'])
        controller.cognitive_state = CognitiveState(data['cognitive_state'])
        controller.processing_load = data['processing_load']
        controller.fatigue = data['fatigue']
        controller.knowledge_states = data['knowledge_states']
        controller.current_task = data['current_task']

        for tid, task_data in data.get('tasks', {}).items():
            task = Task(
                task_id=task_data['task_id'],
                name=task_data['name'],
                priority=task_data['priority'],
                context=np.array(task_data['context']),
                requirements=task_data['requirements'],
                completion_progress=task_data['completion_progress']
            )
            controller.tasks[tid] = task

        return controller
