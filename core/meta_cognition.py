#!/usr/bin/env python3
"""
Meta-Cognitive Monitor for ATLAS Superintelligence

This module implements meta-cognitive capabilities that enable Atlas to monitor
its own learning process, detect when it's stuck or confused, and adjust its
learning strategies accordingly.

Core Capabilities:
1. Learning Process Monitoring - Tracks learning progress and efficiency
2. Confusion Detection - Identifies when understanding breaks down
3. Strategy Adaptation - Adjusts learning approaches based on performance
4. Self-Assessment - Evaluates confidence and uncertainty
5. Resource Allocation - Manages cognitive resources effectively

Meta-Cognitive Processes:
    Monitoring → Evaluation → Strategy Selection → Implementation
         ↑                                      ↓
         └──────────── Feedback ←───────────────┘

Located in: core/meta_cognition.py
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum, auto
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """States of cognitive processing"""
    FOCUSED = "focused"           # Deep concentration
    EXPLORING = "exploring"       # Open exploration
    STRUGGLING = "struggling"     # Difficulty with task
    CONFUSED = "confused"         # Understanding breakdown
    STUCK = "stuck"               # No progress possible
    REFLECTING = "reflecting"     # Meta-cognitive processing
    ADAPTING = "adapting"         # Changing strategy


class LearningStrategy(Enum):
    """Available learning strategies"""
    DEEP_PRACTICE = "deep_practice"      # Focused, deliberate practice
    SPACED_REPETITION = "spaced"         # Distributed practice over time
    INTERLEAVING = "interleaving"        # Mix different topics
    ELABORATION = "elaboration"          # Connect to existing knowledge
    VISUALIZATION = "visualization"      # Create mental images
    ACTIVE_RECALL = "active_recall"      # Test yourself
    TEACHING = "teaching"                # Explain to others
    SIMPLIFICATION = "simplification"    # Break into simpler parts
    ANALOGY = "analogy"                  # Use analogies
    EXPLORATION = "exploration"          # Broad exploration


class UncertaintyType(Enum):
    """Types of uncertainty"""
    ALEATORIC = "aleatoric"       # Inherent randomness
    EPISTEMIC = "epistemic"       # Lack of knowledge
    MODEL = "model"               # Model uncertainty
    DATA = "data"                 # Data uncertainty


@dataclass
class LearningEpisode:
    """A single learning episode"""
    episode_id: str
    task_id: str
    strategy: LearningStrategy
    start_time: float
    
    # Performance metrics
    initial_performance: float = 0.0
    final_performance: float = 0.0
    improvement_rate: float = 0.0
    
    # Cognitive state
    initial_confidence: float = 0.5
    final_confidence: float = 0.5
    confusion_events: int = 0
    stuck_events: int = 0
    
    # Resources
    time_spent: float = 0.0
    attempts: int = 0
    
    # Outcome
    completed: bool = False
    success: bool = False
    
    def finish(self, success: bool, final_performance: float):
        """Mark episode as finished"""
        self.completed = True
        self.success = success
        self.final_performance = final_performance
        self.time_spent = time.time() - self.start_time
        
        if self.time_spent > 0:
            self.improvement_rate = (final_performance - self.initial_performance) / self.time_spent


@dataclass
class StrategyEffectiveness:
    """Track effectiveness of a learning strategy"""
    strategy: LearningStrategy
    usage_count: int = 0
    success_count: int = 0
    avg_improvement: float = 0.0
    avg_time: float = 0.0
    confusion_rate: float = 0.0
    
    def record_outcome(self, success: bool, improvement: float, time_spent: float, confused: bool):
        """Record an outcome for this strategy"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        
        # Update moving averages
        alpha = 0.1
        self.avg_improvement = (1 - alpha) * self.avg_improvement + alpha * improvement
        self.avg_time = (1 - alpha) * self.avg_time + alpha * time_spent
        
        if confused:
            self.confusion_rate = (1 - alpha) * self.confusion_rate + alpha * 1.0
        else:
            self.confusion_rate = (1 - alpha) * self.confusion_rate + alpha * 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0
    
    @property
    def efficiency(self) -> float:
        """Calculate efficiency score"""
        if self.avg_time <= 0:
            return 0.0
        return self.avg_improvement / self.avg_time * self.success_rate


@dataclass
class ConfusionEvent:
    """Record of a confusion event"""
    event_id: str
    timestamp: float
    task_id: str
    confusion_type: str
    severity: float  # 0-1
    context: Dict[str, Any] = field(default_factory=dict)
    resolution: Optional[str] = None
    resolved_at: Optional[float] = None
    
    def resolve(self, resolution: str):
        """Mark confusion as resolved"""
        self.resolution = resolution
        self.resolved_at = time.time()


@dataclass
class MetaCognitiveAssessment:
    """Assessment of current meta-cognitive state"""
    timestamp: float
    cognitive_state: CognitiveState
    
    # Confidence and uncertainty
    overall_confidence: float
    uncertainty_level: float
    uncertainty_type: Optional[UncertaintyType]
    
    # Learning assessment
    learning_rate: float
    progress_velocity: float
    efficiency_score: float
    
    # Recommendations
    recommended_strategy: Optional[LearningStrategy]
    suggested_action: str
    
    # Alerts
    is_stuck: bool
    is_confused: bool
    needs_intervention: bool


class MetaCognitiveMonitor:
    """
    Meta-Cognitive Monitor for ATLAS.
    
    This system enables Atlas to:
    1. Monitor its own learning process in real-time
    2. Detect confusion, stuck states, and inefficiency
    3. Assess confidence and uncertainty
    4. Select appropriate learning strategies
    5. Adapt strategies based on effectiveness
    
    The monitor tracks:
    - Learning episodes and their outcomes
    - Strategy effectiveness over time
    - Confusion and stuck events
    - Performance trends
    - Resource utilization
    
    Based on this information, it provides:
    - Real-time cognitive state assessment
    - Strategy recommendations
    - Intervention alerts
    - Learning optimization suggestions
    """
    
    def __init__(
        self,
        confusion_threshold: float = 0.7,
        stuck_threshold: float = 0.1,
        adaptation_rate: float = 0.1,
        window_size: int = 100,
        enable_auto_adaptation: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the meta-cognitive monitor.
        
        Args:
            confusion_threshold: Threshold for detecting confusion
            stuck_threshold: Performance improvement threshold for stuck detection
            adaptation_rate: Rate for strategy adaptation
            window_size: Size of performance history window
            enable_auto_adaptation: Enable automatic strategy adaptation
            random_seed: Random seed for reproducibility
        """
        self.confusion_threshold = confusion_threshold
        self.stuck_threshold = stuck_threshold
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.enable_auto_adaptation = enable_auto_adaptation
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Learning tracking
        self.learning_episodes: Dict[str, LearningEpisode] = {}
        self.active_episode: Optional[str] = None
        self.episode_counter = 0
        
        # Strategy tracking
        self.strategy_effectiveness: Dict[LearningStrategy, StrategyEffectiveness] = {
            strategy: StrategyEffectiveness(strategy=strategy)
            for strategy in LearningStrategy
        }
        
        # Confusion tracking
        self.confusion_events: deque = deque(maxlen=100)
        self.current_confusion: Optional[ConfusionEvent] = None
        
        # Performance history
        self.performance_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        self.error_history: deque = deque(maxlen=window_size)
        
        # Cognitive state
        self.current_state = CognitiveState.FOCUSED
        self.state_history: deque = deque(maxlen=window_size)
        
        # Statistics
        self.total_episodes = 0
        self.total_confusion_events = 0
        self.total_stuck_events = 0
        self.strategy_adaptations = 0
        
        logger.info(f"Initialized MetaCognitiveMonitor with confusion_threshold={confusion_threshold}")
    
    # ==================== Episode Management ====================
    
    def start_learning_episode(
        self,
        task_id: str,
        strategy: LearningStrategy,
        initial_performance: float = 0.0,
        initial_confidence: float = 0.5
    ) -> str:
        """
        Start a new learning episode.
        
        Args:
            task_id: Task being learned
            strategy: Learning strategy to use
            initial_performance: Initial performance level
            initial_confidence: Initial confidence level
            
        Returns:
            Episode ID
        """
        episode_id = f"ep_{self.episode_counter}_{int(time.time() * 1000)}"
        self.episode_counter += 1
        
        episode = LearningEpisode(
            episode_id=episode_id,
            task_id=task_id,
            strategy=strategy,
            start_time=time.time(),
            initial_performance=initial_performance,
            initial_confidence=initial_confidence
        )
        
        self.learning_episodes[episode_id] = episode
        self.active_episode = episode_id
        self.current_state = CognitiveState.FOCUSED
        
        logger.debug(f"Started learning episode {episode_id} with strategy {strategy.value}")
        return episode_id
    
    def update_episode_progress(
        self,
        performance: float,
        confidence: float,
        error: Optional[float] = None
    ):
        """
        Update progress of active learning episode.
        
        Args:
            performance: Current performance level
            confidence: Current confidence level
            error: Optional error metric
        """
        if self.active_episode is None:
            return
        
        episode = self.learning_episodes[self.active_episode]
        episode.attempts += 1
        
        # Update histories
        self.performance_history.append(performance)
        self.confidence_history.append(confidence)
        if error is not None:
            self.error_history.append(error)
        
        # Check for confusion
        if confidence < (1 - self.confusion_threshold):
            self._detect_confusion(performance, confidence)
        
        # Check if stuck
        if self._detect_stuck():
            episode.stuck_events += 1
            self.current_state = CognitiveState.STUCK
        
        # Update episode
        episode.final_performance = performance
        episode.final_confidence = confidence
    
    def finish_learning_episode(self, success: bool) -> Dict[str, Any]:
        """
        Finish the active learning episode.
        
        Args:
            success: Whether the episode was successful
            
        Returns:
            Episode summary
        """
        if self.active_episode is None:
            return {}
        
        episode = self.learning_episodes[self.active_episode]
        episode.finish(success, episode.final_performance)
        
        # Update strategy effectiveness
        improvement = episode.final_performance - episode.initial_performance
        confused = episode.confusion_events > 0
        
        self.strategy_effectiveness[episode.strategy].record_outcome(
            success=success,
            improvement=improvement,
            time_spent=episode.time_spent,
            confused=confused
        )
        
        self.total_episodes += 1
        self.active_episode = None
        self.current_state = CognitiveState.REFLECTING
        
        logger.debug(f"Finished learning episode {episode.episode_id}: success={success}")
        
        return {
            'episode_id': episode.episode_id,
            'success': success,
            'improvement': improvement,
            'time_spent': episode.time_spent,
            'strategy': episode.strategy.value,
            'confusion_events': episode.confusion_events,
            'stuck_events': episode.stuck_events
        }
    
    # ==================== Confusion Detection ====================
    
    def _detect_confusion(self, performance: float, confidence: float):
        """Detect if the system is confused"""
        # Confusion indicators:
        # 1. Low confidence
        # 2. High error rate
        # 3. Inconsistent performance
        
        confusion_score = 0.0
        
        # Low confidence contributes to confusion
        confusion_score += (1 - confidence) * 0.4
        
        # High error contributes
        if self.error_history:
            recent_error = np.mean(list(self.error_history)[-5:])
            confusion_score += recent_error * 0.3
        
        # Inconsistent performance contributes
        if len(self.performance_history) >= 5:
            recent = list(self.performance_history)[-5:]
            variance = np.var(recent)
            confusion_score += min(1.0, variance * 2) * 0.3
        
        if confusion_score > self.confusion_threshold:
            self._record_confusion_event(confusion_score)
    
    def _record_confusion_event(self, severity: float):
        """Record a confusion event"""
        if self.current_confusion is not None:
            return  # Already confused
        
        event_id = f"conf_{self.total_confusion_events}_{int(time.time() * 1000)}"
        
        event = ConfusionEvent(
            event_id=event_id,
            timestamp=time.time(),
            task_id=self.learning_episodes[self.active_episode].task_id if self.active_episode else "unknown",
            confusion_type="understanding_breakdown",
            severity=severity,
            context={
                'performance_history': list(self.performance_history)[-5:],
                'confidence_history': list(self.confidence_history)[-5:]
            }
        )
        
        self.confusion_events.append(event)
        self.current_confusion = event
        self.total_confusion_events += 1
        self.current_state = CognitiveState.CONFUSED
        
        # Update episode
        if self.active_episode:
            self.learning_episodes[self.active_episode].confusion_events += 1
        
        logger.warning(f"Confusion detected: severity={severity:.3f}")
    
    def resolve_confusion(self, resolution: str):
        """
        Mark current confusion as resolved.
        
        Args:
            resolution: How the confusion was resolved
        """
        if self.current_confusion is None:
            return
        
        self.current_confusion.resolve(resolution)
        self.current_confusion = None
        self.current_state = CognitiveState.ADAPTING
        
        logger.info(f"Confusion resolved: {resolution}")
    
    # ==================== Stuck Detection ====================
    
    def _detect_stuck(self) -> bool:
        """Detect if learning is stuck"""
        if len(self.performance_history) < 10:
            return False
        
        # Check recent performance trend
        recent = list(self.performance_history)[-10:]
        
        # Calculate trend
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
            
            # Stuck if improvement is below threshold
            if trend < self.stuck_threshold:
                self.total_stuck_events += 1
                return True
        
        # Also check for oscillation (going back and forth)
        if len(recent) >= 5:
            diffs = np.diff(recent)
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            if sign_changes >= 3:  # Too much oscillation
                return True
        
        return False
    
    def is_stuck(self) -> bool:
        """Check if currently stuck"""
        return self.current_state == CognitiveState.STUCK
    
    def is_confused(self) -> bool:
        """Check if currently confused"""
        return self.current_state == CognitiveState.CONFUSED
    
    # ==================== Strategy Selection ====================
    
    def recommend_strategy(self, task_characteristics: Optional[Dict] = None) -> LearningStrategy:
        """
        Recommend a learning strategy based on current state and history.
        
        Args:
            task_characteristics: Optional task characteristics
            
        Returns:
            Recommended strategy
        """
        # If confused, try simplification or elaboration
        if self.current_state == CognitiveState.CONFUSED:
            return LearningStrategy.SIMPLIFICATION
        
        # If stuck, try a different approach
        if self.current_state == CognitiveState.STUCK:
            return LearningStrategy.ANALOGY
        
        # Otherwise, use most effective strategy
        best_strategy = max(
            self.strategy_effectiveness.values(),
            key=lambda s: s.efficiency
        )
        
        # If no data, default to deep practice
        if best_strategy.usage_count == 0:
            return LearningStrategy.DEEP_PRACTICE
        
        return best_strategy.strategy
    
    def adapt_strategy(self, forced: bool = False) -> Optional[LearningStrategy]:
        """
        Adapt learning strategy based on current performance.
        
        Args:
            forced: Force adaptation even if not stuck/confused
            
        Returns:
            New strategy if adapted, None otherwise
        """
        if not self.enable_auto_adaptation and not forced:
            return None
        
        should_adapt = forced or self.current_state in [CognitiveState.STUCK, CognitiveState.CONFUSED]
        
        if not should_adapt:
            return None
        
        # Select new strategy
        new_strategy = self.recommend_strategy()
        
        # Don't adapt to same strategy
        if self.active_episode:
            current_strategy = self.learning_episodes[self.active_episode].strategy
            if new_strategy == current_strategy:
                # Pick second best
                strategies = sorted(
                    self.strategy_effectiveness.values(),
                    key=lambda s: s.efficiency,
                    reverse=True
                )
                for s in strategies:
                    if s.strategy != current_strategy:
                        new_strategy = s.strategy
                        break
        
        self.strategy_adaptations += 1
        self.current_state = CognitiveState.ADAPTING
        
        logger.info(f"Adapting strategy to {new_strategy.value}")
        return new_strategy
    
    # ==================== Assessment ====================
    
    def assess_metacognition(self) -> MetaCognitiveAssessment:
        """
        Perform comprehensive meta-cognitive assessment.
        
        Returns:
            MetaCognitiveAssessment
        """
        # Calculate metrics
        learning_rate = self._calculate_learning_rate()
        progress_velocity = self._calculate_progress_velocity()
        efficiency = self._calculate_efficiency()
        
        # Determine uncertainty type
        uncertainty_type = self._classify_uncertainty()
        
        # Determine cognitive state
        cognitive_state = self.current_state
        
        # Calculate overall confidence
        if self.confidence_history:
            overall_confidence = np.mean(list(self.confidence_history)[-10:])
        else:
            overall_confidence = 0.5
        
        # Calculate uncertainty level
        uncertainty_level = 1.0 - overall_confidence
        
        # Recommend strategy
        recommended_strategy = self.recommend_strategy()
        
        # Generate suggestion
        suggested_action = self._generate_suggestion(cognitive_state)
        
        # Determine if intervention needed
        needs_intervention = (
            cognitive_state in [CognitiveState.STUCK, CognitiveState.CONFUSED] or
            uncertainty_level > 0.8 or
            efficiency < 0.2
        )
        
        assessment = MetaCognitiveAssessment(
            timestamp=time.time(),
            cognitive_state=cognitive_state,
            overall_confidence=overall_confidence,
            uncertainty_level=uncertainty_level,
            uncertainty_type=uncertainty_type,
            learning_rate=learning_rate,
            progress_velocity=progress_velocity,
            efficiency_score=efficiency,
            recommended_strategy=recommended_strategy,
            suggested_action=suggested_action,
            is_stuck=cognitive_state == CognitiveState.STUCK,
            is_confused=cognitive_state == CognitiveState.CONFUSED,
            needs_intervention=needs_intervention
        )
        
        self.state_history.append(cognitive_state)
        
        return assessment
    
    def _calculate_learning_rate(self) -> float:
        """Calculate current learning rate"""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        if len(recent) < 2:
            return 0.0
        
        # Fit linear trend
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def _calculate_progress_velocity(self) -> float:
        """Calculate progress velocity"""
        if not self.learning_episodes:
            return 0.0
        
        recent_episodes = sorted(
            self.learning_episodes.values(),
            key=lambda e: e.start_time,
            reverse=True
        )[:10]
        
        if not recent_episodes:
            return 0.0
        
        improvements = [e.final_performance - e.initial_performance for e in recent_episodes if e.completed]
        
        if not improvements:
            return 0.0
        
        return np.mean(improvements)
    
    def _calculate_efficiency(self) -> float:
        """Calculate learning efficiency"""
        if not self.learning_episodes:
            return 0.5
        
        completed = [e for e in self.learning_episodes.values() if e.completed]
        
        if not completed:
            return 0.5
        
        # Efficiency = improvement per unit time
        efficiencies = []
        for episode in completed:
            if episode.time_spent > 0:
                eff = (episode.final_performance - episode.initial_performance) / episode.time_spent
                efficiencies.append(eff)
        
        if not efficiencies:
            return 0.5
        
        # Normalize to 0-1 range
        avg_eff = np.mean(efficiencies)
        return min(1.0, max(0.0, avg_eff * 10 + 0.5))
    
    def _classify_uncertainty(self) -> Optional[UncertaintyType]:
        """Classify the type of uncertainty"""
        if len(self.performance_history) < 5:
            return UncertaintyType.EPISTEMIC  # Not enough data
        
        recent = list(self.performance_history)[-10:]
        variance = np.var(recent)
        
        # High variance suggests model uncertainty
        if variance > 0.2:
            return UncertaintyType.MODEL
        
        # Check for data uncertainty (inconsistent with low variance)
        if self.error_history:
            avg_error = np.mean(list(self.error_history)[-5:])
            if avg_error > 0.3:
                return UncertaintyType.DATA
        
        # Default to epistemic (lack of knowledge)
        return UncertaintyType.EPISTEMIC
    
    def _generate_suggestion(self, state: CognitiveState) -> str:
        """Generate a suggestion based on cognitive state"""
        suggestions = {
            CognitiveState.FOCUSED: "Continue with current approach",
            CognitiveState.EXPLORING: "Maintain open exploration",
            CognitiveState.STRUGGLING: "Consider breaking task into smaller parts",
            CognitiveState.CONFUSED: "Review fundamentals and seek clarification",
            CognitiveState.STUCK: "Try a completely different approach or strategy",
            CognitiveState.REFLECTING: "Take time to consolidate learning",
            CognitiveState.ADAPTING: "Implement the new strategy and monitor results"
        }
        
        return suggestions.get(state, "Continue monitoring progress")
    
    # ==================== Utility Methods ====================
    
    def get_strategy_rankings(self) -> List[Tuple[LearningStrategy, float]]:
        """
        Get ranking of strategies by effectiveness.
        
        Returns:
            List of (strategy, efficiency) tuples
        """
        rankings = [
            (se.strategy, se.efficiency)
            for se in self.strategy_effectiveness.values()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_learning_summary(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent learning.
        
        Args:
            n_episodes: Number of recent episodes to include
            
        Returns:
            Learning summary
        """
        recent = sorted(
            self.learning_episodes.values(),
            key=lambda e: e.start_time,
            reverse=True
        )[:n_episodes]
        
        completed = [e for e in recent if e.completed]
        
        return {
            'total_episodes': len(self.learning_episodes),
            'recent_episodes': len(recent),
            'completed_episodes': len(completed),
            'success_rate': sum(1 for e in completed if e.success) / len(completed) if completed else 0.0,
            'avg_improvement': np.mean([e.final_performance - e.initial_performance for e in completed]) if completed else 0.0,
            'total_confusion_events': self.total_confusion_events,
            'total_stuck_events': self.total_stuck_events,
            'strategy_adaptations': self.strategy_adaptations,
            'current_state': self.current_state.value
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get meta-cognitive statistics"""
        return {
            'total_episodes': self.total_episodes,
            'total_confusion_events': self.total_confusion_events,
            'total_stuck_events': self.total_stuck_events,
            'strategy_adaptations': self.strategy_adaptations,
            'current_state': self.current_state.value,
            'active_episode': self.active_episode,
            'current_confusion': self.current_confusion.event_id if self.current_confusion else None,
            'strategy_rankings': [
                {'strategy': s.value, 'efficiency': e, 'usage_count': self.strategy_effectiveness[s].usage_count}
                for s, e in self.get_strategy_rankings()
            ],
            'learning_summary': self.get_learning_summary()
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize meta-cognitive monitor"""
        return {
            'confusion_threshold': self.confusion_threshold,
            'stuck_threshold': self.stuck_threshold,
            'adaptation_rate': self.adaptation_rate,
            'enable_auto_adaptation': self.enable_auto_adaptation,
            'stats': self.get_stats(),
            'strategy_effectiveness': {
                s.value: {
                    'strategy': se.strategy.value,
                    'usage_count': se.usage_count,
                    'success_count': se.success_count,
                    'avg_improvement': se.avg_improvement,
                    'avg_time': se.avg_time,
                    'confusion_rate': se.confusion_rate
                }
                for s, se in self.strategy_effectiveness.items()
            },
            'confusion_history': [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp,
                    'severity': e.severity,
                    'resolved': e.resolved_at is not None
                }
                for e in self.confusion_events
            ]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MetaCognitiveMonitor':
        """Deserialize meta-cognitive monitor"""
        monitor = cls(
            confusion_threshold=data['confusion_threshold'],
            stuck_threshold=data['stuck_threshold'],
            adaptation_rate=data['adaptation_rate'],
            enable_auto_adaptation=data['enable_auto_adaptation']
        )
        
        # Restore strategy effectiveness
        for strategy_data in data.get('strategy_effectiveness', {}).values():
            strategy = LearningStrategy(strategy_data['strategy'])
            se = monitor.strategy_effectiveness[strategy]
            se.usage_count = strategy_data['usage_count']
            se.success_count = strategy_data['success_count']
            se.avg_improvement = strategy_data['avg_improvement']
            se.avg_time = strategy_data['avg_time']
            se.confusion_rate = strategy_data['confusion_rate']
        
        return monitor
