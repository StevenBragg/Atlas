#!/usr/bin/env python3
"""
Autonomous Goal Generation System for ATLAS Superintelligence

This module implements autonomous goal generation and intrinsic motivation,
enabling Atlas to set its own learning objectives and engage in curiosity-driven
exploration without external direction.

Core Capabilities:
1. Intrinsic Motivation System - Drives based on curiosity, competence, autonomy
2. Goal Generation - Creates novel learning goals autonomously
3. Curiosity-Driven Exploration - Seeks novelty and information gain
4. Goal Hierarchy - Manages short-term and long-term objectives
5. Value Learning - Learns what goals are worth pursuing

Motivation Theory Integration:
- Self-Determination Theory: Competence, Autonomy, Relatedness
- Information Gain: Seeking novelty and reducing uncertainty
- Flow Theory: Balancing challenge with capability
- Curiosity: Preference for the learnable unknown

Located in: core/autonomous_goals.py
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum, auto
from collections import deque, defaultdict
import heapq

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of autonomous goals"""
    LEARNING = "learning"           # Acquire new knowledge
    EXPLORATION = "exploration"     # Discover new patterns
    MASTERY = "mastery"             # Improve existing skills
    CREATION = "creation"           # Generate novel outputs
    OPTIMIZATION = "optimization"   # Improve efficiency
    SOCIAL = "social"               # Interact and learn from others
    META = "meta"                   # Self-improvement goals
    CURIOSITY = "curiosity"         # Driven by information gap


class GoalStatus(Enum):
    """Status of a goal"""
    PROPOSED = "proposed"           # Just generated, not evaluated
    ACTIVE = "active"               # Currently pursuing
    PAUSED = "paused"               # Temporarily suspended
    COMPLETED = "completed"         # Successfully achieved
    FAILED = "failed"               # Could not achieve
    ABANDONED = "abandoned"         # Deliberately given up


class IntrinsicDrive(Enum):
    """Intrinsic motivation drives based on Self-Determination Theory"""
    CURIOSITY = "curiosity"         # Desire to learn and explore
    COMPETENCE = "competence"       # Desire to be effective
    AUTONOMY = "autonomy"           # Desire for self-direction
    RELATEDNESS = "relatedness"     # Desire to connect with others
    CREATIVITY = "creativity"       # Desire to create and express


@dataclass
class Goal:
    """An autonomous goal"""
    goal_id: str
    name: str
    goal_type: GoalType
    description: str
    
    # Goal structure
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    
    # Motivation
    intrinsic_value: float = 0.5      # 0-1 intrinsic motivation
    extrinsic_value: float = 0.0      # 0-1 external reward
    urgency: float = 0.5              # 0-1 time pressure
    
    # Status
    status: GoalStatus = GoalStatus.PROPOSED
    progress: float = 0.0             # 0-1 completion
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    
    # Requirements and outcomes
    required_capabilities: List[str] = field(default_factory=list)
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Learning
    novelty_estimate: float = 0.5     # Expected novelty
    difficulty_estimate: float = 0.5  # Expected difficulty
    information_gain: float = 0.0     # Expected information gain
    
    # Statistics
    activation_count: int = 0
    last_activated: Optional[float] = None
    
    def compute_priority(self) -> float:
        """Compute dynamic priority score"""
        # Base priority from intrinsic value
        priority = self.intrinsic_value * 0.4
        
        # Add urgency component
        if self.deadline:
            time_remaining = self.deadline - time.time()
            if time_remaining > 0:
                urgency_factor = 1.0 - (time_remaining / (time_remaining + 3600))
                priority += urgency_factor * 0.3
            else:
                priority += 0.3  # Past deadline
        else:
            priority += self.urgency * 0.3
        
        # Add information gain component
        priority += self.information_gain * 0.2
        
        # Modulate by difficulty (prefer challenging but achievable)
        difficulty_bonus = 1.0 - abs(self.difficulty_estimate - 0.6)
        priority *= (0.5 + 0.5 * difficulty_bonus)
        
        return min(1.0, max(0.0, priority))
    
    def activate(self):
        """Record activation"""
        self.activation_count += 1
        self.last_activated = time.time()
        if self.status == GoalStatus.PROPOSED:
            self.status = GoalStatus.ACTIVE
            self.started_at = time.time()


@dataclass
class DriveState:
    """Current state of intrinsic drives"""
    curiosity: float = 1.0            # 0-1 curiosity level
    competence: float = 0.5           # 0-1 competence satisfaction
    autonomy: float = 0.5             # 0-1 autonomy satisfaction
    relatedness: float = 0.3          # 0-1 relatedness satisfaction
    creativity: float = 0.5           # 0-1 creative drive
    
    # Drive dynamics
    curiosity_decay: float = 0.95     # Curiosity decay rate
    curiosity_recovery: float = 0.1   # Curiosity recovery rate
    
    def update(self, goal_achieved: bool, novelty_encountered: float):
        """Update drive states based on experience"""
        if goal_achieved:
            self.competence = min(1.0, self.competence + 0.1)
            self.autonomy = min(1.0, self.autonomy + 0.05)
        else:
            self.competence = max(0.0, self.competence - 0.05)
        
        # Curiosity dynamics
        self.curiosity *= self.curiosity_decay
        self.curiosity += novelty_encountered * self.curiosity_recovery
        self.curiosity = min(1.0, max(0.0, self.curiosity))
        
        # Creativity grows with competence
        self.creativity = 0.3 + 0.7 * self.competence
    
    def get_dominant_drive(self) -> IntrinsicDrive:
        """Get the currently dominant drive"""
        drives = {
            IntrinsicDrive.CURIOSITY: self.curiosity,
            IntrinsicDrive.COMPETENCE: self.competence,
            IntrinsicDrive.AUTONOMY: self.autonomy,
            IntrinsicDrive.RELATEDNESS: self.relatedness,
            IntrinsicDrive.CREATIVITY: self.creativity
        }
        return max(drives, key=drives.get)


@dataclass
class ExplorationFrontier:
    """A frontier for curiosity-driven exploration"""
    frontier_id: str
    domain: str
    novelty_score: float
    predicted_difficulty: float
    estimated_reward: float
    last_explored: Optional[float] = None
    exploration_count: int = 0
    
    def compute_exploration_value(self) -> float:
        """Compute value of exploring this frontier"""
        # Novelty is primary driver
        value = self.novelty_score * 0.5
        
        # Add estimated reward
        value += self.estimated_reward * 0.3
        
        # Prefer less explored areas
        exploration_penalty = 1.0 / (1 + self.exploration_count * 0.1)
        value *= exploration_penalty
        
        # Prefer achievable challenges
        difficulty_factor = 1.0 - abs(self.predicted_difficulty - 0.6)
        value *= (0.5 + 0.5 * difficulty_factor)
        
        return value


class AutonomousGoalSystem:
    """
    Autonomous Goal Generation System for ATLAS.
    
    This system enables Atlas to:
    1. Generate its own learning goals based on intrinsic motivation
    2. Engage in curiosity-driven exploration
    3. Balance multiple intrinsic drives (curiosity, competence, autonomy)
    4. Create goal hierarchies for complex objectives
    5. Learn which goals are worth pursuing
    
    The system maintains:
    - A set of active goals
    - Intrinsic drive states
    - Exploration frontiers
    - Goal achievement history
    - Value function for goal evaluation
    """
    
    def __init__(
        self,
        max_active_goals: int = 10,
        max_goal_queue: int = 50,
        curiosity_threshold: float = 0.3,
        novelty_decay: float = 0.95,
        enable_meta_goals: bool = True,
        enable_social_goals: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the autonomous goal system.
        
        Args:
            max_active_goals: Maximum simultaneously active goals
            max_goal_queue: Maximum goals in queue
            curiosity_threshold: Threshold for curiosity-driven exploration
            novelty_decay: Rate at which novelty decays
            enable_meta_goals: Enable self-improvement goals
            enable_social_goals: Enable social learning goals
            random_seed: Random seed for reproducibility
        """
        self.max_active_goals = max_active_goals
        self.max_goal_queue = max_goal_queue
        self.curiosity_threshold = curiosity_threshold
        self.novelty_decay = novelty_decay
        self.enable_meta_goals = enable_meta_goals
        self.enable_social_goals = enable_social_goals
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Goal storage
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()
        self.goal_queue: List[Tuple[float, str]] = []  # (priority, goal_id)
        self.completed_goals: List[str] = []
        self.failed_goals: List[str] = []
        
        # Intrinsic drives
        self.drives = DriveState()
        
        # Exploration frontiers
        self.exploration_frontiers: Dict[str, ExplorationFrontier] = {}
        
        # Goal generation tracking
        self.generation_history: deque = deque(maxlen=1000)
        self.goal_success_rates: Dict[GoalType, float] = defaultdict(lambda: 0.5)
        
        # Value learning
        self.goal_value_estimates: Dict[str, float] = {}
        self.learning_rate = 0.1
        
        # Statistics
        self.total_goals_generated = 0
        self.total_goals_completed = 0
        self.total_exploration_steps = 0
        
        # Initialize with meta-goals
        if self.enable_meta_goals:
            self._initialize_meta_goals()
        
        logger.info(f"Initialized AutonomousGoalSystem with max_active_goals={max_active_goals}")
    
    def _initialize_meta_goals(self):
        """Initialize fundamental meta-goals"""
        meta_goals = [
            {
                'name': 'continuous_learning',
                'description': 'Continuously learn and improve capabilities',
                'type': GoalType.META,
                'intrinsic_value': 1.0
            },
            {
                'name': 'self_improvement',
                'description': 'Improve own algorithms and efficiency',
                'type': GoalType.META,
                'intrinsic_value': 0.95
            },
            {
                'name': 'knowledge_integration',
                'description': 'Integrate knowledge across modalities',
                'type': GoalType.META,
                'intrinsic_value': 0.9
            }
        ]
        
        for goal_spec in meta_goals:
            self.generate_goal(
                goal_type=goal_spec['type'],
                name=goal_spec['name'],
                description=goal_spec['description'],
                intrinsic_value=goal_spec['intrinsic_value']
            )
    
    # ==================== Goal Generation ====================
    
    def generate_goal(
        self,
        goal_type: Optional[GoalType] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        intrinsic_value: Optional[float] = None,
        parent_goal_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Autonomously generate a new goal.
        
        Args:
            goal_type: Type of goal (inferred if None)
            name: Goal name (generated if None)
            description: Goal description
            intrinsic_value: Intrinsic motivation value
            parent_goal_id: Parent goal for hierarchy
            context: Additional context
            
        Returns:
            Goal ID if generated, None otherwise
        """
        # Check capacity
        if len(self.goals) >= self.max_goal_queue:
            logger.debug("Goal queue full, not generating new goal")
            return None
        
        # Determine goal type based on drives if not specified
        if goal_type is None:
            goal_type = self._select_goal_type_by_drives()
        
        # Generate goal specifics
        goal_spec = self._create_goal_spec(goal_type, name, description, context)
        
        # Create goal
        goal_id = f"goal_{self.total_goals_generated}_{int(time.time() * 1000)}"
        
        # Estimate difficulty and novelty
        difficulty = self._estimate_difficulty(goal_type, context)
        novelty = self._estimate_novelty(goal_type, context)
        
        # Compute intrinsic value
        if intrinsic_value is None:
            intrinsic_value = self._compute_intrinsic_value(goal_type, novelty, difficulty)
        
        goal = Goal(
            goal_id=goal_id,
            name=goal_spec['name'],
            goal_type=goal_type,
            description=goal_spec['description'],
            parent_goal=parent_goal_id,
            intrinsic_value=intrinsic_value,
            novelty_estimate=novelty,
            difficulty_estimate=difficulty,
            information_gain=novelty * 0.5  # Novelty contributes to information gain
        )
        
        # Link to parent if specified
        if parent_goal_id and parent_goal_id in self.goals:
            self.goals[parent_goal_id].subgoals.append(goal_id)
        
        self.goals[goal_id] = goal
        self.total_goals_generated += 1
        
        # Add to queue
        priority = goal.compute_priority()
        heapq.heappush(self.goal_queue, (-priority, goal_id))
        
        # Record generation
        self.generation_history.append({
            'timestamp': time.time(),
            'goal_id': goal_id,
            'goal_type': goal_type.value,
            'trigger': context.get('trigger', 'autonomous') if context else 'autonomous'
        })
        
        logger.info(f"Generated goal: {goal.name} ({goal_type.value})")
        return goal_id
    
    def _select_goal_type_by_drives(self) -> GoalType:
        """Select goal type based on current drive states"""
        dominant_drive = self.drives.get_dominant_drive()
        
        drive_to_goal = {
            IntrinsicDrive.CURIOSITY: GoalType.CURIOSITY,
            IntrinsicDrive.COMPETENCE: GoalType.MASTERY,
            IntrinsicDrive.AUTONOMY: GoalType.EXPLORATION,
            IntrinsicDrive.RELATEDNESS: GoalType.SOCIAL,
            IntrinsicDrive.CREATIVITY: GoalType.CREATION
        }
        
        return drive_to_goal.get(dominant_drive, GoalType.LEARNING)
    
    def _create_goal_spec(
        self,
        goal_type: GoalType,
        name: Optional[str],
        description: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Create goal specification based on type"""
        if name is None:
            name = f"{goal_type.value}_{self.total_goals_generated}"
        
        if description is None:
            descriptions = {
                GoalType.LEARNING: f"Learn about {context.get('topic', 'new concepts') if context else 'new concepts'}",
                GoalType.EXPLORATION: f"Explore {context.get('domain', 'unknown territory') if context else 'unknown territory'}",
                GoalType.MASTERY: f"Master {context.get('skill', 'existing capabilities') if context else 'existing capabilities'}",
                GoalType.CREATION: f"Create {context.get('output', 'something new') if context else 'something new'}",
                GoalType.OPTIMIZATION: f"Optimize {context.get('target', 'performance') if context else 'performance'}",
                GoalType.SOCIAL: f"Learn from {context.get('agent', 'others') if context else 'others'}",
                GoalType.META: f"Improve {context.get('aspect', 'self') if context else 'self'}",
                GoalType.CURIOSITY: f"Satisfy curiosity about {context.get('question', 'unknown') if context else 'unknown'}"
            }
            description = descriptions.get(goal_type, "Autonomous goal")
        
        return {'name': name, 'description': description}
    
    def _estimate_difficulty(self, goal_type: GoalType, context: Optional[Dict]) -> float:
        """Estimate difficulty of achieving a goal"""
        # Base difficulty by type
        base_difficulty = {
            GoalType.LEARNING: 0.5,
            GoalType.EXPLORATION: 0.6,
            GoalType.MASTERY: 0.7,
            GoalType.CREATION: 0.8,
            GoalType.OPTIMIZATION: 0.6,
            GoalType.SOCIAL: 0.5,
            GoalType.META: 0.9,
            GoalType.CURIOSITY: 0.4
        }
        
        difficulty = base_difficulty.get(goal_type, 0.5)
        
        # Adjust based on context
        if context:
            if 'complexity' in context:
                difficulty = difficulty * 0.5 + context['complexity'] * 0.5
            if 'prior_success_rate' in context:
                # Easier if we've succeeded before
                difficulty *= (1 - context['prior_success_rate'] * 0.3)
        
        return min(1.0, max(0.1, difficulty))
    
    def _estimate_novelty(self, goal_type: GoalType, context: Optional[Dict]) -> float:
        """Estimate novelty of a goal"""
        # Check similar past goals
        similar_count = sum(
            1 for g in self.goals.values()
            if g.goal_type == goal_type
        )
        
        # Novelty decreases with similar past goals
        novelty = np.exp(-similar_count * 0.1)
        
        # Curiosity boost
        novelty = novelty * 0.7 + self.drives.curiosity * 0.3
        
        return min(1.0, max(0.0, novelty))
    
    def _compute_intrinsic_value(
        self,
        goal_type: GoalType,
        novelty: float,
        difficulty: float
    ) -> float:
        """Compute intrinsic value of a goal"""
        # Base value by type
        base_values = {
            GoalType.LEARNING: 0.7,
            GoalType.EXPLORATION: 0.6,
            GoalType.MASTERY: 0.8,
            GoalType.CREATION: 0.75,
            GoalType.OPTIMIZATION: 0.6,
            GoalType.SOCIAL: 0.5,
            GoalType.META: 0.9,
            GoalType.CURIOSITY: 0.8
        }
        
        value = base_values.get(goal_type, 0.5)
        
        # Modulate by novelty (we value novel goals)
        value = value * 0.7 + novelty * 0.3
        
        # Modulate by difficulty (prefer challenging but achievable)
        difficulty_factor = 1.0 - abs(difficulty - 0.6)
        value *= (0.8 + 0.2 * difficulty_factor)
        
        return min(1.0, max(0.0, value))
    
    # ==================== Curiosity-Driven Exploration ====================
    
    def add_exploration_frontier(
        self,
        domain: str,
        novelty_score: float,
        predicted_difficulty: float,
        estimated_reward: float
    ) -> str:
        """
        Add a new exploration frontier.
        
        Args:
            domain: Domain/area to explore
            novelty_score: Estimated novelty (0-1)
            predicted_difficulty: Estimated difficulty (0-1)
            estimated_reward: Estimated reward for exploration
            
        Returns:
            Frontier ID
        """
        frontier_id = f"frontier_{len(self.exploration_frontiers)}_{int(time.time())}"
        
        frontier = ExplorationFrontier(
            frontier_id=frontier_id,
            domain=domain,
            novelty_score=novelty_score,
            predicted_difficulty=predicted_difficulty,
            estimated_reward=estimated_reward
        )
        
        self.exploration_frontiers[frontier_id] = frontier
        
        # Generate curiosity goal if novelty is high
        if novelty_score > self.curiosity_threshold:
            self.generate_goal(
                goal_type=GoalType.CURIOSITY,
                name=f"explore_{domain}",
                description=f"Explore {domain} to satisfy curiosity",
                intrinsic_value=novelty_score,
                context={'frontier_id': frontier_id, 'domain': domain}
            )
        
        return frontier_id
    
    def select_exploration_target(self) -> Optional[str]:
        """
        Select the best exploration target based on curiosity.
        
        Returns:
            Frontier ID or None
        """
        if not self.exploration_frontiers:
            return None
        
        # Score each frontier
        scored = []
        for frontier_id, frontier in self.exploration_frontiers.items():
            value = frontier.compute_exploration_value()
            # Boost by curiosity drive
            value *= (0.5 + 0.5 * self.drives.curiosity)
            scored.append((value, frontier_id))
        
        # Select best
        scored.sort(reverse=True)
        return scored[0][1] if scored else None
    
    def record_exploration(
        self,
        frontier_id: str,
        novelty_found: float,
        success: bool
    ):
        """
        Record results of exploration.
        
        Args:
            frontier_id: Explored frontier
            novelty_found: Actual novelty discovered
            success: Whether exploration was successful
        """
        if frontier_id not in self.exploration_frontiers:
            return
        
        frontier = self.exploration_frontiers[frontier_id]
        frontier.last_explored = time.time()
        frontier.exploration_count += 1
        
        # Update novelty score based on actual findings
        frontier.novelty_score = frontier.novelty_score * self.novelty_decay + novelty_found * (1 - self.novelty_decay)
        
        # Update drives
        self.drives.update(goal_achieved=success, novelty_encountered=novelty_found)
        
        self.total_exploration_steps += 1
        
        logger.debug(f"Recorded exploration of {frontier_id}: novelty={novelty_found:.3f}, success={success}")
    
    # ==================== Goal Management ====================
    
    def activate_next_goal(self) -> Optional[str]:
        """
        Activate the next goal from the queue.
        
        Returns:
            Activated goal ID or None
        """
        # Check capacity
        if len(self.active_goals) >= self.max_active_goals:
            return None
        
        # Find next goal
        while self.goal_queue:
            neg_priority, goal_id = heapq.heappop(self.goal_queue)
            
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                if goal.status == GoalStatus.PROPOSED:
                    # Activate
                    goal.activate()
                    self.active_goals.add(goal_id)
                    logger.info(f"Activated goal: {goal.name}")
                    return goal_id
        
        return None
    
    def update_goal_progress(self, goal_id: str, progress: float, outcome: Optional[Dict] = None):
        """
        Update goal progress.
        
        Args:
            goal_id: Goal to update
            progress: New progress value (0-1)
            outcome: Optional outcome information
        """
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.progress = min(1.0, max(0.0, progress))
        
        # Check completion
        if goal.progress >= 1.0:
            self.complete_goal(goal_id, success=True, outcome=outcome)
        
        logger.debug(f"Updated goal {goal_id} progress: {progress:.2f}")
    
    def complete_goal(self, goal_id: str, success: bool, outcome: Optional[Dict] = None):
        """
        Mark a goal as completed.
        
        Args:
            goal_id: Goal to complete
            success: Whether goal was achieved
            outcome: Outcome information
        """
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        
        if success:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            self.completed_goals.append(goal_id)
            self.total_goals_completed += 1
            
            # Update success rate
            current_rate = self.goal_success_rates[goal.goal_type]
            self.goal_success_rates[goal.goal_type] = (
                current_rate * 0.9 + 0.1
            )
            
            # Update value estimate
            actual_value = outcome.get('value', goal.intrinsic_value) if outcome else goal.intrinsic_value
            self._update_value_estimate(goal_id, actual_value)
            
            # Update drives
            novelty = outcome.get('novelty', 0.5) if outcome else 0.5
            self.drives.update(goal_achieved=True, novelty_encountered=novelty)
            
            logger.info(f"Completed goal: {goal.name}")
        else:
            goal.status = GoalStatus.FAILED
            self.failed_goals.append(goal_id)
            
            # Update success rate
            current_rate = self.goal_success_rates[goal.goal_type]
            self.goal_success_rates[goal.goal_type] = current_rate * 0.9
            
            self.drives.update(goal_achieved=False, novelty_encountered=0.0)
            
            logger.info(f"Failed goal: {goal.name}")
        
        # Remove from active
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        
        # Complete parent if all subgoals done
        if goal.parent_goal and goal.parent_goal in self.goals:
            self._check_parent_completion(goal.parent_goal)
    
    def _check_parent_completion(self, parent_id: str):
        """Check if parent goal should be completed"""
        parent = self.goals[parent_id]
        
        if not parent.subgoals:
            return
        
        subgoal_statuses = [
            self.goals[sg_id].status if sg_id in self.goals else GoalStatus.COMPLETED
            for sg_id in parent.subgoals
        ]
        
        if all(s in [GoalStatus.COMPLETED, GoalStatus.ABANDONED] for s in subgoal_statuses):
            success_rate = sum(
                1 for s in subgoal_statuses if s == GoalStatus.COMPLETED
            ) / len(subgoal_statuses)
            
            self.complete_goal(parent_id, success=success_rate > 0.5)
    
    def _update_value_estimate(self, goal_id: str, actual_value: float):
        """Update value estimate based on actual outcome"""
        if goal_id not in self.goal_value_estimates:
            self.goal_value_estimates[goal_id] = actual_value
        else:
            # Moving average update
            current = self.goal_value_estimates[goal_id]
            self.goal_value_estimates[goal_id] = (
                current * (1 - self.learning_rate) + actual_value * self.learning_rate
            )
    
    def abandon_goal(self, goal_id: str, reason: str = ""):
        """
        Abandon a goal.
        
        Args:
            goal_id: Goal to abandon
            reason: Reason for abandonment
        """
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = GoalStatus.ABANDONED
        
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        
        logger.info(f"Abandoned goal {goal_id}: {reason}")
    
    # ==================== Autonomous Operation ====================
    
    def run_autonomous_cycle(self) -> Dict[str, Any]:
        """
        Run one autonomous goal generation and management cycle.
        
        Returns:
            Cycle results
        """
        results = {
            'goals_generated': 0,
            'goals_activated': 0,
            'exploration_targets': [],
            'drive_states': {}
        }
        
        # Generate goals based on drives
        dominant_drive = self.drives.get_dominant_drive()
        
        if self.drives.curiosity > self.curiosity_threshold:
            # Generate curiosity-driven goals
            goal_id = self.generate_goal(goal_type=GoalType.CURIOSITY)
            if goal_id:
                results['goals_generated'] += 1
        
        if self.drives.competence < 0.5:
            # Generate mastery goals to build competence
            goal_id = self.generate_goal(goal_type=GoalType.MASTERY)
            if goal_id:
                results['goals_generated'] += 1
        
        if self.drives.creativity > 0.7:
            # Generate creation goals
            goal_id = self.generate_goal(goal_type=GoalType.CREATION)
            if goal_id:
                results['goals_generated'] += 1
        
        # Activate goals if capacity available
        while len(self.active_goals) < self.max_active_goals:
            activated = self.activate_next_goal()
            if activated:
                results['goals_activated'] += 1
            else:
                break
        
        # Select exploration targets
        if self.drives.curiosity > 0.5:
            target = self.select_exploration_target()
            if target:
                results['exploration_targets'].append(target)
        
        # Record drive states
        results['drive_states'] = {
            'curiosity': self.drives.curiosity,
            'competence': self.drives.competence,
            'autonomy': self.drives.autonomy,
            'creativity': self.drives.creativity
        }
        
        return results
    
    # ==================== Utility Methods ====================
    
    def get_active_goals(self) -> List[Goal]:
        """Get list of currently active goals"""
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]
    
    def get_goal_hierarchy(self, root_goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get goal hierarchy.
        
        Args:
            root_goal_id: Root goal (all top-level if None)
            
        Returns:
            Hierarchy dictionary
        """
        if root_goal_id:
            if root_goal_id not in self.goals:
                return {}
            return self._build_hierarchy_recursive(root_goal_id)
        
        # Get all top-level goals
        top_level = [
            gid for gid, goal in self.goals.items()
            if goal.parent_goal is None
        ]
        
        return {
            'goals': [self._build_hierarchy_recursive(gid) for gid in top_level]
        }
    
    def _build_hierarchy_recursive(self, goal_id: str) -> Dict[str, Any]:
        """Recursively build hierarchy"""
        goal = self.goals[goal_id]
        
        return {
            'goal_id': goal_id,
            'name': goal.name,
            'type': goal.goal_type.value,
            'status': goal.status.value,
            'progress': goal.progress,
            'intrinsic_value': goal.intrinsic_value,
            'subgoals': [
                self._build_hierarchy_recursive(sgid)
                for sgid in goal.subgoals
                if sgid in self.goals
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get autonomous goal system statistics"""
        return {
            'total_goals_generated': self.total_goals_generated,
            'total_goals_completed': self.total_goals_completed,
            'active_goals_count': len(self.active_goals),
            'queued_goals_count': len(self.goal_queue),
            'completed_goals_count': len(self.completed_goals),
            'failed_goals_count': len(self.failed_goals),
            'exploration_frontiers_count': len(self.exploration_frontiers),
            'total_exploration_steps': self.total_exploration_steps,
            'drive_states': {
                'curiosity': self.drives.curiosity,
                'competence': self.drives.competence,
                'autonomy': self.drives.autonomy,
                'relatedness': self.drives.relatedness,
                'creativity': self.drives.creativity
            },
            'goal_success_rates': {
                gt.value: rate for gt, rate in self.goal_success_rates.items()
            }
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize autonomous goal system"""
        return {
            'max_active_goals': self.max_active_goals,
            'max_goal_queue': self.max_goal_queue,
            'curiosity_threshold': self.curiosity_threshold,
            'enable_meta_goals': self.enable_meta_goals,
            'enable_social_goals': self.enable_social_goals,
            'stats': self.get_stats(),
            'goals': {
                gid: {
                    'goal_id': goal.goal_id,
                    'name': goal.name,
                    'goal_type': goal.goal_type.value,
                    'description': goal.description,
                    'parent_goal': goal.parent_goal,
                    'subgoals': goal.subgoals,
                    'intrinsic_value': goal.intrinsic_value,
                    'status': goal.status.value,
                    'progress': goal.progress,
                    'created_at': goal.created_at,
                    'novelty_estimate': goal.novelty_estimate,
                    'difficulty_estimate': goal.difficulty_estimate
                }
                for gid, goal in self.goals.items()
            },
            'exploration_frontiers': {
                fid: {
                    'frontier_id': f.frontier_id,
                    'domain': f.domain,
                    'novelty_score': f.novelty_score,
                    'predicted_difficulty': f.predicted_difficulty,
                    'estimated_reward': f.estimated_reward,
                    'exploration_count': f.exploration_count
                }
                for fid, f in self.exploration_frontiers.items()
            }
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AutonomousGoalSystem':
        """Deserialize autonomous goal system"""
        system = cls(
            max_active_goals=data['max_active_goals'],
            max_goal_queue=data['max_goal_queue'],
            curiosity_threshold=data['curiosity_threshold'],
            enable_meta_goals=data['enable_meta_goals'],
            enable_social_goals=data['enable_social_goals']
        )
        
        # Restore goals
        for gid, goal_data in data.get('goals', {}).items():
            goal = Goal(
                goal_id=goal_data['goal_id'],
                name=goal_data['name'],
                goal_type=GoalType(goal_data['goal_type']),
                description=goal_data['description'],
                parent_goal=goal_data.get('parent_goal'),
                subgoals=goal_data.get('subgoals', []),
                intrinsic_value=goal_data['intrinsic_value'],
                status=GoalStatus(goal_data['status']),
                progress=goal_data['progress'],
                created_at=goal_data['created_at'],
                novelty_estimate=goal_data.get('novelty_estimate', 0.5),
                difficulty_estimate=goal_data.get('difficulty_estimate', 0.5)
            )
            system.goals[gid] = goal
            
            if goal.status == GoalStatus.ACTIVE:
                system.active_goals.add(gid)
        
        # Restore exploration frontiers
        for fid, frontier_data in data.get('exploration_frontiers', {}).items():
            frontier = ExplorationFrontier(
                frontier_id=frontier_data['frontier_id'],
                domain=frontier_data['domain'],
                novelty_score=frontier_data['novelty_score'],
                predicted_difficulty=frontier_data['predicted_difficulty'],
                estimated_reward=frontier_data['estimated_reward'],
                exploration_count=frontier_data.get('exploration_count', 0)
            )
            system.exploration_frontiers[fid] = frontier
        
        # Restore statistics
        stats = data.get('stats', {})
        system.total_goals_generated = stats.get('total_goals_generated', 0)
        system.total_goals_completed = stats.get('total_goals_completed', 0)
        system.total_exploration_steps = stats.get('total_exploration_steps', 0)
        
        return system
