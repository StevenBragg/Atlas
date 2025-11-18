"""
Goal-Directed Planning System for ATLAS

Implements autonomous goal generation, hierarchical planning, and value-based
decision making.

This is critical for superintelligence development, enabling:
- Autonomous goal setting based on intrinsic drives
- Hierarchical task decomposition
- Model-based planning using world models
- Value learning and utility maximization
- Meta-goals for self-improvement
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import time

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of goals the system can pursue"""
    SURVIVAL = "survival"  # Maintain operational integrity
    LEARNING = "learning"  # Acquire new knowledge
    EXPLORATION = "exploration"  # Discover new patterns
    OPTIMIZATION = "optimization"  # Improve performance
    CREATION = "creation"  # Generate novel outputs
    SOCIAL = "social"  # Interact with other agents
    META = "meta"  # Self-improvement goals


class PlanStatus(Enum):
    """Status of a plan execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class Goal:
    """Represents a goal to be achieved"""
    name: str
    goal_type: GoalType
    priority: float  # 0-1, higher is more important
    value: float  # Expected utility of achieving this goal
    deadline: Optional[float] = None  # Timestamp deadline
    completion_criteria: Optional[Callable] = None
    subgoals: List['Goal'] = field(default_factory=list)
    parent_goal: Optional['Goal'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    status: PlanStatus = PlanStatus.PENDING

    def __hash__(self):
        return hash((self.name, self.creation_time))


@dataclass
class Action:
    """Represents an action that can be taken"""
    name: str
    preconditions: Dict[str, Any]  # Required state
    effects: Dict[str, Any]  # State changes
    cost: float  # Resource cost
    duration: float  # Time required
    success_probability: float = 1.0


@dataclass
class Plan:
    """Represents a plan to achieve a goal"""
    goal: Goal
    actions: List[Action]
    expected_value: float
    expected_cost: float
    confidence: float
    status: PlanStatus = PlanStatus.PENDING


class GoalPlanningSystem:
    """
    Goal-directed planning system with hierarchical decomposition.

    Capabilities:
    - Autonomous goal generation from drives
    - Hierarchical goal decomposition
    - Model-based planning
    - Value-based decision making
    - Plan execution and monitoring
    """

    def __init__(
        self,
        max_goals: int = 100,
        planning_horizon: int = 10,
        enable_meta_goals: bool = True,
        enable_intrinsic_motivation: bool = True,
        exploration_bonus: float = 0.5,
        learning_value: float = 0.8,
        discount_factor: float = 0.95,
        replan_threshold: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize goal planning system.

        Args:
            max_goals: Maximum number of active goals
            planning_horizon: Maximum planning depth
            enable_meta_goals: Whether to generate meta-goals
            enable_intrinsic_motivation: Whether to use intrinsic rewards
            exploration_bonus: Value bonus for exploration
            learning_value: Intrinsic value of learning
            discount_factor: Future reward discount
            replan_threshold: Threshold for triggering replanning
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.max_goals = max_goals
        self.planning_horizon = planning_horizon
        self.enable_meta_goals = enable_meta_goals
        self.enable_intrinsic_motivation = enable_intrinsic_motivation
        self.exploration_bonus = exploration_bonus
        self.learning_value = learning_value
        self.discount_factor = discount_factor
        self.replan_threshold = replan_threshold

        # Goal storage
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.failed_goals: List[Goal] = []

        # Planning
        self.active_plans: Dict[Goal, Plan] = {}
        self.plan_history: List[Plan] = []

        # Value function approximation
        self.state_values: Dict[str, float] = {}
        self.action_values: Dict[Tuple[str, str], float] = {}  # (state, action) -> value

        # Available actions (to be populated)
        self.action_library: List[Action] = []

        # Intrinsic drives
        self.curiosity_level = 1.0
        self.competence_level = 0.0
        self.autonomy_level = 0.5

        # Statistics
        self.total_goals_created = 0
        self.total_plans_executed = 0
        self.total_replans = 0
        self.goal_success_rate = 0.0

        # Initialize basic meta-goals
        if self.enable_meta_goals:
            self._initialize_meta_goals()

        logger.info("Initialized goal planning system")

    def generate_goal(
        self,
        goal_type: Optional[GoalType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Goal]:
        """
        Autonomously generate a new goal based on current state and drives.

        Args:
            goal_type: Optional specific type of goal to generate
            context: Current context information

        Returns:
            Generated goal, or None if no goal needed
        """
        if context is None:
            context = {}

        # Check if we should generate a new goal
        if len(self.active_goals) >= self.max_goals:
            logger.debug("Max goals reached, not generating new goal")
            return None

        # Determine goal type based on drives if not specified
        if goal_type is None:
            goal_type = self._select_goal_type()

        # Generate goal based on type
        if goal_type == GoalType.LEARNING:
            goal = self._generate_learning_goal(context)
        elif goal_type == GoalType.EXPLORATION:
            goal = self._generate_exploration_goal(context)
        elif goal_type == GoalType.OPTIMIZATION:
            goal = self._generate_optimization_goal(context)
        elif goal_type == GoalType.META:
            goal = self._generate_meta_goal(context)
        else:
            # Default goal
            goal = Goal(
                name=f"{goal_type.value}_goal_{self.total_goals_created}",
                goal_type=goal_type,
                priority=0.5,
                value=0.5,
            )

        if goal:
            self.active_goals.append(goal)
            self.total_goals_created += 1
            logger.info(f"Generated new goal: {goal.name} ({goal_type.value})")

        return goal

    def create_plan(
        self,
        goal: Goal,
        current_state: Dict[str, Any],
    ) -> Optional[Plan]:
        """
        Create a plan to achieve the specified goal.

        Args:
            goal: Goal to plan for
            current_state: Current world state

        Returns:
            Generated plan, or None if no plan found
        """
        # Check if goal has subgoals (hierarchical planning)
        if goal.subgoals:
            # Plan for subgoals first
            subplans = []
            for subgoal in goal.subgoals:
                subplan = self.create_plan(subgoal, current_state)
                if subplan:
                    subplans.append(subplan)

            if not subplans:
                logger.warning(f"Could not plan for subgoals of {goal.name}")
                return None

            # Combine subplans
            combined_actions = []
            total_value = 0.0
            total_cost = 0.0
            min_confidence = 1.0

            for subplan in subplans:
                combined_actions.extend(subplan.actions)
                total_value += subplan.expected_value
                total_cost += subplan.expected_cost
                min_confidence = min(min_confidence, subplan.confidence)

            plan = Plan(
                goal=goal,
                actions=combined_actions,
                expected_value=total_value,
                expected_cost=total_cost,
                confidence=min_confidence,
            )

        else:
            # Leaf goal - use direct planning
            plan = self._plan_single_goal(goal, current_state)

        if plan:
            self.active_plans[goal] = plan
            logger.debug(f"Created plan for {goal.name} with {len(plan.actions)} actions")

        return plan

    def _plan_single_goal(
        self,
        goal: Goal,
        current_state: Dict[str, Any],
        max_depth: Optional[int] = None,
    ) -> Optional[Plan]:
        """
        Plan for a single goal using forward search.

        Args:
            goal: Goal to plan for
            current_state: Current state
            max_depth: Maximum search depth

        Returns:
            Plan, or None if no plan found
        """
        if max_depth is None:
            max_depth = self.planning_horizon

        # Use A* search to find action sequence
        # Priority queue: (f_score, g_cost, state, actions_taken)
        start = (0, 0, current_state, [])
        frontier = [start]
        visited = set()

        while frontier and len(frontier) < 1000:  # Limit search
            f_score, g_cost, state, actions = heapq.heappop(frontier)

            # Convert state to hashable
            state_key = self._state_to_key(state)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Check if goal is satisfied
            if self._is_goal_satisfied(goal, state):
                # Found a plan!
                total_value = goal.value
                confidence = 1.0 / (len(actions) + 1)  # Decreases with plan length

                return Plan(
                    goal=goal,
                    actions=actions,
                    expected_value=total_value,
                    expected_cost=g_cost,
                    confidence=confidence,
                )

            # Stop if too deep
            if len(actions) >= max_depth:
                continue

            # Expand with available actions
            for action in self.action_library:
                # Check preconditions
                if self._check_preconditions(action, state):
                    # Apply action
                    new_state = self._apply_action(action, state)
                    new_actions = actions + [action]
                    new_g_cost = g_cost + action.cost

                    # Estimate remaining cost (heuristic)
                    h_cost = self._heuristic(new_state, goal)
                    new_f_score = new_g_cost + h_cost

                    heapq.heappush(
                        frontier,
                        (new_f_score, new_g_cost, new_state, new_actions)
                    )

        logger.warning(f"No plan found for goal {goal.name}")
        return None

    def execute_step(
        self,
        current_state: Dict[str, Any],
    ) -> Optional[Action]:
        """
        Execute one step of the current plan.

        Args:
            current_state: Current world state

        Returns:
            Action to execute, or None
        """
        # Select highest priority goal with a plan
        active_goal = self._select_active_goal()
        if not active_goal or active_goal not in self.active_plans:
            return None

        plan = self.active_plans[active_goal]

        # Check if plan is still valid (replan if needed)
        if self._should_replan(plan, current_state):
            logger.info(f"Replanning for goal {active_goal.name}")
            new_plan = self.create_plan(active_goal, current_state)
            if new_plan:
                self.active_plans[active_goal] = new_plan
                plan = new_plan
                self.total_replans += 1
            else:
                logger.warning(f"Replanning failed for {active_goal.name}")
                return None

        # Execute next action
        if plan.actions:
            action = plan.actions.pop(0)
            plan.status = PlanStatus.IN_PROGRESS
            logger.debug(f"Executing action: {action.name}")
            return action

        # Plan completed
        plan.status = PlanStatus.COMPLETED
        active_goal.status = PlanStatus.COMPLETED
        active_goal.completion_time = time.time()
        self.completed_goals.append(active_goal)
        self.active_goals.remove(active_goal)

        # Update success rate
        total = len(self.completed_goals) + len(self.failed_goals)
        self.goal_success_rate = len(self.completed_goals) / max(1, total)

        logger.info(f"Goal completed: {active_goal.name}")
        return None

    def _select_goal_type(self) -> GoalType:
        """Select goal type based on current drives."""
        # Weight goal types by drive strengths
        weights = {
            GoalType.LEARNING: self.curiosity_level * self.learning_value,
            GoalType.EXPLORATION: self.curiosity_level * self.exploration_bonus,
            GoalType.OPTIMIZATION: self.competence_level * 0.8,
            GoalType.META: self.autonomy_level * 0.6 if self.enable_meta_goals else 0.0,
        }

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weights
            weights = {k: 1.0 / len(weights) for k in weights}

        # Sample
        goal_types = list(weights.keys())
        probabilities = list(weights.values())
        selected = np.random.choice(goal_types, p=probabilities)

        return selected

    def _generate_learning_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate a goal focused on learning."""
        return Goal(
            name=f"learn_pattern_{self.total_goals_created}",
            goal_type=GoalType.LEARNING,
            priority=0.7,
            value=self.learning_value,
            metadata={'context': context},
        )

    def _generate_exploration_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate a goal focused on exploration."""
        return Goal(
            name=f"explore_state_{self.total_goals_created}",
            goal_type=GoalType.EXPLORATION,
            priority=0.6,
            value=self.exploration_bonus,
            metadata={'context': context},
        )

    def _generate_optimization_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate a goal focused on optimization."""
        return Goal(
            name=f"optimize_performance_{self.total_goals_created}",
            goal_type=GoalType.OPTIMIZATION,
            priority=0.8,
            value=0.9,
            metadata={'context': context},
        )

    def _generate_meta_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate a meta-goal for self-improvement."""
        return Goal(
            name=f"improve_capability_{self.total_goals_created}",
            goal_type=GoalType.META,
            priority=0.9,
            value=1.0,
            metadata={'context': context, 'meta': True},
        )

    def _initialize_meta_goals(self) -> None:
        """Initialize fundamental meta-goals."""
        meta_goals = [
            Goal(
                name="continuous_learning",
                goal_type=GoalType.META,
                priority=0.95,
                value=1.0,
                metadata={'perpetual': True},
            ),
            Goal(
                name="self_improvement",
                goal_type=GoalType.META,
                priority=1.0,
                value=1.0,
                metadata={'perpetual': True},
            ),
        ]

        self.active_goals.extend(meta_goals)

    def _select_active_goal(self) -> Optional[Goal]:
        """Select highest priority active goal."""
        if not self.active_goals:
            return None

        # Filter goals with plans
        goals_with_plans = [g for g in self.active_goals if g in self.active_plans]
        if not goals_with_plans:
            return None

        # Select highest priority
        return max(goals_with_plans, key=lambda g: g.priority)

    def _should_replan(self, plan: Plan, state: Dict[str, Any]) -> bool:
        """Determine if replanning is needed."""
        # Replan if confidence is low
        if plan.confidence < self.replan_threshold:
            return True

        # Replan if next action preconditions not met
        if plan.actions:
            next_action = plan.actions[0]
            if not self._check_preconditions(next_action, state):
                return True

        return False

    def _is_goal_satisfied(self, goal: Goal, state: Dict[str, Any]) -> bool:
        """Check if goal is satisfied in the given state."""
        if goal.completion_criteria:
            return goal.completion_criteria(state)

        # Default: check if goal name matches state
        return goal.name in state and state[goal.name] == True

    def _check_preconditions(self, action: Action, state: Dict[str, Any]) -> bool:
        """Check if action preconditions are met."""
        for key, required_value in action.preconditions.items():
            if key not in state or state[key] != required_value:
                return False
        return True

    def _apply_action(self, action: Action, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action effects to state."""
        new_state = state.copy()
        new_state.update(action.effects)
        return new_state

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to hashable key."""
        items = sorted(state.items())
        return str(items)

    def _heuristic(self, state: Dict[str, Any], goal: Goal) -> float:
        """Estimate cost to achieve goal from state."""
        # Simple heuristic: number of unsatisfied goal conditions
        if goal.completion_criteria:
            return 0.0 if goal.completion_criteria(state) else 1.0

        # Count differences
        differences = 0
        if goal.name not in state or not state[goal.name]:
            differences += 1

        return float(differences)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about goal planning."""
        return {
            'active_goals': len(self.active_goals),
            'completed_goals': len(self.completed_goals),
            'failed_goals': len(self.failed_goals),
            'active_plans': len(self.active_plans),
            'total_goals_created': self.total_goals_created,
            'total_plans_executed': self.total_plans_executed,
            'total_replans': self.total_replans,
            'goal_success_rate': float(self.goal_success_rate),
            'curiosity_level': float(self.curiosity_level),
            'competence_level': float(self.competence_level),
            'autonomy_level': float(self.autonomy_level),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the goal planning system."""
        return {
            'active_goals': [
                {'name': g.name, 'type': g.goal_type.value, 'priority': g.priority}
                for g in self.active_goals
            ],
            'completed_goals': len(self.completed_goals),
            'failed_goals': len(self.failed_goals),
            'drives': {
                'curiosity': self.curiosity_level,
                'competence': self.competence_level,
                'autonomy': self.autonomy_level,
            },
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'GoalPlanningSystem':
        """Create a goal planning system from serialized data."""
        instance = cls()

        # Restore drives
        if 'drives' in data:
            instance.curiosity_level = data['drives'].get('curiosity', 1.0)
            instance.competence_level = data['drives'].get('competence', 0.0)
            instance.autonomy_level = data['drives'].get('autonomy', 0.5)

        return instance
