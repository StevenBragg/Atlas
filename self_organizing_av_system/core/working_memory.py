"""
Working Memory and Attention System for ATLAS

Implements a Global Workspace Theory inspired working memory that enables:
- Limited-capacity active information storage
- Selective attention for focusing computational resources
- Information broadcasting to all cognitive modules
- Task switching and cognitive control
- Conscious-like processing through global availability

The global workspace acts as a "blackboard" where information becomes
globally available to all cognitive processes, enabling coordination
and deliberate reasoning.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import heapq

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of attention"""
    BOTTOM_UP = "bottom_up"  # Stimulus-driven (salience)
    TOP_DOWN = "top_down"    # Goal-driven (relevance)
    EXECUTIVE = "executive"   # Control-driven


class WorkspaceSlotType(Enum):
    """Types of workspace slots"""
    SENSORY = "sensory"        # Sensory input
    SEMANTIC = "semantic"      # Conceptual information
    EPISODIC = "episodic"      # Memory retrieval
    GOAL = "goal"              # Current goals
    ACTION = "action"          # Planned actions
    LANGUAGE = "language"      # Linguistic content
    REASONING = "reasoning"    # Reasoning intermediate


@dataclass
class WorkspaceItem:
    """An item in working memory"""
    item_id: str
    content: np.ndarray
    slot_type: WorkspaceSlotType
    source: str  # Which module created this
    salience: float  # Bottom-up attention
    relevance: float  # Top-down attention (goal relevance)
    activation: float  # Current activation level
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.1
    consolidation_count: int = 0  # How many times refreshed

    def compute_priority(self) -> float:
        """Compute priority for workspace access."""
        # Combine salience, relevance, and recency
        recency = np.exp(-0.1 * (time.time() - self.timestamp))
        return (0.3 * self.salience + 0.4 * self.relevance +
                0.2 * self.activation + 0.1 * recency)


@dataclass
class AttentionFocus:
    """Current focus of attention"""
    target_ids: List[str]
    attention_type: AttentionType
    intensity: float
    duration: float
    start_time: float


class AttentionController:
    """
    Controls attention allocation across working memory.

    Implements both bottom-up (salience-driven) and top-down
    (goal-driven) attention mechanisms.
    """

    def __init__(
        self,
        num_attention_heads: int = 4,
        attention_capacity: float = 1.0,
        switch_cost: float = 0.1,
    ):
        """
        Initialize attention controller.

        Args:
            num_attention_heads: Number of parallel attention foci
            attention_capacity: Total attention capacity
            switch_cost: Cost of switching attention
        """
        self.num_attention_heads = num_attention_heads
        self.attention_capacity = attention_capacity
        self.switch_cost = switch_cost

        # Current attention allocations
        self.current_foci: List[AttentionFocus] = []

        # Attention weights per item
        self.attention_weights: Dict[str, float] = {}

        # Goal-based attention biases
        self.goal_biases: Dict[str, float] = {}

        # Salience map (bottom-up)
        self.salience_map: Dict[str, float] = {}

        # Inhibition of return (prevent re-attending same item)
        self.inhibited_items: Dict[str, float] = {}

        # Statistics
        self.total_switches = 0
        self.total_updates = 0

    def compute_salience(
        self,
        item: WorkspaceItem,
        context_items: List[WorkspaceItem],
    ) -> float:
        """
        Compute bottom-up salience of an item.

        Salience is based on:
        - Novelty (difference from context)
        - Intensity (magnitude of content)
        - Surprise (deviation from expectations)
        """
        # Intensity: magnitude of content
        intensity = np.linalg.norm(item.content)

        # Novelty: difference from other items
        if len(context_items) > 0:
            similarities = []
            for other in context_items:
                if other.item_id != item.item_id:
                    sim = np.dot(item.content, other.content) / (
                        np.linalg.norm(item.content) * np.linalg.norm(other.content) + 1e-8
                    )
                    similarities.append(sim)
            novelty = 1.0 - np.mean(similarities) if similarities else 1.0
        else:
            novelty = 1.0

        # Combine factors
        salience = 0.5 * novelty + 0.3 * min(1.0, intensity / 10.0) + 0.2 * item.salience

        return float(np.clip(salience, 0, 1))

    def compute_relevance(
        self,
        item: WorkspaceItem,
        current_goals: List[np.ndarray],
    ) -> float:
        """
        Compute top-down relevance of an item to current goals.
        """
        if len(current_goals) == 0:
            return 0.5  # Neutral if no goals

        # Compute similarity to each goal
        max_relevance = 0.0
        for goal in current_goals:
            similarity = np.dot(item.content, goal) / (
                np.linalg.norm(item.content) * np.linalg.norm(goal) + 1e-8
            )
            max_relevance = max(max_relevance, similarity)

        # Apply goal bias if exists
        if item.item_id in self.goal_biases:
            max_relevance = 0.7 * max_relevance + 0.3 * self.goal_biases[item.item_id]

        return float(np.clip(max_relevance, 0, 1))

    def update_attention(
        self,
        items: List[WorkspaceItem],
        goals: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Update attention allocation across items.

        Args:
            items: All items in workspace
            goals: Current goal representations

        Returns:
            Attention weights for each item
        """
        if len(items) == 0:
            return {}

        # Compute salience and relevance for each item
        priorities = []
        for item in items:
            salience = self.compute_salience(item, items)
            relevance = self.compute_relevance(item, goals)

            # Check inhibition of return
            inhibition = self.inhibited_items.get(item.item_id, 0.0)

            # Combined priority
            priority = (0.4 * salience + 0.5 * relevance + 0.1 * item.activation) * (1 - inhibition)
            priorities.append((item.item_id, priority))

            self.salience_map[item.item_id] = salience

        # Softmax to get attention weights
        priorities.sort(key=lambda x: x[1], reverse=True)

        total_priority = sum(p for _, p in priorities) + 1e-8
        self.attention_weights = {
            item_id: priority / total_priority
            for item_id, priority in priorities
        }

        # Update inhibition of return (decay)
        for item_id in list(self.inhibited_items.keys()):
            self.inhibited_items[item_id] *= 0.9
            if self.inhibited_items[item_id] < 0.01:
                del self.inhibited_items[item_id]

        self.total_updates += 1

        return self.attention_weights

    def focus_on(
        self,
        item_ids: List[str],
        attention_type: AttentionType = AttentionType.TOP_DOWN,
        duration: float = 1.0,
    ) -> None:
        """Explicitly focus attention on specific items."""
        # Check if this is a switch
        current_ids = set()
        for focus in self.current_foci:
            current_ids.update(focus.target_ids)

        new_ids = set(item_ids)
        if current_ids != new_ids:
            self.total_switches += 1

        # Create new focus
        focus = AttentionFocus(
            target_ids=item_ids,
            attention_type=attention_type,
            intensity=1.0,
            duration=duration,
            start_time=time.time(),
        )

        # Replace oldest focus if at capacity
        if len(self.current_foci) >= self.num_attention_heads:
            self.current_foci.pop(0)

        self.current_foci.append(focus)

        # Boost attention weights for focused items
        for item_id in item_ids:
            if item_id in self.attention_weights:
                self.attention_weights[item_id] = min(
                    1.0, self.attention_weights[item_id] * 2.0
                )

    def set_goal_bias(self, item_id: str, bias: float) -> None:
        """Set a goal-based attention bias for an item."""
        self.goal_biases[item_id] = np.clip(bias, 0, 1)

    def inhibit(self, item_id: str, strength: float = 0.5) -> None:
        """Apply inhibition of return to an item."""
        self.inhibited_items[item_id] = min(1.0, strength)

    def get_most_attended(self, top_k: int = 3) -> List[str]:
        """Get the most attended item IDs."""
        sorted_items = sorted(
            self.attention_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [item_id for item_id, _ in sorted_items[:top_k]]

    def get_stats(self) -> Dict[str, Any]:
        """Get attention statistics."""
        return {
            'num_attention_heads': self.num_attention_heads,
            'current_foci_count': len(self.current_foci),
            'total_switches': self.total_switches,
            'total_updates': self.total_updates,
            'items_with_weights': len(self.attention_weights),
            'items_inhibited': len(self.inhibited_items),
        }


class WorkingMemory:
    """
    Working memory system implementing a Global Workspace.

    Features:
    - Limited capacity storage for active information
    - Attention-gated access and updates
    - Information broadcasting to all modules
    - Decay and refresh mechanisms
    - Integration with other cognitive systems
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magical number
        content_dim: int = 128,
        decay_rate: float = 0.05,
        refresh_threshold: float = 0.3,
        broadcast_threshold: float = 0.5,
    ):
        """
        Initialize working memory.

        Args:
            capacity: Maximum number of items (slots)
            content_dim: Dimension of content vectors
            decay_rate: Rate of activation decay
            refresh_threshold: Threshold for automatic refresh
            broadcast_threshold: Activation needed for global broadcast
        """
        self.capacity = capacity
        self.content_dim = content_dim
        self.decay_rate = decay_rate
        self.refresh_threshold = refresh_threshold
        self.broadcast_threshold = broadcast_threshold

        # Working memory slots
        self.slots: Dict[str, WorkspaceItem] = {}

        # Attention controller
        self.attention = AttentionController()

        # Current goals (for relevance computation)
        self.current_goals: List[np.ndarray] = []

        # Broadcast listeners (callbacks when items are broadcast)
        self.broadcast_listeners: List[Callable[[WorkspaceItem], None]] = []

        # History of broadcast items
        self.broadcast_history: deque = deque(maxlen=100)

        # Central executive state
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []

        # Statistics
        self.total_items_added = 0
        self.total_items_evicted = 0
        self.total_broadcasts = 0
        self.total_refreshes = 0

        # Item counter for unique IDs
        self._item_counter = 0

        logger.info(f"WorkingMemory initialized: capacity={capacity}, dim={content_dim}")

    def _generate_item_id(self) -> str:
        """Generate a unique item ID."""
        self._item_counter += 1
        return f"wm_{self._item_counter}_{int(time.time() * 1000) % 10000}"

    def add(
        self,
        content: np.ndarray,
        slot_type: WorkspaceSlotType,
        source: str,
        salience: float = 0.5,
        relevance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """
        Add an item to working memory.

        If at capacity, the lowest priority item is evicted.

        Args:
            content: Content vector
            slot_type: Type of content
            source: Source module
            salience: Initial salience
            relevance: Initial relevance
            metadata: Optional metadata
            item_id: Optional specific ID

        Returns:
            ID of added item
        """
        # Ensure content is correct dimension
        if len(content) != self.content_dim:
            if len(content) > self.content_dim:
                content = content[:self.content_dim]
            else:
                padded = np.zeros(self.content_dim)
                padded[:len(content)] = content
                content = padded

        # Normalize
        content = content / (np.linalg.norm(content) + 1e-8)

        if item_id is None:
            item_id = self._generate_item_id()

        item = WorkspaceItem(
            item_id=item_id,
            content=content,
            slot_type=slot_type,
            source=source,
            salience=salience,
            relevance=relevance,
            activation=1.0,  # Start fully active
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Check capacity
        if len(self.slots) >= self.capacity and item_id not in self.slots:
            self._evict_lowest_priority()

        self.slots[item_id] = item
        self.total_items_added += 1

        # Update attention
        self.attention.update_attention(list(self.slots.values()), self.current_goals)

        # Check for immediate broadcast
        if item.compute_priority() >= self.broadcast_threshold:
            self._broadcast(item)

        logger.debug(f"Added item {item_id} to working memory ({len(self.slots)}/{self.capacity})")

        return item_id

    def _evict_lowest_priority(self) -> Optional[str]:
        """Evict the lowest priority item."""
        if len(self.slots) == 0:
            return None

        # Find lowest priority
        lowest_id = None
        lowest_priority = float('inf')

        for item_id, item in self.slots.items():
            priority = item.compute_priority()
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_id = item_id

        if lowest_id is not None:
            del self.slots[lowest_id]
            self.total_items_evicted += 1
            logger.debug(f"Evicted item {lowest_id} from working memory")

        return lowest_id

    def get(self, item_id: str) -> Optional[WorkspaceItem]:
        """Get an item by ID."""
        return self.slots.get(item_id)

    def refresh(self, item_id: str) -> bool:
        """
        Refresh an item, preventing decay.

        Returns True if item exists and was refreshed.
        """
        if item_id not in self.slots:
            return False

        item = self.slots[item_id]
        item.activation = 1.0
        item.timestamp = time.time()
        item.consolidation_count += 1
        self.total_refreshes += 1

        return True

    def update_content(
        self,
        item_id: str,
        new_content: np.ndarray,
    ) -> bool:
        """Update the content of an item."""
        if item_id not in self.slots:
            return False

        # Normalize new content
        if len(new_content) != self.content_dim:
            if len(new_content) > self.content_dim:
                new_content = new_content[:self.content_dim]
            else:
                padded = np.zeros(self.content_dim)
                padded[:len(new_content)] = new_content
                new_content = padded

        new_content = new_content / (np.linalg.norm(new_content) + 1e-8)

        self.slots[item_id].content = new_content
        self.slots[item_id].timestamp = time.time()

        return True

    def remove(self, item_id: str) -> bool:
        """Remove an item from working memory."""
        if item_id in self.slots:
            del self.slots[item_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from working memory."""
        self.slots.clear()
        self.attention.attention_weights.clear()

    def step(self) -> List[WorkspaceItem]:
        """
        Perform one time step of working memory dynamics.

        - Applies decay to all items
        - Removes items below threshold
        - Updates attention
        - Broadcasts high-activation items

        Returns:
            List of items that were broadcast
        """
        broadcast_items = []
        items_to_remove = []

        # Decay and check thresholds
        for item_id, item in self.slots.items():
            # Apply decay
            item.activation *= (1.0 - self.decay_rate)

            # Get attention weight
            attention = self.attention.attention_weights.get(item_id, 0.0)

            # Attention can counteract decay
            item.activation += attention * 0.1
            item.activation = min(1.0, item.activation)

            # Check for removal
            if item.activation < 0.1:
                items_to_remove.append(item_id)
            # Check for broadcast
            elif item.activation >= self.broadcast_threshold:
                broadcast_items.append(item)

        # Remove decayed items
        for item_id in items_to_remove:
            del self.slots[item_id]
            self.total_items_evicted += 1

        # Update attention
        self.attention.update_attention(list(self.slots.values()), self.current_goals)

        # Broadcast high-activation items
        for item in broadcast_items:
            self._broadcast(item)

        return broadcast_items

    def _broadcast(self, item: WorkspaceItem) -> None:
        """Broadcast an item to all listeners (global workspace)."""
        self.total_broadcasts += 1
        self.broadcast_history.append((time.time(), item.item_id, item.slot_type.value))

        # Notify listeners
        for listener in self.broadcast_listeners:
            try:
                listener(item)
            except Exception as e:
                logger.warning(f"Broadcast listener error: {e}")

        logger.debug(f"Broadcast item {item.item_id} ({item.slot_type.value})")

    def register_broadcast_listener(
        self,
        listener: Callable[[WorkspaceItem], None],
    ) -> None:
        """Register a callback for broadcast events."""
        self.broadcast_listeners.append(listener)

    def set_goal(self, goal: np.ndarray) -> None:
        """Set the current goal for relevance computation."""
        # Normalize goal
        if len(goal) != self.content_dim:
            if len(goal) > self.content_dim:
                goal = goal[:self.content_dim]
            else:
                padded = np.zeros(self.content_dim)
                padded[:len(goal)] = goal
                goal = padded

        goal = goal / (np.linalg.norm(goal) + 1e-8)
        self.current_goals = [goal]

    def add_goal(self, goal: np.ndarray) -> None:
        """Add an additional goal."""
        if len(goal) != self.content_dim:
            if len(goal) > self.content_dim:
                goal = goal[:self.content_dim]
            else:
                padded = np.zeros(self.content_dim)
                padded[:len(goal)] = goal
                goal = padded

        goal = goal / (np.linalg.norm(goal) + 1e-8)
        self.current_goals.append(goal)

    def clear_goals(self) -> None:
        """Clear all goals."""
        self.current_goals = []

    def push_task(self, task: str) -> None:
        """Push a task onto the task stack."""
        if self.current_task is not None:
            self.task_stack.append(self.current_task)
        self.current_task = task

    def pop_task(self) -> Optional[str]:
        """Pop a task from the stack and restore previous."""
        completed = self.current_task
        if self.task_stack:
            self.current_task = self.task_stack.pop()
        else:
            self.current_task = None
        return completed

    def query_by_type(
        self,
        slot_type: WorkspaceSlotType,
    ) -> List[WorkspaceItem]:
        """Get all items of a specific type."""
        return [
            item for item in self.slots.values()
            if item.slot_type == slot_type
        ]

    def query_by_similarity(
        self,
        query: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[WorkspaceItem, float]]:
        """Find items most similar to a query."""
        # Normalize query
        if len(query) != self.content_dim:
            if len(query) > self.content_dim:
                query = query[:self.content_dim]
            else:
                padded = np.zeros(self.content_dim)
                padded[:len(query)] = query
                query = padded

        query = query / (np.linalg.norm(query) + 1e-8)

        # Compute similarities
        similarities = []
        for item in self.slots.values():
            sim = float(np.dot(query, item.content))
            similarities.append((item, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_most_active(self, top_k: int = 3) -> List[WorkspaceItem]:
        """Get the most active items."""
        items = sorted(
            self.slots.values(),
            key=lambda x: x.activation,
            reverse=True
        )
        return items[:top_k]

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current working memory state."""
        type_counts = {}
        for item in self.slots.values():
            t = item.slot_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        most_active = self.get_most_active(3)
        most_attended = self.attention.get_most_attended(3)

        return {
            'capacity': self.capacity,
            'current_size': len(self.slots),
            'utilization': len(self.slots) / self.capacity,
            'items_by_type': type_counts,
            'current_task': self.current_task,
            'task_stack_depth': len(self.task_stack),
            'num_goals': len(self.current_goals),
            'most_active_ids': [item.item_id for item in most_active],
            'most_attended_ids': most_attended,
            'avg_activation': np.mean([i.activation for i in self.slots.values()]) if self.slots else 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        return {
            'capacity': self.capacity,
            'content_dim': self.content_dim,
            'total_items_added': self.total_items_added,
            'total_items_evicted': self.total_items_evicted,
            'total_broadcasts': self.total_broadcasts,
            'total_refreshes': self.total_refreshes,
            'current_size': len(self.slots),
            'attention_stats': self.attention.get_stats(),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize working memory state."""
        slots_data = {}
        for item_id, item in self.slots.items():
            slots_data[item_id] = {
                'content': item.content.tolist(),
                'slot_type': item.slot_type.value,
                'source': item.source,
                'salience': item.salience,
                'relevance': item.relevance,
                'activation': item.activation,
                'timestamp': item.timestamp,
                'metadata': item.metadata,
            }

        return {
            'capacity': self.capacity,
            'content_dim': self.content_dim,
            'slots': slots_data,
            'current_task': self.current_task,
            'task_stack': self.task_stack,
            'goals': [g.tolist() for g in self.current_goals],
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'WorkingMemory':
        """Deserialize working memory from saved state."""
        instance = cls(
            capacity=data['capacity'],
            content_dim=data['content_dim'],
        )

        for item_id, item_data in data.get('slots', {}).items():
            instance.add(
                content=np.array(item_data['content']),
                slot_type=WorkspaceSlotType(item_data['slot_type']),
                source=item_data['source'],
                salience=item_data['salience'],
                relevance=item_data['relevance'],
                metadata=item_data.get('metadata', {}),
                item_id=item_id,
            )

        instance.current_task = data.get('current_task')
        instance.task_stack = data.get('task_stack', [])

        for goal_list in data.get('goals', []):
            instance.add_goal(np.array(goal_list))

        return instance


class CognitiveController:
    """
    Executive function controller that coordinates working memory,
    attention, and task management.

    Implements:
    - Task switching
    - Inhibitory control
    - Goal maintenance
    - Cognitive flexibility
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        switch_cost_ms: float = 200,
        inhibition_strength: float = 0.5,
    ):
        """
        Initialize cognitive controller.

        Args:
            working_memory: Working memory system to control
            switch_cost_ms: Time cost for task switching
            inhibition_strength: Strength of inhibitory control
        """
        self.wm = working_memory
        self.switch_cost_ms = switch_cost_ms
        self.inhibition_strength = inhibition_strength

        # Active inhibitions
        self.inhibitions: Dict[str, float] = {}

        # Task performance history
        self.task_history: List[Dict[str, Any]] = []

        # Conflict monitoring
        self.conflict_level: float = 0.0

        # Cognitive load estimate
        self.cognitive_load: float = 0.0

    def switch_task(self, new_task: str) -> float:
        """
        Switch to a new task.

        Returns the switch cost in simulated milliseconds.
        """
        old_task = self.wm.current_task

        # Record old task
        if old_task is not None:
            self.task_history.append({
                'task': old_task,
                'end_time': time.time(),
                'wm_items': len(self.wm.slots),
            })

        # Push new task
        self.wm.push_task(new_task)

        # Apply switch cost (increase conflict)
        self.conflict_level = min(1.0, self.conflict_level + 0.2)

        # Clear some working memory (task switch clears irrelevant info)
        self._filter_task_irrelevant()

        return self.switch_cost_ms

    def _filter_task_irrelevant(self) -> None:
        """Remove task-irrelevant items from working memory."""
        # In a full implementation, this would use task-relevance
        # For now, we reduce activation of non-goal items
        for item in self.wm.slots.values():
            if item.slot_type not in [WorkspaceSlotType.GOAL, WorkspaceSlotType.ACTION]:
                item.activation *= 0.7

    def inhibit_response(self, item_id: str) -> bool:
        """
        Inhibit a response/item.

        Used for response inhibition and cognitive control.
        """
        if item_id in self.wm.slots:
            self.inhibitions[item_id] = self.inhibition_strength
            self.wm.attention.inhibit(item_id, self.inhibition_strength)
            return True
        return False

    def release_inhibition(self, item_id: str) -> None:
        """Release inhibition on an item."""
        if item_id in self.inhibitions:
            del self.inhibitions[item_id]

    def monitor_conflict(self) -> float:
        """
        Monitor for response conflict.

        Returns current conflict level.
        """
        # Conflict arises when multiple high-activation items compete
        active_items = self.wm.get_most_active(5)

        if len(active_items) < 2:
            self.conflict_level *= 0.9  # Decay
            return self.conflict_level

        # Compute activation variance (high variance = conflict)
        activations = [item.activation for item in active_items]
        activation_var = np.var(activations)

        # Low variance with high mean = conflict (multiple strong competitors)
        activation_mean = np.mean(activations)
        if activation_var < 0.1 and activation_mean > 0.5:
            self.conflict_level = min(1.0, self.conflict_level + 0.1)
        else:
            self.conflict_level *= 0.95

        return self.conflict_level

    def estimate_cognitive_load(self) -> float:
        """Estimate current cognitive load."""
        # Factors: WM utilization, number of goals, conflict level
        wm_load = len(self.wm.slots) / self.wm.capacity
        goal_load = min(1.0, len(self.wm.current_goals) / 3)
        task_load = min(1.0, len(self.wm.task_stack) / 3)

        self.cognitive_load = (
            0.4 * wm_load +
            0.2 * goal_load +
            0.2 * task_load +
            0.2 * self.conflict_level
        )

        return self.cognitive_load

    def should_switch(self, new_task_priority: float) -> bool:
        """Decide whether to switch to a new task."""
        # Consider current load and switch cost
        load = self.estimate_cognitive_load()

        # Higher load = less willing to switch
        switch_threshold = 0.6 - 0.3 * load

        return new_task_priority > switch_threshold

    def step(self) -> Dict[str, float]:
        """
        Perform one step of cognitive control.

        Returns current control metrics.
        """
        # Update conflict monitoring
        conflict = self.monitor_conflict()

        # Update cognitive load
        load = self.estimate_cognitive_load()

        # Decay inhibitions
        for item_id in list(self.inhibitions.keys()):
            self.inhibitions[item_id] *= 0.95
            if self.inhibitions[item_id] < 0.01:
                del self.inhibitions[item_id]

        # Update working memory
        self.wm.step()

        return {
            'conflict_level': conflict,
            'cognitive_load': load,
            'inhibition_count': len(self.inhibitions),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            'conflict_level': self.conflict_level,
            'cognitive_load': self.cognitive_load,
            'active_inhibitions': len(self.inhibitions),
            'tasks_completed': len(self.task_history),
            'current_task': self.wm.current_task,
            'wm_stats': self.wm.get_stats(),
        }
