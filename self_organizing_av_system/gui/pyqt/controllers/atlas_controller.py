"""
Atlas Controller - Unified Brain for the GUI

This controller integrates all Atlas core systems and provides a single
interface for both Curriculum Learning and Free Play tabs.

Atlas exists as ONE instance shared between both areas, allowing
knowledge transfer between structured and exploratory learning.

Now with persistent memory storage for lifelong learning.
"""

import numpy as np
import logging
import threading
import time
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

from self_organizing_av_system.core.backend import xp, to_cpu, to_gpu, HAS_GPU, get_backend_info
from self_organizing_av_system.core.challenge_learner import ChallengeLearner
from self_organizing_av_system.core.challenge import Challenge, LearningResult, Modality
from self_organizing_av_system.core.curriculum_system import CurriculumSystem, CurriculumLevel, ChallengeResult
from self_organizing_av_system.core.knowledge_base import KnowledgeBase
from self_organizing_av_system.core.text_response import TextResponseLearner, ResponseContext, GeneratedResponse
from self_organizing_av_system.core.creative_canvas_controller import CreativeCanvasController
from self_organizing_av_system.core.progress_tracker import ProgressTracker

# Import database stores
try:
    from self_organizing_av_system.database import VectorStore, GraphStore, NetworkStore
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    VectorStore = None
    GraphStore = None
    NetworkStore = None

logger = logging.getLogger(__name__)


class AtlasController:
    """
    Unified Atlas Brain Controller.

    Integrates:
    - ChallengeLearner: For learning tasks
    - CurriculumSystem: For structured school-like learning
    - KnowledgeBase: Shared episodic + semantic memory
    - TextResponseLearner: For generating chat responses
    - CreativeCanvasController: For 512x512 creative output
    - Visual/Audio processors: For processing AV input

    Single instance shared between Curriculum and Free Play.
    """

    def __init__(self, state_dim: int = 128, data_dir: str = "atlas_data", enable_persistence: bool = True):
        """
        Initialize the unified Atlas controller.

        Args:
            state_dim: Dimension of neural state vectors
            data_dir: Directory for persistent data storage
            enable_persistence: Whether to enable persistent memory storage
        """
        self.state_dim = state_dim
        self.data_dir = Path(data_dir)
        self.enable_persistence = enable_persistence and DATABASE_AVAILABLE

        # Initialize core systems
        logger.info("Initializing Atlas core systems...")

        # Initialize database stores if persistence is enabled
        self.vector_store = None
        self.graph_store = None
        self.network_store = None

        if self.enable_persistence:
            self._init_database_stores()

        # Challenge learning system
        self.challenge_learner = ChallengeLearner(
            state_dim=state_dim,
            learning_rate=0.01,
            batch_size=32,
            verbose=False,
        )

        # Curriculum system (school)
        self.curriculum = CurriculumSystem(state_dim=state_dim)

        # Shared knowledge base (with persistent stores)
        self.knowledge_base = KnowledgeBase(
            state_dim=state_dim,
            max_episodes=50000,
            max_concepts=100000,
            enable_consolidation=True,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )

        # Text response learner
        self.text_learner = TextResponseLearner(
            state_dim=state_dim,
            learning_rate=0.01,
        )

        # Creative canvas controller
        self.canvas_controller = CreativeCanvasController(
            state_dim=state_dim,
            creativity_rate=0.1,
        )

        # Load network weights from persistent store
        if self.enable_persistence:
            self._load_network_weights()

        # Internal state (shared across all systems)
        self.internal_state = np.zeros(state_dim, dtype=np.float32)

        # Processing state
        self.is_learning = False
        self.current_challenge: Optional[Challenge] = None
        self.current_accuracy = 0.0
        self.current_iterations = 0
        self.current_strategy = "Hebbian"

        # AV processing state
        self.av_processing_active = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_audio: Optional[np.ndarray] = None

        # Callbacks
        self._progress_callbacks: List[Callable] = []
        self._network_callbacks: List[Callable] = []
        self._learning_complete_callbacks: List[Callable] = []

        # Auto-learning state
        self._auto_learning_active = False
        self._auto_learning_thread: Optional[threading.Thread] = None
        self._stop_auto_learning = threading.Event()

        # Thread-safe cached network state for UI visualization
        # Updated from learning thread, read by UI thread
        self._network_state_lock = threading.Lock()
        self._cached_network_state: Dict[str, Any] = {
            "num_layers": 2,
            "layers": [],
            "total_neurons": 0,
            "total_connections": 0,
            "neurons_added": 0,
            "neurons_pruned": 0,
            "recent_changes": [],
        }
        self._network_state_update_counter = 0

        # Load any saved state from previous sessions
        self.load_state()

        # Initialize the network state cache so UI has data immediately
        self._update_network_state_cache()

        # Log GPU status
        if HAS_GPU:
            backend_info = get_backend_info()
            logger.info(
                f"AtlasController initialized - GPU ENABLED: {backend_info.get('device_name', 'Unknown')} "
                f"({backend_info.get('device_memory_gb', 0):.1f} GB)"
            )
        else:
            logger.info("AtlasController initialized - CPU only (no GPU detected)")

    def register_progress_callback(self, callback: Callable) -> None:
        """Register a callback for learning progress updates."""
        self._progress_callbacks.append(callback)
        logger.info(f"Progress callback registered, total={len(self._progress_callbacks)}")

    def register_network_callback(self, callback: Callable) -> None:
        """Register a callback for network structure updates."""
        self._network_callbacks.append(callback)

    def register_learning_complete_callback(self, callback: Callable) -> None:
        """Register a callback for when a challenge completes."""
        self._learning_complete_callbacks.append(callback)

    def _notify_progress(self, metrics: Dict[str, Any]) -> None:
        """Notify all progress callbacks."""
        if not self._progress_callbacks:
            logger.warning("No progress callbacks registered!")
            return

        for callback in self._progress_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _notify_network(self, network_state: Dict[str, Any]) -> None:
        """Notify all network callbacks."""
        for callback in self._network_callbacks:
            try:
                callback(network_state)
            except Exception as e:
                logger.error(f"Network callback error: {e}")

    def _notify_learning_complete(self, result: Dict[str, Any]) -> None:
        """Notify all learning complete callbacks."""
        for callback in self._learning_complete_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Learning complete callback error: {e}")

    # ==================== CURRICULUM LEARNING ====================

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum progress statistics."""
        return self.curriculum.get_stats()

    def get_current_level(self) -> int:
        """Get current curriculum level (1-5)."""
        return self.curriculum.current_level.value

    def get_current_level_name(self) -> str:
        """Get current level name."""
        return self.curriculum.CURRICULUM[self.curriculum.current_level].name

    def get_current_challenge_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the current curriculum challenge."""
        return self.curriculum.get_current_challenge()

    def start_curriculum_challenge(
        self,
        level: Optional[CurriculumLevel] = None,
        challenge_index: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Optional[LearningResult]:
        """
        Start a curriculum challenge.

        Args:
            level: Curriculum level (defaults to current)
            challenge_index: Challenge index within level (defaults to current)
            progress_callback: Optional progress callback

        Returns:
            LearningResult if challenge completed, None if no challenge available
        """
        # Set level and challenge if provided
        if level is not None:
            self.curriculum.set_level(level)
        if challenge_index is not None:
            self.curriculum.set_challenge_index(challenge_index)

        challenge_dict = self.curriculum.get_current_challenge()
        if challenge_dict is None:
            return None

        self.is_learning = True
        self.canvas_controller.set_learning_active(True)

        # Create Challenge object
        challenge = self.curriculum.create_challenge_object(challenge_dict)
        self.current_challenge = challenge

        # Execute learning
        def internal_callback(iteration: int, accuracy: float):
            self.current_accuracy = accuracy
            self.current_iterations = iteration
            if progress_callback:
                progress_callback(iteration, accuracy)
            self._notify_progress({
                "accuracy": accuracy,
                "iterations": iteration,
                "strategy": self.current_strategy,
                "challenge": challenge_dict["name"],
                "source": "curriculum",
            })

        result = self.challenge_learner.learn(challenge, callback=internal_callback)

        # Record result
        challenge_result = ChallengeResult(
            challenge_name=challenge_dict["name"],
            level=self.curriculum.current_level,
            accuracy=result.accuracy,
            passed=result.success,
            iterations=result.iterations,
            strategy_used=result.strategy_used,
        )
        self.curriculum.record_result(challenge_result)

        # Store in knowledge base
        self.knowledge_base.store_experience(
            state=self.internal_state,
            context={
                "challenge_name": challenge_dict["name"],
                "modalities": [m.name for m in challenge_dict["modalities"]],
                "accuracy": result.accuracy,
                "success": result.success,
                "level": self.curriculum.current_level.value,
            },
            emotional_valence=0.5 if result.success else -0.2,
            source="curriculum",
        )

        # Update internal state
        self._update_internal_state(result)

        self.is_learning = False
        self.canvas_controller.set_learning_active(False)
        self.current_challenge = None

        return result

    def skip_curriculum_challenge(self) -> bool:
        """Skip the current curriculum challenge."""
        return self.curriculum.skip_challenge()

    def advance_curriculum_level(self) -> bool:
        """Advance to next curriculum level if possible."""
        return self.curriculum.advance_to_next_level()

    # ==================== AUTO CURRICULUM LEARNING ====================

    def start_auto_curriculum_learning(self) -> None:
        """
        Start automatic curriculum learning.

        Atlas will continuously attempt challenges, retrying until they pass,
        then automatically advancing to the next challenge.
        """
        if self._auto_learning_active:
            logger.warning("Auto-learning already active")
            return

        self._stop_auto_learning.clear()
        self._auto_learning_active = True
        self._auto_learning_thread = threading.Thread(
            target=self._auto_learning_loop,
            daemon=True,
            name="AutoCurriculumLearning"
        )
        self._auto_learning_thread.start()
        logger.info("Started auto curriculum learning")

    def stop_auto_curriculum_learning(self) -> None:
        """Stop automatic curriculum learning."""
        if not self._auto_learning_active:
            return

        self._stop_auto_learning.set()
        self._auto_learning_active = False

        if self._auto_learning_thread:
            self._auto_learning_thread.join(timeout=5.0)
            self._auto_learning_thread = None

        logger.info("Stopped auto curriculum learning")

    def is_auto_learning_active(self) -> bool:
        """Check if auto-learning is currently active."""
        return self._auto_learning_active

    def _auto_learning_loop(self) -> None:
        """
        Main auto-learning loop.

        Continuously attempts challenges until stopped.
        Retries failed challenges until they pass.
        """
        logger.info("Auto-learning loop started")
        attempt_count = 0
        max_attempts_per_challenge = 100  # Limit retries to prevent infinite loops

        while not self._stop_auto_learning.is_set():
            # Get current challenge
            challenge_dict = self.curriculum.get_current_challenge()

            if challenge_dict is None:
                # No more challenges at current level - try to advance
                if self.curriculum.advance_to_next_level():
                    logger.info("Advanced to next level")
                    attempt_count = 0
                    self._notify_progress({
                        "message": f"Advanced to {self.curriculum.CURRICULUM[self.curriculum.current_level].name}",
                        "level_up": True,
                    })
                    continue
                else:
                    # Check if we're at max level or if next level needs unlock
                    current_level = self.curriculum.current_level
                    current_progress = self.curriculum.progress[current_level]
                    level_info = self.curriculum.CURRICULUM[current_level]

                    # Check if there's a next level
                    if current_level.value >= 5:
                        logger.info("Completed all curriculum levels!")
                        self._notify_progress({
                            "message": "Curriculum complete! All 5 levels mastered!",
                            "curriculum_complete": True,
                        })
                        time.sleep(10.0)
                        continue

                    # Next level exists but isn't unlocked - check why
                    next_level = CurriculumLevel(current_level.value + 1)
                    avg_accuracy = current_progress.average_accuracy
                    unlock_threshold = level_info.unlock_threshold

                    if avg_accuracy < unlock_threshold:
                        logger.info(
                            f"Level {current_level.value} avg accuracy {avg_accuracy:.1%} "
                            f"< {unlock_threshold:.0%} threshold. Replaying level to improve..."
                        )
                        self._notify_progress({
                            "message": f"Avg accuracy {avg_accuracy:.1%} < {unlock_threshold:.0%}. Replaying level...",
                        })

                        # Reset to replay the level
                        current_progress.current_challenge_index = 0
                        time.sleep(1.0)
                        continue
                    else:
                        # Should be unlocked - force unlock and try again
                        logger.info(f"Forcing unlock of level {next_level.value}")
                        self.curriculum.progress[next_level].unlocked = True
                        continue

            challenge_name = challenge_dict["name"]
            target_accuracy = challenge_dict.get("target_accuracy", 0.7)

            logger.info(f"Auto-learning: Attempting '{challenge_name}' (attempt {attempt_count + 1})")

            self._notify_progress({
                "message": f"Starting: {challenge_name} (attempt {attempt_count + 1})",
                "challenge": challenge_name,
                "attempt": attempt_count + 1,
            })

            # Execute the challenge
            try:
                result = self._execute_single_challenge(challenge_dict)

                if result:
                    passed = result.success
                    accuracy = result.accuracy

                    # Notify completion
                    self._notify_learning_complete({
                        "accuracy": accuracy,
                        "passed": passed,
                        "challenge": challenge_name,
                        "attempts": attempt_count + 1,
                        "strategy": result.strategy_used,
                    })

                    if passed:
                        logger.info(f"Challenge PASSED: {challenge_name} ({accuracy:.1%})")
                        attempt_count = 0  # Reset for next challenge

                        # Brief pause before next challenge
                        time.sleep(1.0)
                    else:
                        logger.info(f"Challenge FAILED: {challenge_name} ({accuracy:.1%}) - Retrying...")
                        attempt_count += 1

                        if attempt_count >= max_attempts_per_challenge:
                            logger.warning(f"Max attempts reached for {challenge_name}, skipping...")
                            self.curriculum.skip_challenge()
                            attempt_count = 0

                        # Brief pause before retry
                        time.sleep(0.5)
                else:
                    logger.error("Challenge returned no result")
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in auto-learning: {e}")
                time.sleep(2.0)

            # Check if we should stop
            if self._stop_auto_learning.is_set():
                break

        logger.info("Auto-learning loop ended")
        self._auto_learning_active = False

    def _execute_single_challenge(self, challenge_dict: Dict[str, Any]) -> Optional[LearningResult]:
        """Execute a single challenge and return the result."""
        self.is_learning = True
        self.canvas_controller.set_learning_active(True)

        # Check if this is a canvas generation challenge
        if challenge_dict.get("data_generator", "").startswith("canvas_"):
            return self._execute_canvas_challenge(challenge_dict)

        # Create Challenge object
        challenge = self.curriculum.create_challenge_object(challenge_dict)
        self.current_challenge = challenge

        # Update network cache at challenge start so UI has initial state
        self._update_network_state_cache()

        # Throttle UI updates to prevent overload
        last_update_time = [0.0]
        last_accuracy = [0.0]
        last_network_update = [0]  # Track iterations for network cache updates
        update_interval = 0.2  # Max 5 updates per second (reduced from 10)
        network_update_interval = 20  # Update network cache every 20 iterations

        # Execute learning with throttled progress callback
        def internal_callback(iteration: int, accuracy: float):
            self.current_accuracy = accuracy
            self.current_iterations = iteration

            # Throttle updates: only update if enough time passed or accuracy changed significantly
            current_time = time.time()
            accuracy_change = abs(accuracy - last_accuracy[0])
            time_elapsed = current_time - last_update_time[0]

            if time_elapsed >= update_interval or accuracy_change >= 0.1:
                last_update_time[0] = current_time
                last_accuracy[0] = accuracy
                self._notify_progress({
                    "accuracy": accuracy,
                    "epoch": iteration,
                    "strategy": self.current_strategy,
                    "challenge": challenge_dict["name"],
                    "source": "auto_curriculum",
                })

            # Update network state cache periodically (from learning thread)
            if iteration - last_network_update[0] >= network_update_interval:
                last_network_update[0] = iteration
                self._update_network_state_cache()

            # Yield to UI thread EVERY iteration to keep GUI responsive
            time.sleep(0.005)  # 5ms yield every iteration

        result = self.challenge_learner.learn(challenge, callback=internal_callback)

        # Record result
        challenge_result = ChallengeResult(
            challenge_name=challenge_dict["name"],
            level=self.curriculum.current_level,
            accuracy=result.accuracy,
            passed=result.success,
            iterations=result.iterations,
            strategy_used=result.strategy_used,
        )
        self.curriculum.record_result(challenge_result)

        # Store in knowledge base
        self.knowledge_base.store_experience(
            state=self.internal_state,
            context={
                "challenge_name": challenge_dict["name"],
                "modalities": [m.name for m in challenge_dict["modalities"]],
                "accuracy": result.accuracy,
                "success": result.success,
                "level": self.curriculum.current_level.value,
            },
            emotional_valence=0.5 if result.success else -0.2,
            source="auto_curriculum",
        )

        # Update internal state
        self._update_internal_state(result)

        # Provide reward to canvas controller for learning
        # This teaches Atlas to generate visuals that correlate with success
        reward = result.accuracy if result.success else (result.accuracy - 0.5)
        self.canvas_controller.provide_reward(reward)
        self.canvas_controller.update_internal_state(self.internal_state)

        # Update network cache one final time to capture end state
        self._update_network_state_cache()

        # Auto-save progress after each challenge
        self.save_state()

        self.is_learning = False
        self.canvas_controller.set_learning_active(False)
        self.current_challenge = None

        return result

    def _execute_canvas_challenge(self, challenge_dict: Dict[str, Any]) -> Optional[LearningResult]:
        """
        Execute a canvas generation challenge.

        Atlas learns to generate images on the 512x512 canvas that match target patterns.
        Uses Hebbian learning on the canvas controller's weights.
        """
        from self_organizing_av_system.core.challenge import LearningResult
        from self_organizing_av_system.core.backend import xp, to_cpu, to_gpu

        challenge_name = challenge_dict["name"]
        target_accuracy = challenge_dict.get("target_accuracy", 0.5)
        max_iterations = 500  # Canvas learning iterations

        logger.info(f"Starting canvas challenge: {challenge_name}")

        # Get target image from curriculum
        target_image = self.curriculum.get_target_image(challenge_dict)
        target_flat = target_image.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Update network cache at start
        self._update_network_state_cache()

        # Learning parameters
        learning_rate = 0.1
        best_accuracy = 0.0
        learning_curve = []
        start_time = time.time()

        # Initialize internal state randomly - no cheating with target info!
        # The system must learn to produce the target through trial and error
        self.internal_state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        self.canvas_controller.update_internal_state(self.internal_state)

        for iteration in range(max_iterations):
            # Generate current canvas
            generated = self.canvas_controller.generate_canvas()
            generated_flat = generated.flatten().astype(np.float32) / 255.0

            # Calculate accuracy using reconstruction error (MSE)
            # Lower MSE = higher accuracy (standard autoencoder metric)
            mse = np.mean((generated_flat - target_flat) ** 2)
            accuracy = max(0.0, 1.0 - mse)

            learning_curve.append(accuracy)
            best_accuracy = max(best_accuracy, accuracy)

            # Update canvas weights using error signal
            error = target_flat - generated_flat
            error_reshaped = error.reshape(512, 512, 3)

            # Compute gradient for canvas weights update
            # The canvas output is: sigmoid(internal_state @ canvas_weights)
            # We want to adjust weights to reduce error
            state = to_gpu(self.internal_state.astype(np.float32))
            state_norm = float(xp.linalg.norm(state))

            if state_norm > 0.01:
                state_normalized = state / (state_norm + 1e-8)
                error_gpu = to_gpu(error.astype(np.float32))

                # Hebbian update: strengthen connections that would produce target
                # ΔW = learning_rate * outer(state, error)
                dW = learning_rate * xp.outer(state_normalized, error_gpu)

                # Apply update with momentum-like effect
                self.canvas_controller.canvas_weights += dW

                # Weight decay to prevent explosion
                self.canvas_controller.canvas_weights *= 0.999

                # Also update internal state to move toward a state that produces target
                # This is key: we modify the internal state to better match what we want
                state_update = learning_rate * 0.1 * to_cpu(error_gpu[:self.state_dim])
                self.internal_state[:len(state_update)] += state_update
                self.internal_state = np.clip(self.internal_state, -2.0, 2.0)
                self.canvas_controller.update_internal_state(self.internal_state)

            # Update UI
            self.current_accuracy = accuracy
            self.current_iterations = iteration
            self.current_strategy = "Canvas-Hebbian"

            # Throttled progress notification
            if iteration % 10 == 0:
                self._notify_progress({
                    "accuracy": accuracy,
                    "epoch": iteration,
                    "strategy": "Canvas-Hebbian",
                    "challenge": challenge_name,
                    "source": "canvas_learning",
                })
                self._update_network_state_cache()

            # Yield to UI
            time.sleep(0.01)

            # Check success
            if accuracy >= target_accuracy:
                logger.info(f"Canvas challenge passed at iteration {iteration}: {accuracy:.2%}")
                break

            # Early stopping if stuck
            if iteration > 100 and len(learning_curve) > 50:
                recent_improvement = max(learning_curve[-50:]) - min(learning_curve[-50:])
                if recent_improvement < 0.01:
                    logger.info(f"Canvas learning plateaued at {accuracy:.2%}")
                    break

        # Final accuracy
        duration = time.time() - start_time
        passed = best_accuracy >= target_accuracy

        # Provide final reward
        reward = best_accuracy if passed else (best_accuracy - 0.5)
        self.canvas_controller.provide_reward(reward)

        # Create result
        result = LearningResult(
            challenge_id=challenge_name[:8],
            challenge_name=challenge_name,
            success=passed,
            accuracy=best_accuracy,
            iterations=iteration + 1,
            duration_seconds=duration,
            strategy_used="Canvas-Hebbian",
            learning_curve=learning_curve,
            final_metrics={"best_accuracy": best_accuracy, "final_accuracy": accuracy},
        )

        # Record in curriculum
        challenge_result = ChallengeResult(
            challenge_name=challenge_name,
            level=self.curriculum.current_level,
            accuracy=best_accuracy,
            passed=passed,
            iterations=iteration + 1,
            strategy_used="Canvas-Hebbian",
        )
        self.curriculum.record_result(challenge_result)

        # Store experience
        self.knowledge_base.store_experience(
            state=self.internal_state,
            context={
                "challenge_name": challenge_name,
                "modalities": ["CANVAS"],
                "accuracy": best_accuracy,
                "success": passed,
                "level": self.curriculum.current_level.value,
            },
            emotional_valence=0.5 if passed else -0.2,
            source="canvas_learning",
        )

        # Update network cache and save
        self._update_network_state_cache()
        self.save_state()

        self.is_learning = False
        self.canvas_controller.set_learning_active(False)
        self.current_challenge = None

        logger.info(f"Canvas challenge complete: {challenge_name} - {'PASSED' if passed else 'FAILED'} ({best_accuracy:.2%})")

        return result

    # ==================== FREE PLAY ====================

    def process_chat_message(self, message: str) -> GeneratedResponse:
        """
        Process a chat message from Free Play and generate response.

        Atlas learns to respond through biology-inspired rules.
        """
        # Check if this is a challenge request
        if self._is_challenge_request(message):
            # Process as a learning challenge
            result = self.process_free_play_challenge(message)
            context = ResponseContext(
                challenge_text=message,
                accuracy=result.accuracy if result else 0.0,
                iterations=result.iterations if result else 0,
                strategy=result.strategy_used if result else "none",
                success=result.success if result else False,
                source="free_play",
            )
        else:
            # General conversation
            context = ResponseContext(
                challenge_text=message,
                accuracy=self.current_accuracy,
                iterations=self.current_iterations,
                strategy=self.current_strategy,
                success=True,
                source="free_play",
            )

        # Generate response through learned weights
        response = self.text_learner.generate_response(context)

        # Store interaction in knowledge base
        self.knowledge_base.store_experience(
            state=self.internal_state,
            context={
                "message": message,
                "response": response.text,
                "type": "conversation",
            },
            source="free_play",
        )

        return response

    def _is_challenge_request(self, message: str) -> bool:
        """Check if message is a learning challenge request."""
        keywords = ["learn", "classify", "recognize", "identify", "predict", "detect"]
        message_lower = message.lower()
        return any(kw in message_lower for kw in keywords)

    def process_free_play_challenge(
        self,
        description: str,
        progress_callback: Optional[Callable] = None
    ) -> Optional[LearningResult]:
        """Process a natural language challenge from Free Play."""
        self.is_learning = True
        self.canvas_controller.set_learning_active(True)

        def internal_callback(iteration: int, accuracy: float):
            self.current_accuracy = accuracy
            self.current_iterations = iteration
            if progress_callback:
                progress_callback(iteration, accuracy)
            self._notify_progress({
                "accuracy": accuracy,
                "iterations": iteration,
                "strategy": self.current_strategy,
                "challenge": description[:50],
                "source": "free_play",
            })

        result = self.challenge_learner.learn(description, callback=internal_callback)

        if result:
            self._update_internal_state(result)

            # Store in knowledge base
            self.knowledge_base.store_experience(
                state=self.internal_state,
                context={
                    "challenge_name": description[:100],
                    "accuracy": result.accuracy,
                    "success": result.success,
                },
                emotional_valence=0.3 if result.success else -0.1,
                source="free_play",
            )

        self.is_learning = False
        self.canvas_controller.set_learning_active(False)

        return result

    def provide_chat_feedback(self, response: GeneratedResponse, positive: bool) -> None:
        """Provide feedback on a chat response to improve learning."""
        feedback_signal = 1.0 if positive else -0.5
        context = ResponseContext(
            accuracy=self.current_accuracy,
            strategy=self.current_strategy,
            source="free_play",
        )
        self.text_learner.learn_from_feedback(response, feedback_signal, context)

    # ==================== AV PROCESSING ====================

    def process_av_frame(self, frame: np.ndarray, audio: Optional[np.ndarray] = None) -> None:
        """
        Process AV input from webcam/microphone.

        This feeds the input INTO Atlas for processing, not just display.
        """
        self.last_frame = frame
        self.last_audio = audio
        self.av_processing_active = True

        # Simple feature extraction
        if frame is not None:
            # Visual features: mean RGB
            visual_features = np.mean(frame, axis=(0, 1)) / 255.0

            # Update part of internal state with visual info
            visual_dim = min(len(visual_features), self.state_dim // 4)
            self.internal_state[:visual_dim] = (
                0.8 * self.internal_state[:visual_dim] +
                0.2 * visual_features[:visual_dim]
            )

        if audio is not None:
            # Audio features: RMS, zero crossings
            rms = np.sqrt(np.mean(audio**2))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)

            audio_idx = self.state_dim // 4
            self.internal_state[audio_idx] = 0.8 * self.internal_state[audio_idx] + 0.2 * rms
            self.internal_state[audio_idx + 1] = 0.8 * self.internal_state[audio_idx + 1] + 0.2 * zero_crossings

        # Update canvas controller with new state
        self.canvas_controller.update_internal_state(self.internal_state)

    # ==================== CREATIVE CANVAS ====================

    def generate_creative_canvas(self) -> np.ndarray:
        """Generate the 512x512 creative canvas."""
        return self.canvas_controller.generate_canvas()

    def get_canvas_stats(self) -> Dict[str, Any]:
        """Get canvas statistics."""
        return self.canvas_controller.get_stats()

    # ==================== KNOWLEDGE BASE ====================

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return self.knowledge_base.get_stats()

    def get_recent_knowledge_event(self) -> str:
        """Get the most recent knowledge event description."""
        return self.knowledge_base.get_recent_event_string()

    # ==================== NETWORK STATE ====================

    def get_network_state(self) -> Dict[str, Any]:
        """
        Get current neural network state for visualization.

        Thread-safe: Returns cached state that is updated from the learning thread.
        This prevents UI thread from blocking on learning engine access.
        """
        with self._network_state_lock:
            # Return a copy to prevent mutation issues
            state = self._cached_network_state.copy()
            # Add current internal state norm (cheap, safe from UI thread)
            state["internal_state_norm"] = float(np.linalg.norm(self.internal_state))
            state["multimodal_size"] = self.state_dim
            return state

    def _update_network_state_cache(self) -> None:
        """
        Update the cached network state from the learning engine.

        Called from the learning thread to avoid UI thread blocking.
        """
        try:
            # Get the real network structure from the learning engine
            network_structure = self.challenge_learner.learning_engine.get_network_structure()

            # Get recent structural changes for animation
            recent_changes = self.challenge_learner.learning_engine.get_recent_structural_changes(20)

            # Count recent additions and prunings
            neurons_added = sum(
                sum(item['count'] for item in change.get('neurons_added', []))
                for change in recent_changes if 'neurons_added' in change
            )
            neurons_pruned = sum(
                sum(item['count'] for item in change.get('neurons_pruned', []))
                for change in recent_changes if 'neurons_pruned' in change
            )

            new_state = {
                # Real network structure
                "num_layers": network_structure.get("num_layers", 0),
                "layers": network_structure.get("layers", []),
                "total_neurons": network_structure.get("total_neurons", 0),
                "total_connections": network_structure.get("total_connections", 0),

                # Structural plasticity events
                "recent_events": network_structure.get("recent_events", []),
                "neurons_added": neurons_added,
                "neurons_pruned": neurons_pruned,

                # Stats from the network
                "stats": network_structure.get("stats", {}),

                # Legacy compatibility fields
                "visual_layers": [l["size"] for l in network_structure.get("layers", [])],
                "audio_layers": [l["size"] for l in network_structure.get("layers", [])],
                "connections": network_structure.get("total_connections", 0),
                "plasticity_events": recent_changes,
            }

            # Thread-safe update
            with self._network_state_lock:
                self._cached_network_state = new_state
                self._network_state_update_counter += 1

        except Exception as e:
            logger.error(f"Error updating network state cache: {e}")

    def get_recent_structural_changes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent structural changes for animation.

        Thread-safe: Returns cached plasticity events.
        """
        with self._network_state_lock:
            events = self._cached_network_state.get("plasticity_events", [])
            return events[:limit] if len(events) > limit else events

    def register_structural_change_callback(self, callback: Callable) -> None:
        """Register callback for structural changes (for real-time visualization)."""
        self.challenge_learner.learning_engine.register_structural_change_callback(callback)

    # ==================== INTERNAL STATE ====================

    def _update_internal_state(self, result: LearningResult) -> None:
        """Update internal state based on learning result."""
        # Blend in learning information
        learning_signal = np.array([
            result.accuracy,
            result.iterations / 1000.0,
            1.0 if result.success else 0.0,
        ])

        # Update a portion of internal state
        idx = self.state_dim // 2
        self.internal_state[idx:idx+3] = (
            0.7 * self.internal_state[idx:idx+3] +
            0.3 * learning_signal
        )

        self.current_strategy = result.strategy_used

    # ==================== STATISTICS ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        backend_info = get_backend_info()

        return {
            "gpu_available": HAS_GPU,
            "gpu_name": backend_info.get("device_name", "N/A"),
            "gpu_memory_gb": backend_info.get("device_memory_gb", 0),
            "state_dim": self.state_dim,
            "is_learning": self.is_learning,
            "current_accuracy": self.current_accuracy,
            "current_strategy": self.current_strategy,
            "av_processing": self.av_processing_active,
            "challenge_learner": self.challenge_learner.get_stats(),
            "curriculum": self.curriculum.get_stats(),
            "knowledge_base": self.knowledge_base.get_stats(),
            "text_learner": self.text_learner.get_stats(),
            "canvas": self.canvas_controller.get_stats(),
        }

    def stop(self) -> None:
        """Stop all background processes."""
        self.stop_auto_curriculum_learning()
        self.save_state()  # Auto-save on shutdown
        self.knowledge_base.stop()
        logger.info("AtlasController stopped")

    def reset_learning(self) -> None:
        """
        Reset all learned weights and progress to start fresh.

        This clears:
        - Canvas weights (re-randomized)
        - Curriculum progress (back to level 1)
        - Internal state
        - Saved state file
        """
        logger.info("Resetting all learning to fresh state...")

        # Stop any active learning
        self.stop_auto_curriculum_learning()

        # Reset canvas weights to random
        self.canvas_controller.reset_weights()

        # Reset internal state
        self.internal_state = np.zeros(self.state_dim, dtype=np.float32)

        # Reset curriculum progress
        self.curriculum.reset_progress()

        # Reset current stats
        self.current_accuracy = 0.0
        self.current_iterations = 0
        self.current_strategy = "Hebbian"

        # Delete saved state file
        save_path = self._get_save_path()
        if save_path.exists():
            save_path.unlink()
            logger.info(f"Deleted saved state: {save_path}")

        # Update network visualization
        self._update_network_state_cache()

        logger.info("Learning reset complete - starting fresh!")

    # ==================== PERSISTENCE ====================

    def _get_save_path(self) -> Path:
        """Get the path for saving Atlas state."""
        # Save in user's home directory under .atlas
        save_dir = Path.home() / ".atlas"
        save_dir.mkdir(exist_ok=True)
        return save_dir / "atlas_state.pkl"

    def save_state(self) -> bool:
        """
        Save Atlas's learned state to disk.

        This is called automatically on shutdown and periodically during learning.
        """
        try:
            save_path = self._get_save_path()

            # Collect state to save
            state = {
                "version": 1,
                "internal_state": self.internal_state.tolist(),
                "current_accuracy": self.current_accuracy,
                "current_strategy": self.current_strategy,

                # Curriculum progress
                "curriculum": {
                    "current_level": self.curriculum.current_level.value,
                    "progress": {},
                    "results": [],
                },

                # Learned capabilities
                "capabilities": {},
            }

            # Save curriculum progress for each level
            for level, progress in self.curriculum.progress.items():
                state["curriculum"]["progress"][level.value] = {
                    "challenges_completed": progress.challenges_completed,
                    "current_challenge_index": progress.current_challenge_index,
                    "accuracies": progress.accuracies,
                    "unlocked": progress.unlocked,
                    "completed": progress.completed,
                }

            # Save curriculum results
            for result in self.curriculum.results:
                state["curriculum"]["results"].append({
                    "challenge_name": result.challenge_name,
                    "level": result.level.value,
                    "accuracy": result.accuracy,
                    "passed": result.passed,
                    "iterations": result.iterations,
                    "strategy_used": result.strategy_used,
                    "timestamp": result.timestamp,
                })

            # Save learned capabilities from learning engine
            for cap_id, cap in self.challenge_learner.learning_engine.capabilities.items():
                state["capabilities"][cap_id] = {
                    "name": cap.name,
                    "description": cap.description,
                    "challenge_id": cap.challenge_id,
                    "proficiency": cap.proficiency,
                    "weights": to_cpu(cap.weights).tolist() if cap.weights is not None else None,
                    "use_count": cap.use_count,
                }

            # Save canvas controller learned state
            state["canvas"] = {
                "canvas_weights": to_cpu(self.canvas_controller.canvas_weights).tolist(),
                "creativity_level": self.canvas_controller.creativity_level,
                "weight_updates": self.canvas_controller.weight_updates,
                "last_reward": self.canvas_controller.last_reward,
            }

            # Save the self-organizing network state
            state["network"] = self.challenge_learner.learning_engine.network.serialize()

            # Write to file
            with open(save_path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Saved Atlas state to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save Atlas state: {e}")
            return False

    def load_state(self) -> bool:
        """
        Load Atlas's learned state from disk.

        This is called automatically on startup.
        """
        try:
            save_path = self._get_save_path()

            if not save_path.exists():
                logger.info("No saved state found, starting fresh")
                return False

            with open(save_path, "rb") as f:
                state = pickle.load(f)

            # Restore internal state
            self.internal_state = np.array(state.get("internal_state", np.zeros(self.state_dim)))
            self.current_accuracy = state.get("current_accuracy", 0.0)
            self.current_strategy = state.get("current_strategy", "Hebbian")

            # Restore curriculum progress
            curriculum_state = state.get("curriculum", {})

            # Set current level
            current_level_value = curriculum_state.get("current_level", 1)
            self.curriculum.current_level = CurriculumLevel(current_level_value)

            # Restore progress for each level
            for level_value, progress_data in curriculum_state.get("progress", {}).items():
                level = CurriculumLevel(int(level_value))
                if level in self.curriculum.progress:
                    self.curriculum.progress[level].challenges_completed = progress_data.get("challenges_completed", 0)
                    self.curriculum.progress[level].current_challenge_index = progress_data.get("current_challenge_index", 0)
                    self.curriculum.progress[level].accuracies = progress_data.get("accuracies", [])
                    self.curriculum.progress[level].unlocked = progress_data.get("unlocked", False)
                    self.curriculum.progress[level].completed = progress_data.get("completed", False)

            # Ensure level 1 is always unlocked
            self.curriculum.progress[CurriculumLevel.LEVEL_1_BASIC].unlocked = True

            # Restore results
            for result_data in curriculum_state.get("results", []):
                result = ChallengeResult(
                    challenge_name=result_data["challenge_name"],
                    level=CurriculumLevel(result_data["level"]),
                    accuracy=result_data["accuracy"],
                    passed=result_data["passed"],
                    iterations=result_data["iterations"],
                    strategy_used=result_data["strategy_used"],
                    timestamp=result_data.get("timestamp", time.time()),
                )
                self.curriculum.results.append(result)

            # Restore canvas controller learned state
            canvas_state = state.get("canvas", {})
            if canvas_state:
                if "canvas_weights" in canvas_state:
                    loaded_weights = np.array(canvas_state["canvas_weights"], dtype=np.float32)
                    # Only load if shape matches (in case state_dim changed)
                    if loaded_weights.shape == self.canvas_controller.canvas_weights.shape:
                        self.canvas_controller.canvas_weights = to_gpu(loaded_weights)
                    else:
                        logger.warning("Canvas weights shape mismatch, using fresh weights")
                self.canvas_controller.creativity_level = canvas_state.get("creativity_level", 0.5)
                self.canvas_controller.weight_updates = canvas_state.get("weight_updates", 0)
                self.canvas_controller.last_reward = canvas_state.get("last_reward", 0.0)
                logger.info(f"Restored canvas state: {self.canvas_controller.weight_updates} weight updates")

            # Restore the self-organizing network state
            network_state = state.get("network", None)
            if network_state:
                try:
                    from self_organizing_av_system.core.self_organizing_network import SelfOrganizingNetwork
                    self.challenge_learner.learning_engine.network = SelfOrganizingNetwork.deserialize(network_state)
                    logger.info(
                        f"Restored network: {self.challenge_learner.learning_engine.network.get_stats()['total_neurons']} neurons"
                    )
                except Exception as e:
                    logger.warning(f"Could not restore network state: {e}")

            # Log what was restored
            stats = self.curriculum.get_stats()
            logger.info(
                f"Loaded Atlas state: Level {stats['current_level']}, "
                f"{stats['completed_challenges']}/{stats['total_challenges']} challenges completed"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load Atlas state: {e}")
            # Delete corrupted state file so we can start fresh
            try:
                save_path = self._get_save_path()
                if save_path.exists():
                    save_path.unlink()
                    logger.info("Deleted corrupted state file, starting fresh")
            except Exception:
                pass
            return False

    # ==================== DATABASE PERSISTENCE ====================

    def _init_database_stores(self) -> None:
        """Initialize persistent database stores."""
        if not DATABASE_AVAILABLE:
            logger.warning("Database module not available, skipping persistence initialization")
            return

        try:
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Initialize vector store (ChromaDB)
            self.vector_store = VectorStore(
                data_dir=str(self.data_dir),
                embedding_dim=self.state_dim,
            )
            logger.info(f"VectorStore initialized: {self.vector_store.get_stats()}")

            # Initialize graph store (SQLite)
            self.graph_store = GraphStore(
                data_dir=str(self.data_dir),
            )
            logger.info(f"GraphStore initialized: {self.graph_store.get_stats()}")

            # Initialize network store (SQLite)
            self.network_store = NetworkStore(
                data_dir=str(self.data_dir),
            )
            logger.info(f"NetworkStore initialized: {self.network_store.get_stats()}")

        except Exception as e:
            logger.error(f"Failed to initialize database stores: {e}")
            self.vector_store = None
            self.graph_store = None
            self.network_store = None
            self.enable_persistence = False

    def _load_network_weights(self) -> None:
        """Load network weights from persistent storage."""
        if not self.network_store:
            return

        try:
            # Load canvas controller weights
            result = self.network_store.load_weights("canvas_controller")
            if result:
                weights, metadata = result
                if weights.shape == self.canvas_controller.canvas_weights.shape:
                    self.canvas_controller.canvas_weights = to_gpu(weights)
                    logger.info(f"Loaded canvas weights from database ({metadata.get('weight_updates', 0)} updates)")
                else:
                    logger.warning(f"Canvas weights shape mismatch: {weights.shape} vs {self.canvas_controller.canvas_weights.shape}")

            # Load challenge learner weights
            result = self.network_store.load_weights("challenge_learner")
            if result:
                weights, metadata = result
                # TODO: Apply to challenge learner
                logger.info("Loaded challenge learner weights from database")

        except Exception as e:
            logger.error(f"Failed to load network weights: {e}")

    def _save_network_weights(self) -> None:
        """Save network weights to persistent storage."""
        if not self.network_store:
            return

        try:
            # Save canvas controller weights
            canvas_weights = to_cpu(self.canvas_controller.canvas_weights)
            self.network_store.save_weights(
                "canvas_controller",
                canvas_weights,
                metadata={
                    "weight_updates": self.canvas_controller.weight_updates,
                    "creativity_level": self.canvas_controller.creativity_level,
                }
            )

            # Log a learning event
            self.network_store.log_learning_event(
                "canvas_controller",
                "save",
                metrics={"weight_updates": self.canvas_controller.weight_updates},
                description="Auto-save during consolidation"
            )

        except Exception as e:
            logger.error(f"Failed to save network weights: {e}")

    def close(self) -> None:
        """
        Close the Atlas controller and all resources.

        Ensures all data is saved and connections are closed.
        """
        logger.info("Closing AtlasController...")

        # Stop background processes
        self.stop()

        # Save network weights
        if self.enable_persistence:
            self._save_network_weights()

        # Close knowledge base (which closes its stores)
        self.knowledge_base.close()

        # Close network store
        if self.network_store:
            self.network_store.close()

        logger.info("AtlasController closed")
