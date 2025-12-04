"""
Processing Thread - Background Learning and AV Processing

Background thread for Atlas's learning and AV data processing.
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any
from queue import Queue, Empty

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtCore import QThread, pyqtSignal
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """
    Background thread for learning and processing.

    Handles:
    - Curriculum challenge training
    - AV data processing (feeding into Atlas)
    - Memory consolidation
    - Continuous learning updates
    """

    metricsUpdated = pyqtSignal(dict)
    learningComplete = pyqtSignal(dict)
    networkUpdated = pyqtSignal(dict)

    # Throttle network updates to prevent UI overload
    NETWORK_UPDATE_INTERVAL_MS = 500  # Update every 500ms max

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._running = False
        self._task_queue: Queue = Queue()
        self._current_task = None
        self._last_network_update_ms = 0

    def run(self):
        """Main processing loop."""
        self._running = True
        logger.info("Processing thread started")

        while self._running:
            try:
                # Check for new tasks
                try:
                    task = self._task_queue.get_nowait()
                    self._process_task(task)
                except Empty:
                    pass

                # Run continuous processing
                self._continuous_processing()

                # Sleep briefly
                self.msleep(10)

            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.msleep(100)

        logger.info("Processing thread stopped")

    def _process_task(self, task: Dict[str, Any]):
        """Process a queued task."""
        task_type = task.get("type")

        if task_type == "curriculum_challenge":
            self._run_curriculum_challenge(task)

        elif task_type == "free_play_learn":
            self._run_free_play_learning(task)

        elif task_type == "process_av":
            self._process_av_data(task)

    def _run_curriculum_challenge(self, task: Dict[str, Any]):
        """Run a curriculum challenge training session."""
        level = task.get("level")
        challenge_idx = task.get("challenge_index")

        logger.info(f"Starting curriculum challenge: level={level}, index={challenge_idx}")

        self._current_task = task

        # Get challenge data
        data, labels = self.controller.curriculum.generate_challenge_data(
            level, challenge_idx
        )

        if data is None:
            logger.error("Failed to generate challenge data")
            return

        # Training parameters
        epochs = 50
        batch_size = 32

        for epoch in range(epochs):
            if not self._running:
                break

            # Train one epoch
            metrics = self.controller.challenge_learner.train_epoch(
                data, labels,
                batch_size=batch_size
            )

            # Update metrics
            metrics["epoch"] = epoch + 1
            metrics["total_epochs"] = epochs
            metrics["progress"] = (epoch + 1) / epochs

            self.metricsUpdated.emit(metrics)

            # Check if we've hit target
            target = self.controller.curriculum.get_challenge_target(level, challenge_idx)
            if metrics.get("accuracy", 0) >= target:
                logger.info(f"Challenge target reached: {metrics['accuracy']:.1%} >= {target:.1%}")
                break

            # Small delay for UI responsiveness
            self.msleep(50)

        # Complete
        result = {
            "level": level,
            "challenge_index": challenge_idx,
            "accuracy": metrics.get("accuracy", 0),
            "passed": metrics.get("accuracy", 0) >= target,
        }

        # Update curriculum progress
        self.controller.curriculum.update_progress(
            level, challenge_idx, result["accuracy"]
        )

        self.learningComplete.emit(result)
        self._current_task = None

    def _run_free_play_learning(self, task: Dict[str, Any]):
        """Run free play learning session."""
        # Get learning data from task
        data = task.get("data")
        context = task.get("context", {})

        if data is None:
            return

        # Process through challenge learner
        result = self.controller.challenge_learner.process_experience(
            data, context
        )

        # Store in knowledge base
        self.controller.knowledge_base.store_experience(
            state=result.get("state"),
            context=context,
            sensory_data=data,
            emotional_valence=result.get("valence", 0),
            source="free_play"
        )

        # Emit metrics
        self.metricsUpdated.emit({
            "type": "free_play",
            "learned": True,
        })

    def _process_av_data(self, task: Dict[str, Any]):
        """Process audio-visual data."""
        frame = task.get("frame")
        audio = task.get("audio")

        if frame is None:
            return

        # Process through controller
        self.controller.process_av_frame(frame, audio)

    def _continuous_processing(self):
        """Run continuous background processing."""
        # Memory consolidation (runs quickly, no throttle needed)
        if hasattr(self.controller, 'knowledge_base'):
            self.controller.knowledge_base.consolidate()

        # Throttled network structure updates
        current_time_ms = int(time.time() * 1000)
        if current_time_ms - self._last_network_update_ms >= self.NETWORK_UPDATE_INTERVAL_MS:
            self._last_network_update_ms = current_time_ms
            # get_network_state() now reads from thread-safe cache, very fast
            network_state = self.controller.get_network_state()
            if network_state:
                self.networkUpdated.emit(network_state)

    def queue_task(self, task: Dict[str, Any]):
        """Add a task to the processing queue."""
        self._task_queue.put(task)

    def stop(self):
        """Stop the processing thread."""
        self._running = False

    def is_processing(self) -> bool:
        """Check if currently processing a task."""
        return self._current_task is not None
