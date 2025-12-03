#!/usr/bin/env python3
"""
Atlas Cloud Service - Main entry point for cloud-hosted Atlas

This service enables Atlas to:
1. Run continuously in the cloud
2. Ingest data from various sources (streams, files, APIs)
3. Learn autonomously without supervision
4. Persist learned knowledge across restarts
5. Scale horizontally for increased capacity
6. Monitor its own performance and health
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual.processor import VisualProcessor
from models.audio.processor import AudioProcessor
from models.multimodal.system import SelfOrganizingAVSystem
from config.configuration import SystemConfig

# Try to import cognitive systems
try:
    from core.unified_intelligence import UnifiedSuperIntelligence
    HAS_UNIFIED_INTELLIGENCE = True
except ImportError:
    HAS_UNIFIED_INTELLIGENCE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class AtlasMetrics:
    """Prometheus metrics for Atlas monitoring."""

    def __init__(self):
        if HAS_PROMETHEUS:
            self.frames_processed = Counter(
                'atlas_frames_processed_total',
                'Total number of frames processed'
            )
            self.learning_cycles = Counter(
                'atlas_learning_cycles_total',
                'Total number of learning cycles completed'
            )
            self.knowledge_items = Gauge(
                'atlas_knowledge_items',
                'Number of items in knowledge store',
                ['memory_type']
            )
            self.processing_time = Histogram(
                'atlas_processing_time_seconds',
                'Time to process each frame',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            )
            self.memory_usage = Gauge(
                'atlas_memory_usage_bytes',
                'Current memory usage'
            )
            self.intelligence_score = Gauge(
                'atlas_intelligence_score',
                'Current unified intelligence quotient'
            )
            self.creativity_index = Gauge(
                'atlas_creativity_index',
                'Current creativity index'
            )
            self.self_improvement_cycles = Counter(
                'atlas_self_improvement_cycles_total',
                'Total self-improvement optimization cycles'
            )
        else:
            # Stub metrics for when prometheus isn't available
            self.frames_processed = None
            self.learning_cycles = None

    def increment_frames(self):
        if self.frames_processed:
            self.frames_processed.inc()

    def increment_learning(self):
        if self.learning_cycles:
            self.learning_cycles.inc()

    def observe_processing_time(self, duration: float):
        if HAS_PROMETHEUS and hasattr(self, 'processing_time'):
            self.processing_time.observe(duration)

    def set_knowledge_items(self, memory_type: str, count: int):
        if HAS_PROMETHEUS and hasattr(self, 'knowledge_items'):
            self.knowledge_items.labels(memory_type=memory_type).set(count)

    def set_intelligence_score(self, score: float):
        if HAS_PROMETHEUS and hasattr(self, 'intelligence_score'):
            self.intelligence_score.set(score)

    def set_creativity_index(self, index: float):
        if HAS_PROMETHEUS and hasattr(self, 'creativity_index'):
            self.creativity_index.set(index)

    def increment_self_improvement(self):
        if HAS_PROMETHEUS and hasattr(self, 'self_improvement_cycles'):
            self.self_improvement_cycles.inc()


class AtlasCloudService:
    """
    Main cloud service for running Atlas autonomously.

    This service manages:
    - System initialization and configuration
    - Continuous learning loops
    - Data ingestion from multiple sources
    - Knowledge persistence and checkpointing
    - Health monitoring and metrics
    - Graceful shutdown and recovery
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("AtlasCloudService")

        # Load configuration
        self.config = SystemConfig(config_path)
        self._apply_cloud_config_overrides()

        # Initialize metrics
        self.metrics = AtlasMetrics()

        # State tracking
        self.running = False
        self.start_time = None
        self.total_frames = 0
        self.total_learning_cycles = 0

        # Initialize the system
        self._initialize_system()

        # Thread management
        self.threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

        self.logger.info("Atlas Cloud Service initialized")

    def _apply_cloud_config_overrides(self):
        """Apply cloud-specific configuration overrides from environment."""
        # Override checkpoint directory from environment
        checkpoint_dir = os.environ.get('ATLAS_CHECKPOINT_DIR', '/data/checkpoints')
        self.config.update("checkpointing", "checkpoint_dir", checkpoint_dir)
        self.config.update("checkpointing", "enabled", True)
        self.config.update("checkpointing", "load_latest", True)

        # Cloud-optimized settings
        self.config.update("checkpointing", "checkpoint_interval",
                          int(os.environ.get('ATLAS_CHECKPOINT_INTERVAL', '1000')))
        self.config.update("checkpointing", "max_checkpoints",
                          int(os.environ.get('ATLAS_MAX_CHECKPOINTS', '5')))

        # Learning parameters from environment
        if os.environ.get('ATLAS_LEARNING_RATE'):
            self.config.update("system", "learning_rate",
                             float(os.environ.get('ATLAS_LEARNING_RATE')))

        # Network size from environment
        if os.environ.get('ATLAS_MULTIMODAL_SIZE'):
            self.config.update("system", "multimodal_size",
                             int(os.environ.get('ATLAS_MULTIMODAL_SIZE')))

    def _initialize_system(self):
        """Initialize the Atlas system components."""
        self.logger.info("Initializing Atlas system...")

        # Get configurations
        visual_config = self.config.get_visual_config()
        audio_config = self.config.get_audio_config()
        system_config = self.config.get_system_config()

        # Create processors
        self.visual_processor = VisualProcessor(
            input_width=visual_config["input_width"],
            input_height=visual_config["input_height"],
            use_grayscale=visual_config["use_grayscale"],
            patch_size=visual_config["patch_size"],
            stride=visual_config["stride"],
            contrast_normalize=visual_config["contrast_normalize"],
            layer_sizes=visual_config["layer_sizes"]
        )

        self.audio_processor = AudioProcessor(
            sample_rate=audio_config["sample_rate"],
            window_size=audio_config["window_size"],
            hop_length=audio_config["hop_length"],
            n_mels=audio_config["n_mels"],
            min_freq=audio_config["min_freq"],
            max_freq=audio_config["max_freq"],
            normalize=audio_config["normalize"],
            layer_sizes=audio_config["layer_sizes"]
        )

        # Create main system
        self.system = SelfOrganizingAVSystem(
            visual_processor=self.visual_processor,
            audio_processor=self.audio_processor,
            multimodal_size=system_config["multimodal_size"],
            prune_interval=system_config["prune_interval"],
            structural_plasticity_interval=system_config["structural_plasticity_interval"],
            learning_rate=system_config["learning_rate"],
            learning_rule=system_config["learning_rule"]
        )

        # Initialize unified intelligence if available
        self.unified_intelligence = None
        if HAS_UNIFIED_INTELLIGENCE and os.environ.get('ATLAS_ENABLE_UNIFIED_INTELLIGENCE', 'true').lower() == 'true':
            try:
                self.unified_intelligence = UnifiedSuperIntelligence(
                    sensory_dim=visual_config["input_width"] * visual_config["input_height"],
                    hidden_dim=system_config["multimodal_size"],
                    enable_self_improvement=True
                )
                self.logger.info("Unified Super Intelligence enabled")
            except Exception as e:
                self.logger.warning(f"Could not initialize unified intelligence: {e}")

        # Try to load latest checkpoint
        self._load_latest_checkpoint()

        self.logger.info("Atlas system initialized successfully")

    def _load_latest_checkpoint(self):
        """Load the most recent checkpoint if available."""
        checkpoint_config = self.config.get_checkpointing_config()
        checkpoint_dir = checkpoint_config["checkpoint_dir"]

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            return

        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pkl"))

        if not checkpoint_files:
            self.logger.info("No existing checkpoints found - starting fresh")
            return

        # Get most recent checkpoint
        latest = max(checkpoint_files, key=os.path.getctime)

        try:
            if self.system.load_state(latest):
                self.total_frames = self.system.frame_count
                self.logger.info(f"Loaded checkpoint with {self.total_frames} frames of experience")
            else:
                self.logger.warning("Failed to load checkpoint - starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def _save_checkpoint(self, is_final: bool = False):
        """Save a checkpoint of the current state."""
        checkpoint_config = self.config.get_checkpointing_config()
        checkpoint_dir = checkpoint_config["checkpoint_dir"]

        os.makedirs(checkpoint_dir, exist_ok=True)

        prefix = "final" if is_final else "checkpoint"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(
            checkpoint_dir,
            f"{prefix}_{timestamp}_{self.system.frame_count}.pkl"
        )

        try:
            if self.system.save_state(checkpoint_file):
                self.logger.info(f"Saved checkpoint: {checkpoint_file}")
                self._cleanup_old_checkpoints(checkpoint_config)
                return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

        return False

    def _cleanup_old_checkpoints(self, checkpoint_config: Dict[str, Any]):
        """Remove old checkpoints keeping only max_checkpoints most recent."""
        import glob
        checkpoint_dir = checkpoint_config["checkpoint_dir"]
        max_checkpoints = checkpoint_config.get("max_checkpoints", 5)

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl"))

        if len(checkpoint_files) > max_checkpoints:
            checkpoint_files.sort(key=os.path.getctime)
            for old_file in checkpoint_files[:-max_checkpoints]:
                try:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old checkpoint: {old_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old checkpoint: {e}")

    def process_frame(self, visual_data: np.ndarray, audio_data: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Process a single audio-visual frame.

        Args:
            visual_data: Visual input array (image)
            audio_data: Audio input array (waveform chunk)
            learn: Whether to enable learning

        Returns:
            Dictionary of processing results
        """
        start_time = time.time()

        # Process through main system
        result = self.system.process_av_pair(visual_data, audio_data, learn=learn)

        # Process through unified intelligence if available
        if self.unified_intelligence and learn:
            try:
                # Flatten visual data for unified intelligence
                flat_visual = visual_data.flatten()
                ui_result = self.unified_intelligence.process(flat_visual)
                result['unified_intelligence'] = ui_result

                # Update metrics from unified intelligence
                if 'intelligence_quotient' in ui_result:
                    self.metrics.set_intelligence_score(ui_result['intelligence_quotient'])
                if 'creativity_index' in ui_result:
                    self.metrics.set_creativity_index(ui_result['creativity_index'])
            except Exception as e:
                self.logger.debug(f"Unified intelligence processing error: {e}")

        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.increment_frames()
        self.metrics.observe_processing_time(processing_time)
        self.total_frames += 1

        # Periodic checkpoint
        checkpoint_interval = self.config.get_checkpointing_config()["checkpoint_interval"]
        if self.total_frames % checkpoint_interval == 0:
            self._save_checkpoint()
            self.total_learning_cycles += 1
            self.metrics.increment_learning()

        return result

    def run_autonomous_learning(self):
        """
        Run autonomous learning from available data sources.

        This method continuously ingests and learns from:
        - Video/image files in the input directory
        - Streaming data sources
        - Generated/synthetic data for exploration
        """
        self.logger.info("Starting autonomous learning loop...")

        input_dir = os.environ.get('ATLAS_INPUT_DIR', '/data/input')

        while not self.shutdown_event.is_set():
            try:
                # Check for new data files
                self._process_input_directory(input_dir)

                # Run self-improvement cycles if unified intelligence is available
                if self.unified_intelligence:
                    self._run_self_improvement_cycle()

                # Generate and process synthetic exploration data
                if os.environ.get('ATLAS_ENABLE_EXPLORATION', 'true').lower() == 'true':
                    self._exploration_cycle()

                # Small delay to prevent CPU spinning
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in autonomous learning: {e}")
                time.sleep(1.0)  # Back off on errors

    def _process_input_directory(self, input_dir: str):
        """Process any new files in the input directory."""
        import glob
        import cv2

        if not os.path.exists(input_dir):
            return

        # Find video files
        video_files = glob.glob(os.path.join(input_dir, "*.mp4")) + \
                     glob.glob(os.path.join(input_dir, "*.avi")) + \
                     glob.glob(os.path.join(input_dir, "*.mkv"))

        # Find image files
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.png")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg"))

        # Process a subset of files each cycle
        for video_file in video_files[:1]:  # Process one video at a time
            self._process_video_file(video_file)

        for image_file in image_files[:10]:  # Process up to 10 images
            self._process_image_file(image_file)

    def _process_video_file(self, video_path: str):
        """Process a video file for learning."""
        import cv2

        self.logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {video_path}")
            return

        visual_config = self.config.get_visual_config()
        target_size = (visual_config["input_width"], visual_config["input_height"])

        frame_count = 0
        max_frames = int(os.environ.get('ATLAS_MAX_VIDEO_FRAMES', '1000'))

        while cap.isOpened() and frame_count < max_frames and not self.shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            frame = cv2.resize(frame, target_size)
            if visual_config["use_grayscale"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Generate synthetic audio (since video may not have audio or we're using headless)
            audio_chunk = np.random.randn(1024) * 0.01  # Low noise audio

            # Process
            self.process_frame(frame, audio_chunk, learn=True)
            frame_count += 1

        cap.release()

        # Move processed file to archive
        archive_dir = os.path.join(os.path.dirname(video_path), "processed")
        os.makedirs(archive_dir, exist_ok=True)
        new_path = os.path.join(archive_dir, os.path.basename(video_path))
        try:
            os.rename(video_path, new_path)
            self.logger.info(f"Archived processed video to: {new_path}")
        except Exception as e:
            self.logger.warning(f"Could not archive video: {e}")

    def _process_image_file(self, image_path: str):
        """Process an image file for learning."""
        import cv2

        visual_config = self.config.get_visual_config()
        target_size = (visual_config["input_width"], visual_config["input_height"])

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return

            frame = cv2.resize(frame, target_size)
            if visual_config["use_grayscale"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Generate synthetic audio
            audio_chunk = np.random.randn(1024) * 0.01

            # Process multiple times for better learning
            for _ in range(3):
                self.process_frame(frame, audio_chunk, learn=True)

            # Archive processed image
            archive_dir = os.path.join(os.path.dirname(image_path), "processed")
            os.makedirs(archive_dir, exist_ok=True)
            new_path = os.path.join(archive_dir, os.path.basename(image_path))
            os.rename(image_path, new_path)

        except Exception as e:
            self.logger.warning(f"Error processing image {image_path}: {e}")

    def _run_self_improvement_cycle(self):
        """Run a self-improvement cycle using the unified intelligence."""
        if not self.unified_intelligence:
            return

        try:
            # Only run periodically
            if self.total_frames % 1000 != 0:
                return

            self.logger.info("Running self-improvement cycle...")

            # Trigger self-improvement
            if hasattr(self.unified_intelligence, 'self_improvement'):
                improvement_result = self.unified_intelligence.self_improvement.optimize()
                if improvement_result:
                    self.metrics.increment_self_improvement()
                    self.logger.info(f"Self-improvement cycle completed: {improvement_result}")
        except Exception as e:
            self.logger.debug(f"Self-improvement error: {e}")

    def _exploration_cycle(self):
        """Generate and learn from synthetic exploration data."""
        # Only explore periodically
        if self.total_frames % 100 != 0:
            return

        visual_config = self.config.get_visual_config()

        # Generate diverse synthetic patterns for exploration
        patterns = [
            # Edges and gradients
            self._generate_gradient_pattern(visual_config),
            # Noise patterns (texture learning)
            self._generate_noise_pattern(visual_config),
            # Geometric shapes
            self._generate_shape_pattern(visual_config),
        ]

        for pattern in patterns:
            audio_chunk = np.random.randn(1024) * 0.1
            self.process_frame(pattern, audio_chunk, learn=True)

    def _generate_gradient_pattern(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate a gradient pattern for edge learning."""
        w, h = config["input_width"], config["input_height"]
        angle = np.random.rand() * 2 * np.pi
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        gradient = ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8) * 255).astype(np.uint8)
        return gradient

    def _generate_noise_pattern(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate a noise pattern for texture learning."""
        w, h = config["input_width"], config["input_height"]
        noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        return noise

    def _generate_shape_pattern(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate a geometric shape pattern."""
        w, h = config["input_width"], config["input_height"]
        pattern = np.zeros((h, w), dtype=np.uint8)

        # Random circle
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(min(w, h) // 8, min(w, h) // 3)
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        pattern[mask] = 255

        return pattern

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        uptime = time.time() - self.start_time if self.start_time else 0

        status = {
            "status": "running" if self.running else "stopped",
            "uptime_seconds": uptime,
            "total_frames_processed": self.total_frames,
            "total_learning_cycles": self.total_learning_cycles,
            "system_frame_count": self.system.frame_count if self.system else 0,
            "has_unified_intelligence": self.unified_intelligence is not None,
        }

        # Add unified intelligence status if available
        if self.unified_intelligence:
            try:
                status["unified_intelligence_status"] = {
                    "enabled": True,
                    "modules": list(self.unified_intelligence.__dict__.keys())[:10]
                }
            except Exception:
                pass

        return status

    def start(self):
        """Start the cloud service."""
        self.logger.info("Starting Atlas Cloud Service...")

        self.running = True
        self.start_time = time.time()

        # Start HTTP server for health checks and metrics
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()
        self.threads.append(http_thread)

        # Start autonomous learning thread
        learning_thread = threading.Thread(target=self.run_autonomous_learning, daemon=True)
        learning_thread.start()
        self.threads.append(learning_thread)

        self.logger.info("Atlas Cloud Service started successfully")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Keep main thread alive
        try:
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

        self.stop()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def stop(self):
        """Stop the cloud service gracefully."""
        self.logger.info("Stopping Atlas Cloud Service...")

        self.running = False
        self.shutdown_event.set()

        # Save final checkpoint
        self._save_checkpoint(is_final=True)

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.logger.info("Atlas Cloud Service stopped")

    def _run_http_server(self):
        """Run HTTP server for health checks and metrics."""

        service = self  # Capture reference for handler

        class AtlasHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = service.get_status()
                    self.wfile.write(json.dumps(status).encode())

                elif self.path == '/metrics':
                    if HAS_PROMETHEUS:
                        self.send_response(200)
                        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                        self.end_headers()
                        self.wfile.write(generate_latest())
                    else:
                        self.send_response(503)
                        self.end_headers()

                elif self.path == '/status':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = service.get_status()
                    self.wfile.write(json.dumps(status, indent=2).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

        port = int(os.environ.get('ATLAS_HTTP_PORT', '8080'))
        server = HTTPServer(('0.0.0.0', port), AtlasHandler)
        self.logger.info(f"HTTP server listening on port {port}")

        while not self.shutdown_event.is_set():
            server.handle_request()


def main():
    """Main entry point for cloud service."""
    # Setup logging
    log_level = os.environ.get('ATLAS_LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("Main")
    logger.info("=" * 60)
    logger.info("ATLAS: Autonomously Teaching, Learning And Self-organizing")
    logger.info("Cloud Service Starting...")
    logger.info("=" * 60)

    # Get config path from environment
    config_path = os.environ.get('ATLAS_CONFIG_PATH')

    # Create and start service
    service = AtlasCloudService(config_path)
    service.start()


if __name__ == "__main__":
    main()
