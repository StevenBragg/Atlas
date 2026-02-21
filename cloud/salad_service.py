#!/usr/bin/env python3
"""
Atlas Salad Cloud Service

Optimized service for running Atlas on Salad Cloud's distributed GPU network.
This service enables Atlas to:
1. Utilize GPU acceleration for faster learning
2. Run continuously on Salad Cloud's distributed nodes
3. Persist learned knowledge across node migrations
4. Self-heal and recover from node interruptions
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

# Check for GPU availability using CuPy backend (preferred) or torch fallback
try:
    from core.backend import HAS_GPU, get_backend_info
    backend_info = get_backend_info()
    HAS_CUDA = HAS_GPU
    GPU_NAME = backend_info.get('device_name', 'GPU') if HAS_GPU else 'CPU'
    GPU_MEMORY = backend_info.get('device_memory_gb', 0) if HAS_GPU else 0
except ImportError:
    # Fall back to torch for GPU detection
    try:
        import torch
        HAS_CUDA = torch.cuda.is_available()
        if HAS_CUDA:
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            GPU_NAME = "CPU"
            GPU_MEMORY = 0
    except ImportError:
        HAS_CUDA = False
        GPU_NAME = "CPU"
        GPU_MEMORY = 0

# Add parent directory and self_organizing_av_system to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_soav_dir = os.path.join(_parent_dir, 'self_organizing_av_system')
# Insert soav_dir first so its config/ package is preferred over root-level config/
if _soav_dir not in sys.path:
    sys.path.insert(0, _soav_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

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

# Try to import text learning
try:
    from core.text_learning import TextLearningModule
    from cloud.text_api import TextLearningAPI, create_text_handler
    HAS_TEXT_LEARNING = True
except ImportError:
    HAS_TEXT_LEARNING = False

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class SaladCloudMetrics:
    """Prometheus metrics optimized for Salad Cloud monitoring."""

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
            self.gpu_utilization = Gauge(
                'atlas_gpu_utilization_percent',
                'GPU utilization percentage'
            )
            self.gpu_memory_used = Gauge(
                'atlas_gpu_memory_used_gb',
                'GPU memory used in GB'
            )
            self.processing_time = Histogram(
                'atlas_processing_time_seconds',
                'Time to process each frame',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            )
            self.intelligence_score = Gauge(
                'atlas_intelligence_score',
                'Current unified intelligence quotient'
            )
            self.node_uptime = Gauge(
                'atlas_node_uptime_seconds',
                'Time since service started on this node'
            )

    def update_gpu_metrics(self):
        """Update GPU metrics from PyTorch."""
        if HAS_PROMETHEUS and HAS_CUDA:
            try:
                import torch
                # Get GPU utilization (approximation based on memory)
                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                utilization = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0

                self.gpu_utilization.set(utilization)
                self.gpu_memory_used.set(mem_allocated)
            except Exception:
                pass


class AtlasSaladService:
    """
    Main service for running Atlas on Salad Cloud.

    Optimizations for Salad Cloud:
    - GPU-accelerated processing when available
    - Graceful handling of node preemption
    - Automatic checkpoint save before shutdown
    - Health checks for Container Gateway
    - Metrics for monitoring
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("AtlasSaladService")

        # Salad Cloud metadata
        self.salad_machine_id = os.environ.get('SALAD_MACHINE_ID', 'unknown')
        self.salad_container_group_id = os.environ.get('SALAD_CONTAINER_GROUP_ID', 'unknown')

        self.logger.info(f"Starting Atlas on Salad Cloud")
        self.logger.info(f"Machine ID: {self.salad_machine_id}")
        self.logger.info(f"Container Group: {self.salad_container_group_id}")
        self.logger.info(f"GPU Available: {HAS_CUDA}")
        if HAS_CUDA:
            self.logger.info(f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")

        # Load configuration
        self.config = SystemConfig(config_path)
        self._apply_salad_config()

        # Initialize metrics
        self.metrics = SaladCloudMetrics()

        # State tracking
        self.running = False
        self.start_time = None
        self.total_frames = 0
        self.ready = False  # For readiness probe

        # Initialize the system
        self._initialize_system()

        # Thread management
        self.threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

        self.logger.info("Atlas Salad Service initialized")

    def _apply_salad_config(self):
        """Apply Salad Cloud optimized configuration."""
        # Checkpoint settings for distributed environment
        checkpoint_dir = os.environ.get('ATLAS_CHECKPOINT_DIR', '/data/checkpoints')
        self.config.update("checkpointing", "checkpoint_dir", checkpoint_dir)
        self.config.update("checkpointing", "enabled", True)
        self.config.update("checkpointing", "load_latest", True)

        # More frequent checkpoints on Salad (nodes can be preempted)
        self.config.update("checkpointing", "checkpoint_interval",
                          int(os.environ.get('ATLAS_CHECKPOINT_INTERVAL', '500')))
        self.config.update("checkpointing", "max_checkpoints",
                          int(os.environ.get('ATLAS_MAX_CHECKPOINTS', '3')))

        # GPU-optimized settings
        if HAS_CUDA:
            # Larger batch sizes for GPU
            self.config.update("system", "multimodal_size",
                             int(os.environ.get('ATLAS_MULTIMODAL_SIZE', '256')))
        else:
            self.config.update("system", "multimodal_size",
                             int(os.environ.get('ATLAS_MULTIMODAL_SIZE', '128')))

        # Learning parameters
        self.config.update("system", "learning_rate",
                          float(os.environ.get('ATLAS_LEARNING_RATE', '0.01')))

    def _initialize_system(self):
        """Initialize the Atlas system with GPU support if available."""
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

        # Initialize text learning module if available
        self.text_module = None
        self.text_api = None
        if HAS_TEXT_LEARNING and os.environ.get('ATLAS_ENABLE_TEXT_LEARNING', 'true').lower() == 'true':
            try:
                self.text_module = TextLearningModule()
                self.text_api = TextLearningAPI(self.text_module)
                self.logger.info("Text Learning Module enabled")
            except Exception as e:
                self.logger.warning(f"Could not initialize text learning module: {e}")

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

        # Load text learning checkpoint if available
        if self.text_module:
            text_checkpoint = os.path.join(checkpoint_dir, "text_learning_state.pkl")
            if os.path.exists(text_checkpoint):
                try:
                    if self.text_module.load_state(text_checkpoint):
                        self.logger.info(f"Loaded text learning checkpoint")
                    else:
                        self.logger.warning("Failed to load text learning checkpoint")
                except Exception as e:
                    self.logger.error(f"Error loading text learning checkpoint: {e}")

    def _save_checkpoint(self, is_final: bool = False):
        """Save a checkpoint of the current state."""
        checkpoint_config = self.config.get_checkpointing_config()
        checkpoint_dir = checkpoint_config["checkpoint_dir"]

        os.makedirs(checkpoint_dir, exist_ok=True)

        prefix = "final" if is_final else "checkpoint"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(
            checkpoint_dir,
            f"{prefix}_{self.salad_machine_id}_{timestamp}_{self.system.frame_count}.pkl"
        )

        success = False
        try:
            if self.system.save_state(checkpoint_file):
                self.logger.info(f"Saved checkpoint: {checkpoint_file}")
                self._cleanup_old_checkpoints(checkpoint_config)
                success = True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

        # Save text learning checkpoint if available
        if self.text_module:
            text_checkpoint_file = os.path.join(checkpoint_dir, "text_learning_state.pkl")
            try:
                if self.text_module.save_state(text_checkpoint_file):
                    self.logger.info(f"Saved text learning checkpoint")
                else:
                    self.logger.warning("Failed to save text learning checkpoint")
            except Exception as e:
                self.logger.error(f"Error saving text learning checkpoint: {e}")

        return success

    def _cleanup_old_checkpoints(self, checkpoint_config: Dict[str, Any]):
        """Remove old checkpoints keeping only max_checkpoints most recent."""
        import glob
        checkpoint_dir = checkpoint_config["checkpoint_dir"]
        max_checkpoints = checkpoint_config.get("max_checkpoints", 3)

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
        """Process a single audio-visual frame with GPU acceleration."""
        start_time = time.time()

        # Process through main system
        result = self.system.process_av_pair(visual_data, audio_data, learn=learn)

        # Process through unified intelligence if available
        if self.unified_intelligence and learn:
            try:
                flat_visual = visual_data.flatten()
                ui_result = self.unified_intelligence.process(flat_visual)
                result['unified_intelligence'] = ui_result

                if HAS_PROMETHEUS and 'intelligence_quotient' in ui_result:
                    self.metrics.intelligence_score.set(ui_result['intelligence_quotient'])
            except Exception as e:
                self.logger.debug(f"Unified intelligence processing error: {e}")

        # Update metrics
        processing_time = time.time() - start_time
        if HAS_PROMETHEUS:
            self.metrics.frames_processed.inc()
            self.metrics.processing_time.observe(processing_time)
            self.metrics.update_gpu_metrics()

        self.total_frames += 1

        # Periodic checkpoint (more frequent on Salad)
        checkpoint_interval = self.config.get_checkpointing_config()["checkpoint_interval"]
        if self.total_frames % checkpoint_interval == 0:
            self._save_checkpoint()
            if HAS_PROMETHEUS:
                self.metrics.learning_cycles.inc()

        return result

    def run_autonomous_learning(self):
        """Run autonomous learning loop optimized for Salad Cloud."""
        self.logger.info("Starting autonomous learning loop...")

        input_dir = os.environ.get('ATLAS_INPUT_DIR', '/data/input')

        while not self.shutdown_event.is_set():
            try:
                # Process input files if available
                self._process_input_directory(input_dir)

                # Run self-improvement cycles
                if self.unified_intelligence and self.total_frames % 1000 == 0:
                    self._run_self_improvement_cycle()

                # Generate exploration data
                if os.environ.get('ATLAS_ENABLE_EXPLORATION', 'true').lower() == 'true':
                    self._exploration_cycle()

                time.sleep(0.01)  # Small delay, GPU can handle fast processing

            except Exception as e:
                self.logger.error(f"Error in autonomous learning: {e}")
                time.sleep(1.0)

    def _process_input_directory(self, input_dir: str):
        """Process any files in the input directory."""
        import glob
        import cv2

        if not os.path.exists(input_dir):
            return

        # Find video files
        video_files = glob.glob(os.path.join(input_dir, "*.mp4")) + \
                     glob.glob(os.path.join(input_dir, "*.avi"))

        # Find image files
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.png"))

        for video_file in video_files[:1]:
            self._process_video_file(video_file)

        for image_file in image_files[:20]:  # Process more images with GPU
            self._process_image_file(image_file)

    def _process_video_file(self, video_path: str):
        """Process a video file with GPU acceleration."""
        import cv2

        self.logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        visual_config = self.config.get_visual_config()
        target_size = (visual_config["input_width"], visual_config["input_height"])

        frame_count = 0
        max_frames = int(os.environ.get('ATLAS_MAX_VIDEO_FRAMES', '5000'))

        while cap.isOpened() and frame_count < max_frames and not self.shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, target_size)
            if visual_config["use_grayscale"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            audio_chunk = np.random.randn(1024) * 0.01
            self.process_frame(frame, audio_chunk, learn=True)
            frame_count += 1

        cap.release()

        # Archive processed file
        archive_dir = os.path.join(os.path.dirname(video_path), "processed")
        os.makedirs(archive_dir, exist_ok=True)
        try:
            os.rename(video_path, os.path.join(archive_dir, os.path.basename(video_path)))
        except Exception:
            pass

    def _process_image_file(self, image_path: str):
        """Process an image file."""
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

            audio_chunk = np.random.randn(1024) * 0.01

            # Process multiple times for better learning
            for _ in range(3):
                self.process_frame(frame, audio_chunk, learn=True)

            # Archive
            archive_dir = os.path.join(os.path.dirname(image_path), "processed")
            os.makedirs(archive_dir, exist_ok=True)
            os.rename(image_path, os.path.join(archive_dir, os.path.basename(image_path)))

        except Exception as e:
            self.logger.warning(f"Error processing image {image_path}: {e}")

    def _run_self_improvement_cycle(self):
        """Run self-improvement cycle."""
        if not self.unified_intelligence:
            return

        try:
            self.logger.info("Running self-improvement cycle...")
            if hasattr(self.unified_intelligence, 'self_improvement'):
                self.unified_intelligence.self_improvement.optimize()
        except Exception as e:
            self.logger.debug(f"Self-improvement error: {e}")

    def _exploration_cycle(self):
        """Generate synthetic data for exploration."""
        if self.total_frames % 50 != 0:  # More frequent with GPU
            return

        visual_config = self.config.get_visual_config()
        w, h = visual_config["input_width"], visual_config["input_height"]

        # Generate diverse patterns
        patterns = [
            np.random.randint(0, 256, (h, w), dtype=np.uint8),  # Noise
            self._gradient_pattern(w, h),  # Gradient
            self._shape_pattern(w, h),  # Shapes
        ]

        for pattern in patterns:
            audio_chunk = np.random.randn(1024) * 0.1
            self.process_frame(pattern, audio_chunk, learn=True)

    def _gradient_pattern(self, w: int, h: int) -> np.ndarray:
        """Generate gradient pattern."""
        angle = np.random.rand() * 2 * np.pi
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        return ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8) * 255).astype(np.uint8)

    def _shape_pattern(self, w: int, h: int) -> np.ndarray:
        """Generate shape pattern."""
        pattern = np.zeros((h, w), dtype=np.uint8)
        cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
        radius = np.random.randint(min(w, h)//8, min(w, h)//3)
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        pattern[mask] = 255
        return pattern

    def get_status(self) -> Dict[str, Any]:
        """Get current service status for health checks."""
        uptime = time.time() - self.start_time if self.start_time else 0

        status = {
            "status": "running" if self.running else "stopped",
            "ready": self.ready,
            "uptime_seconds": uptime,
            "total_frames_processed": self.total_frames,
            "system_frame_count": self.system.frame_count if self.system else 0,
            "gpu": {
                "available": HAS_CUDA,
                "name": GPU_NAME,
                "memory_gb": GPU_MEMORY
            },
            "salad_cloud": {
                "machine_id": self.salad_machine_id,
                "container_group_id": self.salad_container_group_id
            },
            "unified_intelligence": self.unified_intelligence is not None,
            "text_learning": self.text_module.get_stats() if self.text_module else None
        }

        return status

    def start(self):
        """Start the Salad Cloud service."""
        self.logger.info("Starting Atlas Salad Cloud Service...")

        self.running = True
        self.start_time = time.time()

        # Start HTTP server for health checks (required for Salad Container Gateway)
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()
        self.threads.append(http_thread)

        # Start autonomous learning thread
        learning_thread = threading.Thread(target=self.run_autonomous_learning, daemon=True)
        learning_thread.start()
        self.threads.append(learning_thread)

        # Mark as ready for Salad Cloud readiness probe
        self.ready = True

        self.logger.info("Atlas Salad Cloud Service started successfully")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Keep main thread alive
        try:
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1.0)
                # Update uptime metric
                if HAS_PROMETHEUS:
                    self.metrics.node_uptime.set(time.time() - self.start_time)
        except KeyboardInterrupt:
            pass

        self.stop()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals - important for Salad Cloud node preemption."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def stop(self):
        """Stop the service gracefully - save checkpoint before shutdown."""
        self.logger.info("Stopping Atlas Salad Cloud Service...")

        self.running = False
        self.ready = False
        self.shutdown_event.set()

        # CRITICAL: Save checkpoint before shutdown (Salad nodes can be preempted)
        self.logger.info("Saving final checkpoint before shutdown...")
        self._save_checkpoint(is_final=True)

        # Wait for threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.logger.info("Atlas Salad Cloud Service stopped")

    def _run_http_server(self):
        """Run HTTP server for Salad Cloud Container Gateway."""

        service = self

        class SaladHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == '/health':
                    # Liveness probe
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = service.get_status()
                    self.wfile.write(json.dumps(status).encode())

                elif self.path == '/ready':
                    # Readiness probe for Salad Container Gateway
                    if service.ready:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"ready": True}).encode())
                    else:
                        self.send_response(503)
                        self.end_headers()

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
        server = HTTPServer(('0.0.0.0', port), SaladHandler)
        self.logger.info(f"HTTP server listening on port {port}")

        while not self.shutdown_event.is_set():
            server.handle_request()


def main():
    """Main entry point for Salad Cloud service."""
    log_level = os.environ.get('ATLAS_LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("Main")
    logger.info("=" * 60)
    logger.info("ATLAS: Autonomously Teaching, Learning And Self-organizing")
    logger.info("Salad Cloud GPU-Accelerated Service")
    logger.info("=" * 60)

    config_path = os.environ.get('ATLAS_CONFIG_PATH')

    service = AtlasSaladService(config_path)
    service.start()


if __name__ == "__main__":
    main()
