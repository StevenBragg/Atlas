#!/usr/bin/env python3
"""
Live demo of the Self-Organizing Audio-Visual Learning System.

This script sets up and runs the system with default configuration,
processing input from the webcam and microphone.
"""
import os
import sys
import logging
import time
import glob
import numpy as np
import yaml
import argparse
import threading
from pathlib import Path
import warnings
import multiprocessing
import concurrent.futures
import random

# Suppress matplotlib GUI thread warnings and tight_layout warnings
warnings.filterwarnings("ignore", "Starting a Matplotlib GUI outside of the main thread")
warnings.filterwarnings("ignore", "This figure includes Axes that are not compatible with tight_layout")

# Add parent directory to path to import the package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

# Import directly from modules to avoid circular imports
from models.visual.processor import VisualProcessor
from models.audio.processor import AudioProcessor
# Import system directly from core.system instead of through core
from core.system import SelfOrganizingAVSystem
from utils.capture import AVCapture
# Import the new Tkinter monitor
from gui.tk_monitor import TkMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
try:
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "debug_av_system.log")
    
    # Add additional file handler for debug logging
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file_path}")
except (OSError, IOError, PermissionError) as e:
    # Continue without file logging if there's a permission issue
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not set up file logging: {e}. Continuing with console logging only.")

logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs

# Get the module path
module_path = Path(__file__).parent.parent.absolute()

# Add verbose argument parser
parser = argparse.ArgumentParser(description="Run live demo of self-organizing AV system")
parser.add_argument("--config", type=str, default=None, help="Path to config file")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--debug-display", action="store_true", help="Enable special debug display mode")
parser.add_argument("--no-display", action="store_true", help="Disable display and use threaded approach")
parser.add_argument("--lightweight-display", action="store_true", help="Use a lighter-weight display that updates less frequently but is more stable")
parser.add_argument("--no-checkpoint", action="store_true", help="Don't load any checkpoints, start with fresh weights")
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
parser.add_argument("--enable-rgb-control", action="store_true", help="Enable direct RGB pixel control on startup")
parser.add_argument("--rgb-learning-rate", type=float, default=0.01, help="Learning rate for RGB control")
parser.add_argument("--enable-rgb-mutation", action="store_true", help="Enable periodic RGB weight mutations for exploration")
parser.add_argument("--auto-expression", action="store_true", help="Enable automatic expression mode with RGB mutations and feedback")
parser.add_argument("--mutation-interval", type=float, default=30.0, help="Interval between mutations in seconds")
parser.add_argument("--feedback-interval", type=float, default=15.0, help="Interval between automatic feedback in seconds")
parser.add_argument("--mutation-strength", type=float, default=0.2, help="Strength of RGB mutations (0-1)")
parser.add_argument("--enhanced-rgb", action="store_true", help="Enable enhanced independent RGB channel control")
args = parser.parse_args()

# Set logging level based on verbosity
if args.verbose:
    logger.setLevel(logging.DEBUG)
    logging.getLogger("self_organizing_av_system").setLevel(logging.DEBUG)

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint file in the directory"""
    try:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            return None
        
        # Get all checkpoint files
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".pkl"):
                # Get full path for stat
                full_path = os.path.join(checkpoint_dir, f)
                # Get file modification time
                mod_time = os.path.getmtime(full_path)
                checkpoints.append((f, mod_time))
        
        # Sort by modification time (newest last)
        checkpoints.sort(key=lambda x: x[1])
        
        # Return the newest checkpoint
        return checkpoints[-1][0] if checkpoints else None
    except Exception as e:
        logging.error(f"Error finding checkpoints: {e}")
        return None

def load_checkpoint_file(checkpoint_path: str, system) -> 'SelfOrganizingAVSystem':
    """Load a checkpoint file"""
    try:
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return system
            
        # Try to load the checkpoint
        success = system.load_state(checkpoint_path)
        if success:
            logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        else:
            logging.error(f"Failed to load checkpoint (load_state returned False)")
        return system
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return system

def main():
    """Run a live demo of the self-organizing AV system."""
    logger.info("Starting Self-Organizing Audio-Visual System Live Demo")
    
    # Set up multiprocessing to use all available cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Detected {num_cores} CPU cores, configuring for parallel processing")
    
    # Set numpy to use multiple threads
    try:
        # Set OpenMP environment variables first (NumPy may use these)
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
        
        # Different versions of NumPy use different methods to control threading
        try:
            # Try modern NumPy threading API
            import numpy as np
            np.set_num_threads(num_cores)
            logger.info(f"NumPy configured to use {num_cores} threads with set_num_threads")
        except AttributeError:
            try:
                # Try with threadpool_size
                import numpy as np
                np.__config__.threadpool_size = num_cores
                logger.info(f"NumPy configured with threadpool_size = {num_cores}")
            except (AttributeError, ImportError):
                # If neither work, rely on environment variables
                logger.info("NumPy threading configured via environment variables")
        
        logger.info(f"NumPy configured to use {num_cores} threads")
        
        # Try to enable OpenMP parallelism for numerical libraries
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
        logger.info("Environment variables set for parallel numerical libraries")
    except Exception as e:
        logger.warning(f"Could not fully configure multi-threading: {e}")
    
    # Configure matplotlib for optimal performance (Keep general mpl settings for potential future use, remove monitor-specific ones)
    import matplotlib as mpl
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mpl.rcParams['agg.path.chunksize'] = 10000

    try:
        # Load configuration
        if args.config:
            config_path = args.config
        else:
            config_path = os.path.join(module_path, "config", "default_config.yaml")
        
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Make all layer sizes even smaller to maximize performance
        logger.info("Creating minimal-sized neural pathways for maximum performance")
        min_layer_sizes_visual = config["visual"].get("layer_sizes", [25, 15, 8])
        min_layer_sizes_visual = [max(4, s//2) for s in min_layer_sizes_visual]  # Reduce by half, but not below 4
        logger.info(f"Using minimal visual layer sizes: {min_layer_sizes_visual}")
        
        visual_processor = VisualProcessor(
            input_width=config["visual"].get("input_width", 32),  # Keep inputs small
            input_height=config["visual"].get("input_height", 32),
            use_grayscale=True,  # Always use grayscale for speed
            patch_size=config["visual"].get("patch_size", 16),
            stride=config["visual"].get("stride", 12),  # Large stride means fewer patches
            contrast_normalize=config["visual"].get("contrast_normalize", True),
            layer_sizes=min_layer_sizes_visual
        )
        
        min_layer_sizes_audio = config["audio"].get("layer_sizes", [20, 12, 6])
        min_layer_sizes_audio = [max(4, s//2) for s in min_layer_sizes_audio]  # Reduce by half, but not below 4
        logger.info(f"Using minimal audio layer sizes: {min_layer_sizes_audio}")
        
        audio_processor = AudioProcessor(
            sample_rate=config["audio"].get("sample_rate", 22050),
            window_size=config["audio"].get("window_size", 256),  # Smaller window
            hop_length=config["audio"].get("hop_length", 128),    # Smaller hop
            n_mels=config["audio"].get("n_mels", 12),            # Fewer mel bands
            min_freq=config["audio"].get("min_freq", 50),
            max_freq=config["audio"].get("max_freq", 8000),
            normalize=config["audio"].get("normalize", True),
            layer_sizes=min_layer_sizes_audio
        )
        
        # Create the self-organizing system
        min_multimodal_size = max(10, config["system"].get("multimodal_size", 20) // 2)  # Half size, min 10
        logger.info(f"Using minimal multimodal association size: {min_multimodal_size}")
        
        # Add RGB control parameters to system config
        system_config = {
            "multimodal_size": min_multimodal_size,
            "prune_interval": config["system"].get("prune_interval", 1000),
            "structural_plasticity_interval": config["system"].get("structural_plasticity_interval", 5000),
            "learning_rate": config["system"].get("learning_rate", 0.01),
            "learning_rule": config["system"].get("learning_rule", 'oja'),
            # RGB control config
            "rgb_learning_rate": args.rgb_learning_rate,
            "rgb_learning_enabled": True,
            "enhanced_rgb_control": args.enhanced_rgb,
        }
        
        system = SelfOrganizingAVSystem(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config=system_config
        )
        
        # Configure checkpointing
        checkpoint_config = config.get("checkpointing", {})
        enable_checkpointing = checkpoint_config.get("enabled", False)
        checkpoint_interval = checkpoint_config.get("checkpoint_interval", 1000)
        checkpoint_dir = checkpoint_config.get("checkpoint_dir", "checkpoints")
        load_latest = checkpoint_config.get("load_latest", False)
        save_on_exit = checkpoint_config.get("save_on_exit", False)
        checkpoint_max = checkpoint_config.get("max_checkpoints", 5)
        
        # Load latest checkpoint if available
        if args.no_checkpoint:
            logger.info("Skipping checkpoint loading as requested")
        elif args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            system = load_checkpoint_file(args.checkpoint, system)
        elif enable_checkpointing and load_latest:
            # Ensure checkpoint directory exists before looking for checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Find latest checkpoint with better error handling
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"Loading latest checkpoint: {checkpoint_path}")
                system = load_checkpoint_file(checkpoint_path, system)
            else:
                logger.info("No checkpoints found, starting fresh")
                
            # Force load the actual latest checkpoint if available (failsafe check)
            if not args.no_checkpoint:
                checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')])
                if checkpoints and not latest_checkpoint:
                    # If find_latest_checkpoint failed but files exist
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                    logger.info(f"Loading fallback checkpoint: {checkpoint_path}")
                    system = load_checkpoint_file(checkpoint_path, system)
        
        # Setup AV capture
        logger.info("Setting up audio-video capture...")
        capture_params = config.get("capture", {})
        
        # Added debug print for camera initialization
        logger.debug(f"Camera parameters: width={capture_params.get('video_width', 640)}, " +
                     f"height={capture_params.get('video_height', 480)}, " +
                     f"fps={capture_params.get('fps', 30)}")
                     
        capture = AVCapture(
            video_width=capture_params.get("video_width", 640),
            video_height=capture_params.get("video_height", 480),
            fps=capture_params.get("fps", 30),
            sample_rate=capture_params.get("sample_rate", 22050),
            channels=capture_params.get("audio_channels", capture_params.get("channels", 1)),
            chunk_size=capture_params.get("chunk_size", 1024),
            device_index=capture_params.get("device_index", None)
        )
        
        # Start capture with better error handling
        if not capture.start():
            logger.error("Failed to start audio-video capture. Check camera and microphone.")
            
            # Testing if camera devices are available
            logger.info("Trying to list available camera devices...")
            import cv2
            for i in range(5):  # Check first 5 camera indices
                test_cap = cv2.VideoCapture(i)
                is_opened = test_cap.isOpened()
                test_cap.release()
                logger.info(f"Camera index {i}: {'available' if is_opened else 'not available'}")
                
            # Continue with test patterns since the monitor will handle it
            logger.info("Continuing with test patterns for debugging")
        else:
            logger.info("Audio-video capture started successfully")
            
            # Verify frame capture is working
            test_frame, test_audio = capture.get_av_pair(block=True, timeout=2.0)
            if test_frame is not None:
                logger.info(f"Successfully captured test frame: {test_frame.shape}, {test_frame.dtype}")
                logger.debug(f"Frame range: min={np.min(test_frame)}, max={np.max(test_frame)}, mean={np.mean(test_frame):.2f}")
            else:
                logger.warning("Failed to capture test frame, may use test patterns")
                test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
                
            if test_audio is not None:
                logger.info(f"Successfully captured test audio: {test_audio.shape}, {test_audio.dtype}")
                logger.debug(f"Audio range: min={np.min(test_audio):.4f}, max={np.max(test_audio):.4f}, mean={np.mean(test_audio):.4f}")
            else:
                logger.warning("Failed to capture test audio, may use test patterns")
                test_audio = np.zeros(1024, dtype=np.float32)
        
        # Initialize direct RGB pixel control - enable by default for better expressiveness
        enable_rgb_control = args.enable_rgb_control or args.auto_expression
        if enable_rgb_control:
            logger.info("Enabling direct RGB pixel control")
            system.set_direct_pixel_control(True)
        
        # Set enhanced RGB control mode - enable by default for auto-expression
        enhanced_rgb = args.enhanced_rgb or args.auto_expression
        if enhanced_rgb and not system.enhanced_rgb_control:
            system.enhanced_rgb_control = True
            logger.info("Enabling enhanced independent RGB channel control")
        
        # Configure auto-expression parameters
        auto_expression = args.auto_expression
        mutation_interval = args.mutation_interval
        feedback_interval = args.feedback_interval
        mutation_strength = args.mutation_strength
        
        if auto_expression:
            logger.info(f"Auto-expression mode enabled with mutation interval={mutation_interval}s, " +
                       f"feedback interval={feedback_interval}s, mutation strength={mutation_strength}")
        
        # Start the new Tkinter monitor if display is enabled
        monitor = None
        if not args.no_display:
            # Ensure system has initial state that can be queried by the monitor
            logger.info("Initializing system with test data...")
            system.process_av_pair(
                test_frame if test_frame is not None else np.zeros((64, 64, 3), dtype=np.uint8),
                test_audio if test_audio is not None else np.zeros(1024, dtype=np.float32),
                learn=False  # Don't learn from initial test data
            )

            logger.info("Starting Tkinter monitor...")
            try:
                # Configure a faster update interval for better responsiveness
                update_interval_ms = 200 if args.lightweight_display else 100
                monitor = TkMonitor(system=system, update_interval_ms=update_interval_ms) # Pass the system instance
                monitor.start() # Start the GUI thread
                logger.info("Tkinter monitor started.")
            except Exception as e:
                logger.error(f"Failed to start Tkinter monitor: {e}", exc_info=True)
                monitor = None # Ensure monitor is None if it failed
        else:
            logger.info("Display disabled, not starting monitor.")

        # Define the main AV processing loop
        def processing_loop():
            frame_count = 0
            start_time = time.time()
            checkpoint_count = 0
            last_status_time = start_time
            last_mutation_time = start_time
            last_feedback_time = start_time
            
            # Keep track of the current aesthetic direction
            current_aesthetic = 0.0  # Value between -1.0 and 1.0 representing preferred aesthetic
            aesthetic_momentum = 0.0  # Directional momentum to create more continuous patterns
            
            try:
                while capture.running:
                    # Wait for a synchronized AV pair
                    frame, audio = capture.get_av_pair(block=True, timeout=capture.frame_interval * 2)
                    if frame is None or audio is None:
                        continue
                    
                    # Process and learn (using process_av_pair which uses the correct processing methods inside)
                    _ = system.process_av_pair(frame, audio, learn=True)
                    frame_count += 1
                    
                    # Apply auto-expression if enabled
                    current_time = time.time()
                    
                    # Apply RGB mutations for exploration if enabled
                    if (args.enable_rgb_mutation or auto_expression) and system.direct_pixel_control:
                        if current_time - last_mutation_time >= mutation_interval:
                            # Calculate mutation rate for visible changes
                            if auto_expression:
                                # More controlled mutation in auto mode
                                mutation_rate = mutation_strength * (0.5 + 0.5 * random.random())  
                            else:
                                # More random mutation in manual mode
                                mutation_rate = 0.2 * random.random() + 0.05  # Random between 5-25%
                                
                            system.apply_rgb_mutation(mutation_rate)
                            logger.info(f"Applied RGB mutation with rate {mutation_rate:.3f}")
                            last_mutation_time = current_time
                    
                    # Apply feedback in auto-expression mode
                    if auto_expression and system.direct_pixel_control:
                        if current_time - last_feedback_time >= feedback_interval:
                            # Update the aesthetic direction with some randomness but following momentum
                            aesthetic_momentum = 0.7 * aesthetic_momentum + 0.3 * (random.random() * 2.0 - 1.0)
                            current_aesthetic = 0.7 * current_aesthetic + 0.3 * aesthetic_momentum
                            current_aesthetic = max(-1.0, min(1.0, current_aesthetic))  # Clamp
                            
                            # Apply feedback using the current aesthetic direction
                            feedback = current_aesthetic * (0.6 + 0.4 * random.random())  # Scale with some randomness
                            system.learn_from_rgb_feedback(feedback)
                            logger.info(f"Applied auto RGB feedback: {feedback:.2f} (aesthetic: {current_aesthetic:.2f})")
                            last_feedback_time = current_time
                            
                            # Occasional strong reinforcement to create more distinct patterns
                            if random.random() < 0.4:  # 40% chance
                                strong_feedback = 0.8 if current_aesthetic > 0 else -0.8
                                system.learn_from_rgb_feedback(strong_feedback * 0.5)  # Half strength to avoid overshooting
                                logger.info(f"Applied strong RGB feedback: {strong_feedback:.2f}")
                    
                    # Report FPS more frequently to track performance
                    if current_time - last_status_time >= 2.0:  # Update every 2 seconds
                        elapsed = current_time - start_time
                        fps = frame_count / elapsed
                        logger.info(f"Processed {frame_count} frames ({fps:.2f} FPS)")
                        last_status_time = current_time
                    
                    # Checkpoint interval
                    if enable_checkpointing and frame_count % checkpoint_interval == 0:
                        # Make sure checkpoint directory exists
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        cp = os.path.join(checkpoint_dir, f"checkpoint_{frame_count}.pkl")
                        try:
                            system.save_state(cp)
                            logger.info(f"Saved checkpoint to {cp}")
                            checkpoint_count += 1
                            if checkpoint_count > checkpoint_max:
                                cps = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')])
                                if len(cps) > checkpoint_max:
                                    old = os.path.join(checkpoint_dir, cps[0])
                                    os.remove(old)
                                    logger.info(f"Removed oldest checkpoint: {old}")
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint to {cp}: {e}")
                            # Try to create directory again and retry once
                            try:
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                system.save_state(cp)
                                logger.info(f"Successfully saved checkpoint after retry")
                            except Exception as retry_error:
                                logger.error(f"Retry failed: {retry_error}")
            except KeyboardInterrupt:
                logger.info("Processing loop interrupted")

        # Start processing in background
        proc_thread = threading.Thread(target=processing_loop)
        proc_thread.daemon = True
        proc_thread.start()
        
        # Keep the application alive until capture stops or user interrupts
        try:
            # Just wait for processing to complete or user to interrupt
            while capture.running:
                time.sleep(0.1)  # Simple wait, GUI updates on its own thread
        except KeyboardInterrupt:
            logger.info("User requested shutdown")
        
        # Cleanup
        logger.info("Shutting down...")

        # Stop the monitor if it was started
        if monitor is not None:
            try:
                logger.info("Stopping monitor...")
                monitor.stop()
                logger.info("Monitor stopped.")
            except Exception as e:
                logger.error(f"Error stopping monitor: {e}", exc_info=True)

        # Stop capture
        try:
            logger.info("Stopping capture...")
            capture.stop()
            logger.info("Capture stopped.")
        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)
        
        try:
            # Wait with timeout to avoid hanging
            if proc_thread.is_alive():
                logger.info("Waiting for processing thread to finish...")
                proc_thread.join(timeout=3.0)
                if proc_thread.is_alive():
                     logger.warning("Processing thread did not finish cleanly.")
                else:
                    logger.info("Processing thread finished.")
        except Exception as e:
            logger.error(f"Error joining processing thread: {e}", exc_info=True)
        
        # Save checkpoint on exit if configured
        if enable_checkpointing and save_on_exit:
            os.makedirs(checkpoint_dir, exist_ok=True)
            try:
                cp_path = os.path.join(checkpoint_dir, f"checkpoint_exit_{int(time.time())}.pkl")
                system.save_state(cp_path)
                logger.info(f"Saved checkpoint on exit to {cp_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint on exit: {e}")
        logger.info("Demo complete")
        
    except Exception as e:
        logger.exception(f"Error in demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 