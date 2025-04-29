import numpy as np
import logging
import threading
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import datetime

logger = logging.getLogger(__name__)


class NetworkMonitor:
    """
    Monitor for tracking and visualizing the self-organizing system's state.
    
    This class provides:
    1. Real-time tracking of system performance metrics
    2. Visualization of neural activity, connections, and learning progress
    3. Periodic snapshots of system state for later analysis
    4. Basic UI for interacting with the system
    """
    
    def __init__(
        self,
        system,
        config: Optional[Dict[str, Any]] = None,
        enable_visualization: bool = True,
        update_interval: float = 0.5,
        snapshot_dir: str = "./snapshots",
        max_history_length: int = 1000,
    ):
        """
        Initialize the network monitor.
        
        Args:
            system: The self-organizing system to monitor
            config: Monitor configuration
            enable_visualization: Whether to enable visualization
            update_interval: Interval between visualization updates (seconds)
            snapshot_dir: Directory to save snapshots
            max_history_length: Maximum length of history to keep
        """
        self.system = system
        
        # Set default configuration if not provided
        if config is None:
            config = {}
        
        self.config = config
        self.enable_visualization = enable_visualization
        self.update_interval = update_interval
        self.snapshot_dir = snapshot_dir
        self.max_history_length = max_history_length
        
        # Create snapshot directory if needed
        if self.config.get("save_snapshots", True):
            os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.history = {
            "timestamps": [],
            "visual_errors": [],
            "audio_errors": [],
            "cross_modal_errors": [],
            "prediction_errors": [],
            "surprise_signals": [],
            "stability_measures": [],
            "structural_changes": [],
            "neuron_activity": [],
            "processing_times": [],
        }
        
        # Visualization state
        self.figures = {}
        self.axes = {}
        self.visualization_thread = None
        self.stop_flag = threading.Event()
        
        # Last update time
        self.last_update_time = None
        self.last_snapshot_time = None
        
        logger.info(f"Initialized network monitor with update interval {update_interval}s")
    
    def start(self, non_blocking: bool = True):
        """
        Start the monitoring thread.
        
        Args:
            non_blocking: Whether to run in a separate thread (non-blocking)
        """
        if self.visualization_thread is not None and self.visualization_thread.is_alive():
            logger.warning("Monitor is already running")
            return
        
        # Reset stop flag
        self.stop_flag.clear()
        
        if non_blocking:
            # Start in a separate thread
            self.visualization_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.visualization_thread.start()
            logger.info("Started monitor thread")
        else:
            # Run in current thread
            self._monitor_loop()
    
    def stop(self):
        """Stop the monitoring thread."""
        if self.visualization_thread is None or not self.visualization_thread.is_alive():
            logger.warning("Monitor is not running")
            return
        
        # Set stop flag
        self.stop_flag.set()
        
        # Wait for thread to end
        self.visualization_thread.join(timeout=2.0)
        
        logger.info("Stopped monitor thread")
    
    def update(self, result: Optional[Dict[str, Any]] = None):
        """
        Update the monitor with the latest processing result.
        
        Args:
            result: Dictionary with processing results (optional)
        """
        # Skip if visualization is disabled
        if not self.enable_visualization:
            return
        
        # Get current time
        current_time = time.time()
        
        # Get system stats if result not provided
        if result is None:
            # Get stats directly from system
            stats = self.system.get_stats()
            
            # Create minimal result dictionary
            result = {
                "visual_result": {},
                "audio_result": {},
                "reconstruction_errors": {
                    "visual": 0.0,
                    "audio": 0.0
                },
                "surprise": 0.0,
                "processing_time": 0.0
            }
        else:
            # Get stats from system
            stats = self.system.get_stats()
        
        # Update history
        self.history["timestamps"].append(current_time)
        
        # Extract metrics from result
        visual_error = result.get("reconstruction_errors", {}).get("visual", 0.0)
        audio_error = result.get("reconstruction_errors", {}).get("audio", 0.0)
        cross_modal_error = np.mean([visual_error, audio_error])
        prediction_error = result.get("prediction_error", 0.0)
        surprise = result.get("surprise", 0.0)
        processing_time = result.get("processing_time", 0.0)
        
        # Check component stats
        multimodal_stats = stats.get("components", {}).get("multimodal_association", {})
        stability_stats = stats.get("components", {}).get("stability", {})
        plasticity_stats = stats.get("components", {}).get("structural_plasticity", {})
        
        # Stability measure: target vs. actual activity
        target_activity = stability_stats.get("target_activity", 0.1)
        current_activity = stability_stats.get("current_activity", 0.0)
        stability_measure = abs(target_activity - current_activity) 
        
        # Get structural changes
        structural_changes = {
            "neurons_added": plasticity_stats.get("total_neurons_added", 0),
            "connections_pruned": plasticity_stats.get("total_connections_pruned", 0),
            "connections_sprouted": plasticity_stats.get("total_connections_sprouted", 0),
            "active_neuron_fraction": plasticity_stats.get("active_neuron_fraction", 0.0)
        }
        
        # Add metrics to history
        self.history["visual_errors"].append(visual_error)
        self.history["audio_errors"].append(audio_error)
        self.history["cross_modal_errors"].append(cross_modal_error)
        self.history["prediction_errors"].append(prediction_error)
        self.history["surprise_signals"].append(surprise)
        self.history["stability_measures"].append(stability_measure)
        self.history["structural_changes"].append(structural_changes)
        self.history["processing_times"].append(processing_time)
        
        # Limit history length
        if len(self.history["timestamps"]) > self.max_history_length:
            self.history["timestamps"] = self.history["timestamps"][-self.max_history_length:]
            self.history["visual_errors"] = self.history["visual_errors"][-self.max_history_length:]
            self.history["audio_errors"] = self.history["audio_errors"][-self.max_history_length:]
            self.history["cross_modal_errors"] = self.history["cross_modal_errors"][-self.max_history_length:]
            self.history["prediction_errors"] = self.history["prediction_errors"][-self.max_history_length:]
            self.history["surprise_signals"] = self.history["surprise_signals"][-self.max_history_length:]
            self.history["stability_measures"] = self.history["stability_measures"][-self.max_history_length:]
            self.history["structural_changes"] = self.history["structural_changes"][-self.max_history_length:]
            self.history["processing_times"] = self.history["processing_times"][-self.max_history_length:]
        
        # Update last update time
        self.last_update_time = current_time
        
        # Take a snapshot if needed
        if (self.config.get("save_snapshots", True) and 
            (self.last_snapshot_time is None or 
             current_time - self.last_snapshot_time > self.config.get("snapshot_interval", 60))):
            self.take_snapshot()
            self.last_snapshot_time = current_time
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            # Initialize visualization
            if self.enable_visualization:
                self._setup_visualization()
            
            # Loop until stopped
            while not self.stop_flag.is_set():
                # Update visualization
                if self.enable_visualization:
                    self._update_visualization()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
        finally:
            if self.enable_visualization:
                self._cleanup_visualization()
    
    def _setup_visualization(self):
        """Set up visualization figures."""
        # Close any existing figures
        plt.close('all')
        
        # Create main performance figure
        self.figures["performance"] = plt.figure(figsize=(12, 10))
        self.figures["performance"].suptitle("Self-Organizing AV System Performance", fontsize=16)
        
        gs = self.figures["performance"].add_gridspec(3, 2)
        
        # Reconstruction errors
        self.axes["reconstruction"] = self.figures["performance"].add_subplot(gs[0, 0])
        self.axes["reconstruction"].set_title("Reconstruction Errors")
        self.axes["reconstruction"].set_xlabel("Frames")
        self.axes["reconstruction"].set_ylabel("Error")
        self.axes["reconstruction"].grid(True)
        
        # Prediction error and surprise
        self.axes["prediction"] = self.figures["performance"].add_subplot(gs[0, 1])
        self.axes["prediction"].set_title("Prediction Error & Surprise")
        self.axes["prediction"].set_xlabel("Frames")
        self.axes["prediction"].set_ylabel("Value")
        self.axes["prediction"].grid(True)
        
        # Stability
        self.axes["stability"] = self.figures["performance"].add_subplot(gs[1, 0])
        self.axes["stability"].set_title("Stability Measures")
        self.axes["stability"].set_xlabel("Frames")
        self.axes["stability"].set_ylabel("Value")
        self.axes["stability"].grid(True)
        
        # Structural changes
        self.axes["structure"] = self.figures["performance"].add_subplot(gs[1, 1])
        self.axes["structure"].set_title("Structural Changes")
        self.axes["structure"].set_xlabel("Frames")
        self.axes["structure"].set_ylabel("Count")
        self.axes["structure"].grid(True)
        
        # Processing time
        self.axes["processing"] = self.figures["performance"].add_subplot(gs[2, 0])
        self.axes["processing"].set_title("Processing Time")
        self.axes["processing"].set_xlabel("Frames")
        self.axes["processing"].set_ylabel("Time (s)")
        self.axes["processing"].grid(True)
        
        # Active neurons
        self.axes["activity"] = self.figures["performance"].add_subplot(gs[2, 1])
        self.axes["activity"].set_title("Neuron Activity")
        self.axes["activity"].set_xlabel("Frames")
        self.axes["activity"].set_ylabel("Fraction Active")
        self.axes["activity"].grid(True)
        
        # Adjust layout
        self.figures["performance"].tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Show figure
        plt.ion()  # Interactive mode
        self.figures["performance"].show()
    
    def _update_visualization(self):
        """Update visualization with current data."""
        if not self.history["timestamps"]:
            # No data yet
            return
        
        # Get data for x-axis
        if len(self.history["timestamps"]) > 1:
            x = list(range(len(self.history["timestamps"])))
            start_idx = max(0, len(x) - self.max_history_length)
            x = x[start_idx:]
        else:
            x = [0]
        
        # Update reconstruction errors
        visual_errors = self.history["visual_errors"][-len(x):]
        audio_errors = self.history["audio_errors"][-len(x):]
        cross_modal_errors = self.history["cross_modal_errors"][-len(x):]
        
        self.axes["reconstruction"].clear()
        self.axes["reconstruction"].plot(x, visual_errors, 'r-', label='Visual')
        self.axes["reconstruction"].plot(x, audio_errors, 'b-', label='Audio')
        self.axes["reconstruction"].plot(x, cross_modal_errors, 'g-', label='Cross-modal')
        self.axes["reconstruction"].set_title("Reconstruction Errors")
        self.axes["reconstruction"].set_xlabel("Frames")
        self.axes["reconstruction"].set_ylabel("Error")
        self.axes["reconstruction"].legend()
        self.axes["reconstruction"].grid(True)
        
        # Update prediction error and surprise
        prediction_errors = self.history["prediction_errors"][-len(x):]
        surprise_signals = self.history["surprise_signals"][-len(x):]
        
        self.axes["prediction"].clear()
        self.axes["prediction"].plot(x, prediction_errors, 'r-', label='Prediction Error')
        self.axes["prediction"].plot(x, surprise_signals, 'b-', label='Surprise')
        self.axes["prediction"].set_title("Prediction Error & Surprise")
        self.axes["prediction"].set_xlabel("Frames")
        self.axes["prediction"].set_ylabel("Value")
        self.axes["prediction"].legend()
        self.axes["prediction"].grid(True)
        
        # Update stability
        stability_measures = self.history["stability_measures"][-len(x):]
        
        self.axes["stability"].clear()
        self.axes["stability"].plot(x, stability_measures, 'g-', label='Activity Error')
        self.axes["stability"].set_title("Stability Measures")
        self.axes["stability"].set_xlabel("Frames")
        self.axes["stability"].set_ylabel("Value")
        self.axes["stability"].legend()
        self.axes["stability"].grid(True)
        
        # Update structural changes
        if self.history["structural_changes"]:
            neurons_added = [sc.get("neurons_added", 0) for sc in self.history["structural_changes"][-len(x):]]
            connections_pruned = [sc.get("connections_pruned", 0) for sc in self.history["structural_changes"][-len(x):]]
            connections_sprouted = [sc.get("connections_sprouted", 0) for sc in self.history["structural_changes"][-len(x):]]
            
            self.axes["structure"].clear()
            self.axes["structure"].plot(x, neurons_added, 'r-', label='Neurons Added')
            self.axes["structure"].plot(x, connections_pruned, 'b-', label='Connections Pruned')
            self.axes["structure"].plot(x, connections_sprouted, 'g-', label='Connections Sprouted')
            self.axes["structure"].set_title("Structural Changes")
            self.axes["structure"].set_xlabel("Frames")
            self.axes["structure"].set_ylabel("Count")
            self.axes["structure"].legend()
            self.axes["structure"].grid(True)
        
        # Update processing time
        processing_times = self.history["processing_times"][-len(x):]
        
        self.axes["processing"].clear()
        self.axes["processing"].plot(x, processing_times, 'b-')
        self.axes["processing"].set_title("Processing Time")
        self.axes["processing"].set_xlabel("Frames")
        self.axes["processing"].set_ylabel("Time (s)")
        self.axes["processing"].grid(True)
        
        # Update active neurons
        if self.history["structural_changes"]:
            active_fractions = [sc.get("active_neuron_fraction", 0.0) for sc in self.history["structural_changes"][-len(x):]]
            
            self.axes["activity"].clear()
            self.axes["activity"].plot(x, active_fractions, 'g-')
            self.axes["activity"].set_title("Neuron Activity")
            self.axes["activity"].set_xlabel("Frames")
            self.axes["activity"].set_ylabel("Fraction Active")
            self.axes["activity"].grid(True)
        
        # Update figure
        try:
            self.figures["performance"].canvas.draw_idle()
            self.figures["performance"].canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
    
    def _cleanup_visualization(self):
        """Clean up visualization resources."""
        try:
            plt.close('all')
        except:
            pass
    
    def take_snapshot(self):
        """
        Take a snapshot of the current system state and save visualizations.
        """
        # Skip if saving is disabled
        if not self.config.get("save_snapshots", True):
            return
        
        # Create snapshot timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Save performance plots
        if self.enable_visualization and self.figures.get("performance") is not None:
            try:
                self.figures["performance"].savefig(
                    os.path.join(snapshot_dir, "performance.png"),
                    dpi=120
                )
            except Exception as e:
                logger.error(f"Error saving performance plot: {e}")
        
        # Save system state snapshot
        try:
            # Get system stats
            stats = self.system.get_stats()
            
            # Save stats as text
            with open(os.path.join(snapshot_dir, "stats.txt"), "w") as f:
                f.write(f"System Stats - {timestamp}\n")
                f.write("=" * 40 + "\n\n")
                
                # System-level stats
                f.write(f"Frames processed: {stats.get('frames_processed', 0)}\n")
                f.write(f"Multimodal size: {stats.get('multimodal_size', 0)}\n")
                f.write(f"Visual output size: {stats.get('visual_output_size', 0)}\n")
                f.write(f"Audio output size: {stats.get('audio_output_size', 0)}\n")
                f.write(f"Learning enabled: {stats.get('learning_enabled', False)}\n\n")
                
                # Component stats
                components = stats.get("components", {})
                for component_name, component_stats in components.items():
                    f.write(f"{component_name.upper()} Stats:\n")
                    f.write("-" * 40 + "\n")
                    
                    for key, value in component_stats.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"    {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    
                    f.write("\n")
            
            # Save history data as numpy arrays
            np.savez(
                os.path.join(snapshot_dir, "history.npz"),
                timestamps=np.array(self.history["timestamps"]),
                visual_errors=np.array(self.history["visual_errors"]),
                audio_errors=np.array(self.history["audio_errors"]),
                cross_modal_errors=np.array(self.history["cross_modal_errors"]),
                prediction_errors=np.array(self.history["prediction_errors"]),
                surprise_signals=np.array(self.history["surprise_signals"]),
                stability_measures=np.array(self.history["stability_measures"]),
                processing_times=np.array(self.history["processing_times"])
            )
            
            logger.info(f"Saved system snapshot to {snapshot_dir}")
            
        except Exception as e:
            logger.error(f"Error saving system snapshot: {e}", exc_info=True)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the monitoring session.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history["timestamps"]:
            return {
                "frames_processed": 0,
                "monitoring_duration": 0,
                "avg_processing_time": 0,
                "avg_visual_error": 0,
                "avg_audio_error": 0,
                "avg_prediction_error": 0,
                "avg_surprise": 0
            }
        
        # Calculate basic stats
        frames_processed = len(self.history["timestamps"])
        
        if frames_processed > 0:
            monitoring_duration = self.history["timestamps"][-1] - self.history["timestamps"][0]
            avg_processing_time = np.mean(self.history["processing_times"])
            avg_visual_error = np.mean(self.history["visual_errors"])
            avg_audio_error = np.mean(self.history["audio_errors"])
            avg_prediction_error = np.mean(self.history["prediction_errors"])
            avg_surprise = np.mean(self.history["surprise_signals"])
        else:
            monitoring_duration = 0
            avg_processing_time = 0
            avg_visual_error = 0
            avg_audio_error = 0
            avg_prediction_error = 0
            avg_surprise = 0
        
        # Calculate learning progress
        if frames_processed > 100:
            # Compare first and last 100 frames
            early_visual_error = np.mean(self.history["visual_errors"][:100])
            recent_visual_error = np.mean(self.history["visual_errors"][-100:])
            visual_improvement = max(0, early_visual_error - recent_visual_error) / max(0.0001, early_visual_error)
            
            early_audio_error = np.mean(self.history["audio_errors"][:100])
            recent_audio_error = np.mean(self.history["audio_errors"][-100:])
            audio_improvement = max(0, early_audio_error - recent_audio_error) / max(0.0001, early_audio_error)
            
            early_prediction_error = np.mean(self.history["prediction_errors"][:100])
            recent_prediction_error = np.mean(self.history["prediction_errors"][-100:])
            prediction_improvement = max(0, early_prediction_error - recent_prediction_error) / max(0.0001, early_prediction_error)
            
            learning_progress = {
                "visual_improvement": float(visual_improvement),
                "audio_improvement": float(audio_improvement),
                "prediction_improvement": float(prediction_improvement)
            }
        else:
            learning_progress = {
                "visual_improvement": 0.0,
                "audio_improvement": 0.0,
                "prediction_improvement": 0.0
            }
        
        return {
            "frames_processed": frames_processed,
            "monitoring_duration": float(monitoring_duration),
            "avg_processing_time": float(avg_processing_time),
            "avg_visual_error": float(avg_visual_error),
            "avg_audio_error": float(avg_audio_error),
            "avg_prediction_error": float(avg_prediction_error),
            "avg_surprise": float(avg_surprise),
            "learning_progress": learning_progress
        } 