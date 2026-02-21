import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import logging
import numpy as np
import scipy.signal
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from self_organizing_av_system.core.system import SelfOrganizingAVSystem

logger = logging.getLogger(__name__)

class NetworkMonitor:
    """Visual monitor for the self-organizing AV system."""

    def __init__(self, system: 'SelfOrganizingAVSystem', fps_target: int = 30, skip_frames: int = 3):
        self.system = system
        self.skip_frames = skip_frames  # Skip frames to reduce drawing overhead
        self.frame_counter = 0
        self.fps_target = fps_target
        self.last_update_time = time.time()

    def update(self, visual_input: np.ndarray, audio_input: np.ndarray, result: Dict[str, Any]) -> None:
        """
        Update the visualization with new data.
        
        Args:
            visual_input: Raw visual input
            audio_input: Raw audio input
            result: Processing result from the system
        """
        # Skip frames to improve responsiveness
        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            return
        
        # Track if any updates occurred (to avoid unnecessary drawing)
        any_updates = False
        
        # Update audio visualization
        if self.audio_ax is not None and audio_input is not None:
            try:
                # Only update every few frames to reduce overhead
                if self.frame_counter % (self.skip_frames * 2) == 0:
                    # Clear previous plot
                    self.audio_ax.clear()
                    
                    # Create spectrogram of audio
                    if len(audio_input) > 0:
                        # Add small amount of noise to avoid all-zero inputs
                        audio_with_noise = audio_input + np.random.normal(0, 1e-10, audio_input.shape)
                        
                        # Calculate proper nperseg and noverlap values
                        nperseg = min(256, len(audio_with_noise))  # Ensure nperseg is at most the length
                        noverlap = max(0, min(nperseg-1, int(nperseg * 0.75)))  # Ensure noverlap < nperseg
                        
                        # Generate spectrogram
                        _, _, spec = scipy.signal.spectrogram(
                            audio_with_noise, 
                            fs=48000, 
                            nperseg=nperseg, 
                            noverlap=noverlap
                        )
                        
                        # Display spectrogram
                        self.audio_ax.imshow(
                            10 * np.log10(spec + 1e-10),  # Convert to dB scale with small offset
                            aspect='auto',
                            origin='lower',
                            cmap='viridis'
                        )
                        
                        self.audio_ax.set_title("Audio Input")
                        self.audio_ax.set_xlabel("Time")
                        self.audio_ax.set_ylabel("Frequency")
                        any_updates = True
                
            except Exception as e:
                logger.error(f"Error updating audio visualization: {e}")

        # Update neuron visualizations
        if self.frame_counter % (self.skip_frames * 2) == 0 and "multimodal_state" in result:
            # Update combined neural activations (more efficient than separate plots)
            if self.association_ax is not None:
                try:
                    # Clear previous plot
                    self.association_ax.clear()
                    
                    # Get multimodal activity
                    mm_activity = result.get("multimodal_state", np.zeros(1))
                    
                    # Reshape to 2D if 1D
                    if len(mm_activity.shape) == 1:
                        dim = int(np.sqrt(len(mm_activity)))
                        mm_activity = mm_activity[:dim*dim].reshape(dim, dim)
                    
                    # Display activity
                    self.association_ax.imshow(
                        mm_activity,
                        cmap='viridis',
                        vmin=0,
                        vmax=1
                    )
                    
                    self.association_ax.set_title("Neural Activations")
                    self.association_ax.axis('off')  # Turn off axis for performance
                    any_updates = True
                    
                except Exception as e:
                    logger.error(f"Error updating neural visualization: {e}")
            
        # Update figure
        if any_updates:
            # Use blit-like approach for faster drawing
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception as e:
                logger.error(f"Error updating figure: {e}")
                
        # Limit update rate to target FPS
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        target_interval = 1.0 / self.fps_target
        
        if elapsed < target_interval:
            # Sleep to maintain target FPS
            time.sleep(max(0, target_interval - elapsed))
        
        self.last_update_time = time.time()

    def setup_visualization(self) -> None:
        """Setup the visualization layout."""
        try:
            # Create figure with subplots
            self.fig = plt.figure(figsize=(10, 8))
            grid = gridspec.GridSpec(3, 2, figure=self.fig)
            
            # Input visualizations
            self.visual_ax = self.fig.add_subplot(grid[0, 0])
            self.audio_ax = self.fig.add_subplot(grid[0, 1])
            
            # Association visualizations
            self.association_ax = self.fig.add_subplot(grid[1, :])
            # Prediction visualization - simplified to one plot
            self.prediction_ax = self.fig.add_subplot(grid[2, :])
            
            # Set title
            self.fig.suptitle("Self-Organizing AV System Monitor")
            
            # Adjust layout
            self.fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Set interactive mode
            plt.ion()
            
            # Show figure
            self.fig.show()
            
            self.visualization_ready = True
        
        except Exception as e:
            logger.error(f"Error setting up visualization: {e}")
            self.visualization_ready = False 