import tkinter as tk
from tkinter import ttk
import threading
import logging
import time
import numpy as np
from PIL import Image, ImageTk
from typing import Optional

class TkMonitor:
    """
    A Tkinter-based GUI monitor for the Self-Organizing AV System.

    Displays:
    - A video panel controlled by the model's output.
    - Information about the model's current architecture.
    - A dynamic visualization of the neural network structure.
    """
    def __init__(self, system, update_interval_ms=100, video_panel_size=(320, 240)):
        """
        Initialize the Tkinter Monitor.

        Args:
            system: The SelfOrganizingAVSystem instance to monitor.
            update_interval_ms: How often to refresh the GUI in milliseconds.
            video_panel_size: The (width, height) of the video panel.
        """
        self.logger = logging.getLogger(__name__)
        self.system = system
        self.update_interval_ms = update_interval_ms
        self.video_panel_width, self.video_panel_height = video_panel_size
        self.is_running = False
        self.root = None
        self.video_canvas = None
        self.arch_text = None
        self.video_photo_image = None # Store PhotoImage to prevent garbage collection
        self.model_direct_control = False  # Flag for model's direct pixel control
        self.network_canvas = None
        self.network_photo_image = None
        self.prev_connections = 0
        self.prev_neurons = 0
        
        # Initialize UI control elements to None to prevent attribute errors
        self.grayscale_var = None
        self.contrast_scale = None
        self.filter_scale = None
        self.direct_control_var = None

        # Add model control parameters
        self.current_video_params = {
            'size': video_panel_size,
            'grayscale': False,
            'contrast': 1.0,
            'filter_strength': 0.5
        }

        # Use a reasonable default image size to avoid memory issues
        safe_size = (min(320, self.video_panel_width), min(240, self.video_panel_height))
        try:
            # Try to create a placeholder image of the requested size
            self._last_valid_image = Image.new('RGB', video_panel_size, (128, 128, 128))
        except (MemoryError, OSError):
            # If that fails, use a smaller size
            self.logger.warning(f"Could not create image of size {video_panel_size}, falling back to smaller size {safe_size}")
            self._last_valid_image = Image.new('RGB', safe_size, (128, 128, 128))
            # Update the panel size to match
            self.video_panel_width, self.video_panel_height = safe_size

    def _create_widgets(self):
        """Create the Tkinter widgets."""
        try:
            self.root = tk.Tk()
            self.root.title("Self-Organizing AV System Monitor")

            # Main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)

            # Video Panel (Left)
            video_frame = ttk.LabelFrame(main_frame, text="Model Video Output", padding="5")
            video_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(0, weight=1)

            self.video_canvas = tk.Canvas(video_frame,
                                        width=self.video_panel_width,
                                        height=self.video_panel_height,
                                        bg="gray")
            self.video_canvas.pack(expand=True, fill=tk.BOTH)

            # Architecture Info (Right)
            arch_frame = ttk.LabelFrame(main_frame, text="Model Architecture", padding="5")
            arch_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E))
            main_frame.grid_columnconfigure(1, weight=0) # Don't expand horizontally

            self.arch_text = tk.Text(arch_frame, width=40, height=20, wrap=tk.WORD, state=tk.DISABLED)
            self.arch_text.pack(expand=True, fill=tk.BOTH)
            
            # Neural Network Visualization (Middle section of architecture frame)
            self.network_canvas = tk.Canvas(arch_frame, width=320, height=200, bg="black")
            self.network_canvas.pack(expand=True, fill=tk.BOTH, pady=10)

            # Add Control Panel (Bottom)
            control_frame = ttk.LabelFrame(main_frame, text="Model Controls", padding="5")
            control_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
            
            # Add control widgets
            self.grayscale_var = tk.BooleanVar()
            ttk.Checkbutton(control_frame, text="Grayscale", variable=self.grayscale_var,
                        command=self._update_model_params).pack(side=tk.LEFT, padx=5)
            
            ttk.Label(control_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
            self.contrast_scale = ttk.Scale(control_frame, from_=0.5, to=2.0, 
                                        command=lambda v: self._update_model_params())
            self.contrast_scale.set(1.0)
            self.contrast_scale.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(control_frame, text="Filter Strength:").pack(side=tk.LEFT, padx=5)
            self.filter_scale = ttk.Scale(control_frame, from_=0.0, to=1.0,
                                        command=lambda v: self._update_model_params())
            self.filter_scale.set(0.5)
            self.filter_scale.pack(side=tk.LEFT, padx=5)
            
            # Add direct pixel control toggle
            self.direct_control_var = tk.BooleanVar()
            ttk.Checkbutton(control_frame, text="Direct Pixel Control", variable=self.direct_control_var,
                        command=self._toggle_direct_control).pack(side=tk.LEFT, padx=5)

            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        except Exception as e:
            self.logger.error(f"Error creating widgets: {e}")
            # Ensure minimal functionality even if UI creation fails
            if not self.root:
                try:
                    self.root = tk.Tk()
                    self.root.title("AV System Monitor (Error Mode)")
                    tk.Label(self.root, text=f"Error initializing full UI: {e}").pack()
                    self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
                except Exception as e2:
                    self.logger.error(f"Critical error creating fallback UI: {e2}")

    def _toggle_direct_control(self):
        """Toggle the model's direct pixel control mode"""
        try:
            if self.direct_control_var:
                self.model_direct_control = self.direct_control_var.get()
                if hasattr(self.system, 'set_direct_pixel_control'):
                    self.system.set_direct_pixel_control(self.model_direct_control)
        except Exception as e:
            self.logger.error(f"Error toggling direct control: {e}")
        
    def _update_model_params(self):
        """Update model parameters based on GUI controls"""
        try:
            # Safely get values from UI elements with fallbacks
            grayscale = self.grayscale_var.get() if hasattr(self, 'grayscale_var') and self.grayscale_var else False
            contrast = self.contrast_scale.get() if hasattr(self, 'contrast_scale') and self.contrast_scale else 1.0
            filter_strength = self.filter_scale.get() if hasattr(self, 'filter_scale') and self.filter_scale else 0.5
            
            new_params = {
                'grayscale': grayscale,
                'contrast': contrast,
                'filter_strength': filter_strength, 
                'output_size': (self.video_panel_width, self.video_panel_height)
            }
            
            # Update current params cache
            self.current_video_params.update(new_params)
            
            # Send to system if method exists
            if hasattr(self.system, 'update_video_params'):
                self.system.update_video_params(new_params)
        except Exception as e:
            self.logger.error(f"Error updating model params: {e}")
            
    def _create_network_visualization(self, arch_info):
        """
        Create a visual representation of the neural network architecture
        showing layers, neurons, and connections.
        """
        try:
            # Get network information
            visual_layers = arch_info.get("Visual Layers", [])
            audio_layers = arch_info.get("Audio Layers", [])
            multimodal_size = arch_info.get("Multimodal Size", 10)
            
            # Get current actual network structure info
            current_size = arch_info.get("Current Multimodal Size", multimodal_size)
            neurons_added = arch_info.get("Structural Changes", {}).get("Neurons Added", 0)
            neurons_pruned = arch_info.get("Structural Changes", {}).get("Neurons Pruned", 0)
            connections = arch_info.get("Approx. Connections", 0)
            
            # Safely get plasticity events
            plasticity_events = []
            try:
                plasticity_events = arch_info.get("Structural Changes", {}).get("Recent Plasticity Events", [])
            except (AttributeError, TypeError):
                # If we can't get the events for any reason, just use an empty list
                pass
            
            # Create a blank image for drawing
            width, height = 320, 200
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Colors for different components
            visual_color = (180, 50, 180)  # Purple
            audio_color = (50, 200, 200)   # Cyan
            multimodal_color = (200, 200, 50) # Yellow
            new_neuron_color = (50, 255, 50)  # Bright green for new neurons
            pruned_neuron_color = (255, 0, 0)  # Red for recently pruned neurons
            
            # Calculate positions
            margin_x = 20
            margin_y = 20
            available_width = width - 2 * margin_x
            available_height = height - 2 * margin_y
            
            # Layout parameters
            visual_x = margin_x
            audio_x = width - margin_x
            multimodal_x = width // 2
            
            # Track if we detected structural changes
            detected_changes = False
            
            # Check if network structure has changed
            if current_size != self.prev_neurons or connections != self.prev_connections:
                detected_changes = True
                # Add text showing growth/pruning
                growth_text = f"+{neurons_added} / -{neurons_pruned}"
                font_size = 0.4
                font_color = (255, 255, 255)
                
                # Draw text at bottom of the visualization
                cv_text(img, growth_text, (width//2 - 20, height - 15), font_color)
                
                # Track new state
                self.prev_neurons = current_size
            
            # Draw visual pathway (left side)
            visual_positions = []
            for i, layer_size in enumerate(visual_layers):
                layer_height = min(available_height, layer_size * 8)
                spacing = max(3, layer_height / max(1, layer_size))
                
                for j in range(min(layer_size, 20)):  # Limit to 20 neurons per layer for visualization
                    y = margin_y + j * spacing
                    if y < height - margin_y:
                        # Draw neuron as a circle
                        x = visual_x + i * 30
                        cv_circle(img, (x, int(y)), 3, visual_color, -1)
                        visual_positions.append((x, int(y)))
            
            # Draw audio pathway (right side)
            audio_positions = []
            for i, layer_size in enumerate(audio_layers):
                layer_height = min(available_height, layer_size * 8)
                spacing = max(3, layer_height / max(1, layer_size))
                
                for j in range(min(layer_size, 20)):  # Limit visualization
                    y = margin_y + j * spacing
                    if y < height - margin_y:
                        # Draw neuron as a circle
                        x = audio_x - i * 30
                        cv_circle(img, (x, int(y)), 3, audio_color, -1)
                        audio_positions.append((x, int(y)))
            
            # Draw multimodal neurons (center) - use actual current size
            multimodal_positions = []
            multimodal_height = min(available_height, current_size * 8)
            multimodal_spacing = max(3, multimodal_height / max(1, current_size))
            
            for i in range(min(current_size, 20)):  # Limit visualization but use CURRENT size
                y = margin_y + i * multimodal_spacing
                if y < height - margin_y:
                    x = multimodal_x
                    
                    # Use different colors for new or pruned neurons
                    if i >= current_size - neurons_added:  # Recent new neurons
                        cv_circle(img, (x, int(y)), 4, new_neuron_color, -1)
                    else:
                        cv_circle(img, (x, int(y)), 4, multimodal_color, -1)
                        
                    multimodal_positions.append((x, int(y)))
            
            # Draw connections between layers
            # Get connection info if available
            curr_connections = connections
            
            # Calculate connection intensity based on current vs previous connections
            if self.prev_connections > 0:
                connection_ratio = float(curr_connections) / float(self.prev_connections)
                # Highlight intensity changes
                if connection_ratio > 1.1:  # At least 10% more connections
                    connection_intensity = 1.0  # Full brightness for growth
                elif connection_ratio < 0.9:  # At least 10% fewer connections
                    connection_intensity = 0.3  # Dimmer for pruning
                else:
                    connection_intensity = 0.7  # Normal brightness
            else:
                connection_intensity = 0.7  # Default
                
            # Visual to multimodal connections
            if visual_positions and multimodal_positions:
                # Show connection density by number of lines drawn
                connection_count = min(150, len(visual_positions) * len(multimodal_positions))
                # Scale by actual connection count relative to last count
                if self.prev_connections > 0:
                    connection_scale = min(1.5, max(0.5, curr_connections / self.prev_connections))
                    connection_count = int(connection_count * connection_scale)
                
                # Draw some connections (more if more connections exist)
                for _ in range(connection_count):
                    from_idx = np.random.randint(0, len(visual_positions))
                    to_idx = np.random.randint(0, len(multimodal_positions))
                    
                    from_pos = visual_positions[from_idx]
                    to_pos = multimodal_positions[to_idx]
                    
                    # Draw line with alpha blending
                    cv_line(img, from_pos, to_pos, visual_color, 1, connection_intensity)
            
            # Audio to multimodal connections
            if audio_positions and multimodal_positions:
                connection_count = min(150, len(audio_positions) * len(multimodal_positions))
                # Scale by connection count
                if self.prev_connections > 0:
                    connection_scale = min(1.5, max(0.5, curr_connections / self.prev_connections))
                    connection_count = int(connection_count * connection_scale)
                
                # Draw some connections
                for _ in range(connection_count):
                    from_idx = np.random.randint(0, len(audio_positions))
                    to_idx = np.random.randint(0, len(multimodal_positions))
                    
                    from_pos = audio_positions[from_idx]
                    to_pos = multimodal_positions[to_idx]
                    
                    # Draw line with alpha blending
                    cv_line(img, from_pos, to_pos, audio_color, 1, connection_intensity)
            
            # Draw recurrent connections within multimodal layer
            if len(multimodal_positions) > 1:
                # Add some recurrent connections - more if recent plasticity events
                recurrent_count = min(50, len(multimodal_positions) * len(multimodal_positions) // 4)
                
                for _ in range(recurrent_count):
                    from_idx = np.random.randint(0, len(multimodal_positions))
                    to_idx = np.random.randint(0, len(multimodal_positions))
                    
                    if from_idx != to_idx:  # Don't connect to self
                        from_pos = multimodal_positions[from_idx]
                        to_pos = multimodal_positions[to_idx]
                        
                        # Draw curved recurrent connections
                        cv_curved_line(img, from_pos, to_pos, (150, 150, 200), 1, 0.4)
            
            # Update previous connections count for change detection
            if connections != self.prev_connections:
                # Only print to log if significant change
                if abs(connections - self.prev_connections) > 5:
                    self.logger.info(f"Network connections changed: {self.prev_connections} -> {connections}")
                self.prev_connections = connections
            
            # Add growth/prune markers if there were recent structural events
            if plasticity_events and len(plasticity_events) > 0:
                # Show a small indicator that structural changes occurred
                cv_text(img, "Growth/Prune Events!", (width//2 - 60, 15), (255, 255, 0))
            
            # Draw frame counter at the bottom
            frames_processed = arch_info.get("Frames Processed", 0)
            cv_text(img, f"Frames: {frames_processed}", (5, height - 5), (150, 150, 150))
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error creating network visualization: {e}")
            # Return a fallback blank image with error message
            img = np.zeros((200, 320, 3), dtype=np.uint8)
            cv_text(img, "Visualization Error", (100, 100), (255, 50, 50))
            return img
    
    def _update_gui(self):
        """Fetch data from the system and update the GUI elements."""
        if not self.is_running or not self.root:
            return

        try:
            # Send current video params to model before getting output
            self._update_model_params()
            
            # Get video output
            video_output = self.system.get_video_panel_output(self.current_video_params)
                
            # 1. Update Architecture Info
            arch_info = self.system.get_architecture_info()
            if self.arch_text:
                self.arch_text.config(state=tk.NORMAL)
                self.arch_text.delete(1.0, tk.END)
                # Format the dictionary nicely
                info_str = ""
                if isinstance(arch_info, dict):
                    for key, value in arch_info.items():
                        info_str += f"{key}: {value}\n"
                else:
                    info_str = str(arch_info) # Fallback if not a dict
                self.arch_text.insert(tk.END, info_str)
                self.arch_text.config(state=tk.DISABLED)

            # 2. Update Neural Network Visualization
            if self.network_canvas:
                network_img = self._create_network_visualization(arch_info)
                network_pil = Image.fromarray(network_img)
                self.network_photo_image = ImageTk.PhotoImage(image=network_pil)
                self.network_canvas.create_image(0, 0, anchor=tk.NW, image=self.network_photo_image)

            # Check if we need to update the direct control checkbox
            if hasattr(self.system, 'direct_pixel_control') and hasattr(self, 'direct_control_var') and self.direct_control_var:
                is_enabled = self.system.direct_pixel_control
                if is_enabled != self.direct_control_var.get():
                    self.direct_control_var.set(is_enabled)
                    self.model_direct_control = is_enabled

            # 3. Update Video Panel
            if self.video_canvas and isinstance(video_output, np.ndarray):
                # Validate the shape of the array
                expected_shape = (self.video_panel_height, self.video_panel_width, 3)
                
                if video_output.shape == expected_shape and video_output.dtype == np.uint8:
                    # Use the model's direct pixel output
                    img = Image.fromarray(video_output)
                    self._last_valid_image = img
                elif video_output.shape != expected_shape:
                    # Try to reshape or resize if possible
                    self.logger.warning(f"Reshaping video output from {video_output.shape} to {expected_shape}")
                    try:
                        # If it's a flat array, reshape it
                        if video_output.size == self.video_panel_width * self.video_panel_height * 3:
                            reshaped = video_output.reshape(expected_shape)
                            img = Image.fromarray(reshaped.astype(np.uint8))
                            self._last_valid_image = img
                        else:
                            # Otherwise create a new image and resize
                            temp_img = Image.fromarray(video_output.astype(np.uint8))
                            img = temp_img.resize((self.video_panel_width, self.video_panel_height))
                            self._last_valid_image = img
                    except Exception as e:
                        self.logger.error(f"Failed to reshape video output: {e}")
                        img = self._last_valid_image
                else:
                    # Convert to uint8 if needed
                    if video_output.dtype != np.uint8:
                        # Scale float values from [0,1] to [0,255]
                        if video_output.dtype == np.float32 or video_output.dtype == np.float64:
                            if np.max(video_output) <= 1.0:
                                video_output = (video_output * 255).astype(np.uint8)
                            else:
                                video_output = np.clip(video_output, 0, 255).astype(np.uint8)
                    
                    img = Image.fromarray(video_output)
                    self._last_valid_image = img
            elif video_output is not None:
                self.logger.warning(f"Received non-ndarray video output: type={type(video_output)}. Expected NumPy array.")
                img = self._last_valid_image # Use last valid image on error
            else:
                # If system returns None, use the last valid image
                img = self._last_valid_image

            # Convert PIL Image to Tkinter PhotoImage and display
            try:
                if self.video_canvas:
                    self.video_photo_image = ImageTk.PhotoImage(image=img)
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_photo_image)
            except (tk.TclError, OSError, MemoryError) as e:
                # If there's an error creating the PhotoImage (could be memory issues)
                self.logger.error(f"Error creating PhotoImage: {e}")
                # Try with a smaller image as fallback
                try:
                    small_img = img.resize((160, 120), Image.Resampling.NEAREST)
                    self.video_photo_image = ImageTk.PhotoImage(image=small_img)
                    if self.video_canvas:
                        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_photo_image)
                    self.logger.info("Successfully displayed smaller fallback image")
                except Exception as e2:
                    self.logger.error(f"Even fallback image failed: {e2}")
                    # Just give up on updating the image this cycle

        except Exception as e:
            self.logger.error(f"Error updating Tkinter GUI: {e}", exc_info=True)

        # Schedule the next update
        if self.is_running and self.root:
            try:
                self.root.after(self.update_interval_ms, self._update_gui)
            except tk.TclError:
                # Root window might have been destroyed
                self.is_running = False

    def _run_gui(self):
        """Create and run the Tkinter main loop."""
        try:
            self._create_widgets()
            self.is_running = True
            if self.root:
                self.root.after(self.update_interval_ms, self._update_gui) # Start update loop
                self.logger.info("Starting Tkinter main loop...")
                self.root.mainloop()
                self.logger.info("Tkinter main loop finished.")
        except Exception as e:
            self.logger.error(f"Error initializing Tkinter GUI: {e}", exc_info=True)
        finally:
            self.is_running = False # Ensure updates stop even if mainloop crashes

    def start(self):
        """Start the GUI in a separate thread."""
        if self.is_running:
            self.logger.warning("Monitor GUI is already running.")
            return

        # Run the Tkinter main loop in a separate thread
        # This is necessary because mainloop() is blocking
        self.gui_thread = threading.Thread(target=self._run_gui, daemon=True)
        self.gui_thread.start()

        # Give the GUI thread a moment to initialize and check if it started successfully
        time.sleep(0.5)
        if self.gui_thread.is_alive() and not self.is_running:
            # This condition might happen if _create_widgets fails quickly
            self.logger.warning("GUI thread started but Tkinter main loop initialization might have failed.")
        elif not self.gui_thread.is_alive():
            self.logger.error("GUI thread failed to start.")

    def stop(self):
        """Stop the GUI."""
        self.logger.info("Stopping Tkinter monitor...")
        self.is_running = False
        if self.root:
            # Safely destroy the window from the main thread or GUI thread
            # Using `after` ensures it runs in the GUI thread's context
            try:
                self.root.after(0, self.root.destroy)
            except tk.TclError:
                # Window might already be destroyed
                pass
        
        # Clear any stored references that might cause issues
        self.video_photo_image = None
        self.network_photo_image = None
        
        if hasattr(self, 'gui_thread') and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=1.0) # Wait briefly for the thread

    def _on_closing(self):
        """Handle the window close button."""
        self.logger.info("GUI window closed by user.")
        self.stop()

# Utility drawing functions for the network visualization
def cv_circle(img, center, radius, color, thickness):
    """Draw a circle on the image"""
    x, y = center
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        r = min(radius, 10)  # Limit radius
        for i in range(max(0, y-r), min(h, y+r+1)):
            for j in range(max(0, x-r), min(w, x+r+1)):
                dist = np.sqrt((i-y)**2 + (j-x)**2)
                if dist <= r:
                    if thickness < 0:  # Fill circle
                        img[i, j] = color
                    elif dist >= r - thickness:  # Draw outline
                        img[i, j] = color

def cv_line(img, pt1, pt2, color, thickness, alpha=1.0):
    """Draw a line with alpha transparency on the image"""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = img.shape[:2]
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            # Alpha blending
            if alpha < 1.0:
                img[y1, x1] = (img[y1, x1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
            else:
                img[y1, x1] = color
                
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy 

def cv_text(img, text, position, color, font_scale=0.4):
    """Draw text on the image"""
    x, y = position
    h, w = img.shape[:2]
    
    # Simple text drawing
    for i, char in enumerate(text):
        char_x = x + i * 6  # Simple fixed-width font
        if 0 <= char_x < w and 0 <= y < h:
            # Draw a simple bitmap representation of the character
            for dy in range(-3, 4):
                for dx in range(-2, 3):
                    px, py = char_x + dx, y + dy
                    if 0 <= px < w and 0 <= py < h:
                        # Simple bitmap for characters
                        if (dx == 0 and abs(dy) < 3) or (dy == 0 and abs(dx) < 2) or (abs(dx) == 1 and abs(dy) == 1):
                            img[py, px] = color

def cv_curved_line(img, pt1, pt2, color, thickness, alpha=0.4):
    """Draw a curved line between two points"""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = img.shape[:2]
    
    # Calculate control point for curve
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    
    # Offset control point perpendicular to line
    dx = x2 - x1
    dy = y2 - y1
    
    # Perpendicular direction
    length = max(1, np.sqrt(dx**2 + dy**2))
    px = -dy / length * 20  # Control point offset
    py = dx / length * 20
    
    # Control point
    cx = mx + px
    cy = my + py
    
    # Draw quadratic Bezier curve (approximation)
    steps = 20
    prev_x, prev_y = x1, y1
    
    for i in range(1, steps + 1):
        t = i / steps
        # Quadratic Bezier formula
        x = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        y = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
        
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h and 0 <= prev_x < w and 0 <= prev_y < h:
            # Draw line segment
            cv_line(img, (prev_x, prev_y), (x, y), color, thickness, alpha)
        
        prev_x, prev_y = x, y 