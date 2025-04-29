import cv2
import numpy as np
import pyaudio
import threading
import time
from typing import Tuple, Optional, Callable, List, Dict, Any
import queue
import logging


class AVCapture:
    """
    Synchronized audio-video capture from webcam and microphone.
    
    This utility handles capturing synchronized audio and video streams
    for processing by the self-organizing system.
    """
    
    def __init__(
        self,
        video_width: int = 640,
        video_height: int = 480,
        fps: int = 30,
        sample_rate: int = 22050,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None
    ):
        """
        Initialize the audio-video capture.
        
        Args:
            video_width: Width of captured video frames
            video_height: Height of captured video frames
            fps: Target frames per second
            sample_rate: Audio sample rate
            channels: Audio channels (1 for mono, 2 for stereo)
            chunk_size: Audio chunk size (samples per chunk)
            device_index: Specific audio device index to use (or None for default)
        """
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        # Initialize state
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        
        # Queues for synchronized AV data
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames (1 second at 30fps)
        self.audio_queue = queue.Queue(maxsize=30)  # Buffer up to 30 audio chunks
        
        # Thread management
        self.video_thread = None
        self.audio_thread = None
        
        # Latest captured data
        self.latest_frame = None
        self.latest_audio = None
        
        # Logger
        self.logger = logging.getLogger('AVCapture')
        self.logger.setLevel(logging.INFO)
    
    def start(self) -> bool:
        """
        Start capturing audio and video.
        
        Returns:
            Whether capture was successfully started
        """
        if self.running:
            self.logger.warning("Capture already running")
            return False
        
        # Try default camera index
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Release any partial capture and try DirectShow backend on camera index 0
            try:
                self.cap.release()
            except:
                pass
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
            if not self.cap.isOpened():
                self.logger.error("Failed to open camera with index 0 or DirectShow backend")
                return False
        
        # Configure camera properties explicitly
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Check if video capture opened successfully
        if not self.cap.isOpened():
            self.logger.error("Failed to configure video capture")
            return False
        
        # Initialize audio capture
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
                stream_callback=self._audio_callback
            )
            # Explicitly start the audio stream so callbacks are invoked
            self.stream.start_stream()
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
            self.cap.release()
            return False
        
        # Start capture threads
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        self.video_thread = threading.Thread(target=self._video_capture_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        self.logger.info("AV capture started successfully")
        return True
    
    def stop(self) -> None:
        """Stop capturing audio and video."""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for video thread to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        
        # Clean up resources
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        self.logger.info(f"AV capture stopped after {self.frame_count} frames")
    
    def get_av_pair(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a synchronized audio-video pair.
        
        Args:
            block: Whether to block until data is available
            timeout: Maximum time to wait for data
            
        Returns:
            Tuple of (video_frame, audio_chunk) or (None, None) if not available
        """
        try:
            # Try to get a frame and corresponding audio
            frame = self.frame_queue.get(block=block, timeout=timeout)
            audio = self.audio_queue.get(block=block, timeout=timeout)
            
            self.latest_frame = frame
            self.latest_audio = audio
            
            return frame, audio
        except queue.Empty:
            # Queue is empty and would block, or timed out
            return None, None
    
    def get_frame(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the next video frame.
        
        Args:
            block: Whether to block until data is available
            timeout: Maximum time to wait for data
            
        Returns:
            Video frame or None if not available
        """
        try:
            frame = self.frame_queue.get(block=block, timeout=timeout)
            self.latest_frame = frame
            return frame
        except queue.Empty:
            return None
    
    def get_audio(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the next audio chunk.
        
        Args:
            block: Whether to block until data is available
            timeout: Maximum time to wait for data
            
        Returns:
            Audio data or None if not available
        """
        try:
            audio = self.audio_queue.get(block=block, timeout=timeout)
            self.latest_audio = audio
            return audio
        except queue.Empty:
            return None
    
    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the latest captured audio and video data.
        
        Returns:
            Tuple of (latest_frame, latest_audio)
        """
        return self.latest_frame, self.latest_audio
    
    def _video_capture_loop(self) -> None:
        """Video capture thread function."""
        prev_time = time.time()
        use_test_pattern = False
        test_pattern_counter = 0
        retry_camera_count = 0  # Counter for periodic camera retry
        
        # Try to read at least one frame on startup to confirm camera works
        ret, first_frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to read initial frame from camera, switching to test pattern mode")
            use_test_pattern = True
        else:
            self.logger.info(f"Camera successfully initialized with frame size: {first_frame.shape}")
            # Process and queue the first frame
            frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            try:
                self.frame_queue.put(frame_rgb, block=False)
            except queue.Full:
                pass
        
        # Main capture loop
        while self.running:
            # Calculate time to wait for target FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            
            if elapsed < self.frame_interval:
                # Sleep to maintain target frame rate
                time.sleep(self.frame_interval - elapsed)
            
            # Periodically retry camera if using test pattern
            if use_test_pattern:
                retry_camera_count += 1
                # Try to reconnect to camera every 60 frames (about 2 seconds at 30fps)
                if retry_camera_count >= 60:
                    self.logger.info("Retrying camera connection...")
                    retry_camera_count = 0
                    # Release any existing capture
                    self.cap.release()
                    # Try to reconnect with DirectShow backend
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        self.logger.info("Successfully reconnected to camera!")
                        use_test_pattern = False
                    else:
                        self.logger.warning("Camera reconnection failed, continuing with test pattern")
            
            if not use_test_pattern:
                # Capture frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to capture video frame, retrying...")
                    # Try to reinitialize camera if capture fails
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(0)
                    
                    # Try one more time
                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.error("Camera capture still failing, switching to test pattern mode")
                        use_test_pattern = True
                    else:
                        # Convert to RGB (OpenCV captures in BGR)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # Convert to RGB (OpenCV captures in BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if use_test_pattern:
                # Generate a test pattern when camera fails
                test_pattern_counter += 1
                
                # Create a simple animated test pattern
                frame_rgb = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                
                # Draw a grid of white dots
                frame_rgb[::20, ::20] = 255
                
                # Add moving colored shapes
                pos = test_pattern_counter % 100
                size = 40
                
                # Red square
                x1, y1 = pos, pos
                x2, y2 = min(x1 + size, self.video_width-1), min(y1 + size, self.video_height-1)
                frame_rgb[y1:y2, x1:x2, 0] = 255
                
                # Green circle
                center_x = self.video_width // 2 + int(100 * np.sin(test_pattern_counter * 0.1))
                center_y = self.video_height // 2
                for y in range(max(0, center_y - size), min(self.video_height, center_y + size)):
                    for x in range(max(0, center_x - size), min(self.video_width, center_x + size)):
                        if (x - center_x)**2 + (y - center_y)**2 < size**2:
                            frame_rgb[y, x, 1] = 255
                
                # Add frame counter text
                fontScale = 1.0
                cv2.putText(frame_rgb, f"TEST FRAME {test_pattern_counter}", 
                           (20, self.video_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
                
                if test_pattern_counter % 30 == 0:
                    self.logger.info(f"Generated test pattern frame #{test_pattern_counter}")
            
            # DEBUG: Log frame details every 100 frames
            if self.frame_count % 100 == 0:
                self.logger.debug(f"Captured frame {self.frame_count}: shape={frame_rgb.shape}, dtype={frame_rgb.dtype}, range=[{np.min(frame_rgb)}, {np.max(frame_rgb)}]")
            
            # Try to put frame in the queue
            try:
                self.frame_queue.put(frame_rgb, block=False)
                self.latest_frame = frame_rgb  # Also update latest_frame directly
            except queue.Full:
                # Queue is full, discard oldest frame and try again
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame_rgb, block=False)
                    self.latest_frame = frame_rgb
                except:
                    pass
            
            prev_time = time.time()
            self.frame_count += 1
    
    def _audio_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """
        Callback for audio stream processing.
        
        This is called by the PyAudio library when new audio data is available.
        """
        if not self.running:
            return (in_data, pyaudio.paComplete)
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Always use real microphone data directly (silent-input fallback removed)
        
        # DEBUG: Log audio details (but not too frequently)
        if self.frame_count % 100 == 0:  # Log less frequently
            self.logger.debug(f"Audio data: shape={audio_data.shape}, dtype={audio_data.dtype}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        
        # Try to put audio data in the queue
        try:
            self.audio_queue.put(audio_data, block=False)
            self.latest_audio = audio_data  # Also update latest_audio directly
        except queue.Full:
            # Queue is full, discard oldest chunk and try again
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(audio_data, block=False)
                self.latest_audio = audio_data
            except:
                pass
        
        return (in_data, pyaudio.paContinue)
    
    def __enter__(self):
        """Context manager entry method."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop()


class VideoFileReader:
    """
    Utility to read video files and their audio streams.
    
    This provides a similar interface to AVCapture, but reads from a file.
    """
    
    def __init__(
        self,
        video_file: str,
        target_fps: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ):
        """
        Initialize the video file reader.
        
        Args:
            video_file: Path to the video file
            target_fps: Target FPS to convert to (or None to keep original)
            target_width: Target width to resize to (or None to keep original)
            target_height: Target height to resize to (or None to keep original)
        """
        self.video_file = video_file
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        
        # Initialize state
        self.running = False
        self.frame_count = 0
        self.frame_index = 0
        
        # Store video data
        self.frames = []
        self.audio_waveform = None
        self.sample_rate = None
        
        # Logger
        self.logger = logging.getLogger('VideoFileReader')
        self.logger.setLevel(logging.INFO)
    
    def load(self) -> bool:
        """
        Load the video file into memory.
        
        Returns:
            Whether loading was successful
        """
        try:
            self.logger.info(f"Loading video file: {self.video_file}")
            
            # Open video file
            cap = cv2.VideoCapture(self.video_file)
            
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {self.video_file}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Calculate target properties
            target_fps = self.target_fps if self.target_fps else fps
            target_width = self.target_width if self.target_width else width
            target_height = self.target_height if self.target_height else height
            
            # Read all frames
            self.frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize if needed
                if target_width != width or target_height != height:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.frames.append(frame_rgb)
            
            cap.release()
            
            self.frame_count = len(self.frames)
            self.logger.info(f"Loaded {self.frame_count} frames")
            
            # For audio, we would need a library like moviepy or librosa
            # to extract the audio stream. This is a simplified placeholder
            self.audio_waveform = np.zeros((self.frame_count * 1024,))  # Dummy audio
            self.sample_rate = 22050
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading video file: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the video playback.
        
        Returns:
            Whether starting was successful
        """
        if not self.frames:
            success = self.load()
            if not success:
                return False
        
        self.running = True
        self.frame_index = 0
        return True
    
    def stop(self) -> None:
        """Stop the video playback."""
        self.running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next video frame.
        
        Returns:
            Video frame or None if at the end
        """
        if not self.running or self.frame_index >= self.frame_count:
            return None
        
        frame = self.frames[self.frame_index]
        self.frame_index += 1
        return frame
    
    def get_audio_waveform(self) -> Tuple[np.ndarray, int]:
        """
        Get the complete audio waveform.
        
        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        return self.audio_waveform, self.sample_rate
    
    def reset(self) -> None:
        """Reset to the beginning of the video."""
        self.frame_index = 0
    
    def get_all_frames(self) -> List[np.ndarray]:
        """
        Get all video frames.
        
        Returns:
            List of all video frames
        """
        return self.frames
    
    def __enter__(self):
        """Context manager entry method."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop()

    def _capture_thread(self):
        """
        Thread for capturing frames from the camera.
        """
        self.logger.info(f"Starting video capture thread with camera {self.camera_id}")
        
        # Initialize OpenCV video capture
        cap = cv2.VideoCapture(self.camera_id)
        
        # Try to set properties if explicitly specified
        if self.video_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        if self.video_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual capture properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera properties: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        # Initialize test pattern variables if we need them
        blank_frame_count = 0
        test_pattern_active = False
        test_frame_counter = 0
        
        # Calculate frame interval for desired FPS
        target_interval = 1.0 / (self.fps if self.fps else actual_fps)
        last_frame_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            current_time = time.time()
            
            # Check if we should create a test pattern
            if not ret or frame is None or (frame.size == 0) or np.mean(frame) < 5.0:
                blank_frame_count += 1
                
                if blank_frame_count > 10 and not test_pattern_active:
                    self.logger.warning(f"Camera {self.camera_id} not providing valid frames, using test patterns")
                    test_pattern_active = True
                
                if test_pattern_active:
                    # Create a test pattern
                    frame = self._generate_test_frame(test_frame_counter, actual_width, actual_height)
                    test_frame_counter += 1
                    ret = True
            else:
                # Reset counters when we get valid frames
                blank_frame_count = 0
                if test_pattern_active:
                    test_pattern_active = False
                    self.logger.info(f"Camera {self.camera_id} is now providing valid frames")
            
            # Only process if a frame was captured
            if ret:
                # Increment frame counter
                self.frame_count += 1
                
                # Limit frame rate if needed
                elapsed = current_time - last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                
                # Store the captured frame
                try:
                    self.video_queue.put(frame, block=False)
                    self.latest_frame = frame  # Also update latest_frame directly
                except queue.Full:
                    # Queue is full, discard oldest frame and try again
                    try:
                        self.video_queue.get_nowait()
                        self.video_queue.put(frame, block=False)
                        self.latest_frame = frame  # Also update latest_frame directly
                    except:
                        pass
                
                last_frame_time = time.time()
                
                # Log frame details occasionally
                if self.frame_count % 100 == 0:  # Log less frequently
                    self.logger.debug(f"Video frame #{self.frame_count}: shape={frame.shape}, type={frame.dtype}, mean={np.mean(frame):.1f}")
            else:
                self.logger.warning(f"Failed to capture frame from camera {self.camera_id}")
                time.sleep(0.1)  # Avoid CPU spin if camera is disconnected
        
        # Release the camera when done
        cap.release()
        self.logger.info("Video capture thread stopped")

    def _generate_test_frame(self, frame_counter, width, height):
        """
        Generate a test pattern frame when the camera is not working.
        
        Args:
            frame_counter: Counter used to animate the pattern
            width: Width of the frame to generate
            height: Height of the frame to generate
            
        Returns:
            A test pattern frame of the specified dimensions
        """
        # Ensure width and height are valid
        width = width if width > 100 else 640
        height = height if height > 100 else 480
        
        # Create different test patterns that cycle
        pattern_type = (frame_counter // 150) % 4
        
        if pattern_type == 0:
            # Colored bars with moving diagonal line
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Create color bars
            bar_width = width // 6
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (0, 255, 255),  # Cyan
                (255, 0, 255),  # Magenta
            ]
            for i, color in enumerate(colors):
                x1 = i * bar_width
                x2 = (i + 1) * bar_width
                frame[:, x1:x2] = color
                
            # Add diagonal moving line
            line_pos = (frame_counter * 5) % (width + height)
            for i in range(height):
                j = line_pos - i
                if 0 <= j < width:
                    frame[i, j] = (255, 255, 255)
                    
        elif pattern_type == 1:
            # Checkerboard pattern with moving squares
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            square_size = 40
            offset = frame_counter % (2 * square_size)
            
            for y in range(-square_size, height, 2 * square_size):
                for x in range(-square_size, width, 2 * square_size):
                    y1 = max(0, y + offset)
                    y2 = min(height, y + square_size + offset)
                    x1 = max(0, x + offset)
                    x2 = min(width, x + square_size + offset)
                    if y1 < y2 and x1 < x2:
                        frame[y1:y2, x1:x2] = (255, 255, 255)
                        
            for y in range(0, height, 2 * square_size):
                for x in range(0, width, 2 * square_size):
                    y1 = max(0, y + offset)
                    y2 = min(height, y + square_size + offset)
                    x1 = max(0, x + offset)
                    x2 = min(width, x + square_size + offset)
                    if y1 < y2 and x1 < x2:
                        frame[y1:y2, x1:x2] = (255, 255, 255)
                        
        elif pattern_type == 2:
            # Concentric circles
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            max_radius = min(width, height) // 2
            
            for r in range(10, max_radius, 30):
                # Shifting colors based on frame counter
                color = [
                    (frame_counter + r) % 255,
                    (frame_counter * 2 + r) % 255,
                    (frame_counter * 3 + r) % 255
                ]
                cv2.circle(frame, (center_x, center_y), r, color, 5)
                
        else:
            # Text with frame counter and timestamp
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw grid lines
            for x in range(0, width, 100):
                cv2.line(frame, (x, 0), (x, height), (0, 80, 0), 1)
            for y in range(0, height, 100):
                cv2.line(frame, (0, y), (width, y), (0, 80, 0), 1)
                
            # Draw frame info
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Test Pattern - No camera signal", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_counter}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {timestamp}", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add a small animated indicator to show the system is working
        indicator_pos = frame_counter % width
        cv2.rectangle(frame, (indicator_pos, height-10), (indicator_pos+20, height), (255, 255, 255), -1)
        
        return frame 