import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional, Union
import logging

# Import directly from the pathway module instead of through core
from self_organizing_av_system.core.pathway import NeuralPathway


class AudioProcessor:
    """
    Processes audio signals and feeds them through an auditory neural pathway.
    
    This module handles preprocessing of raw audio waveforms into suitable inputs
    for the self-organizing neural network.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        window_size: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        min_freq: int = 50,
        max_freq: int = 8000,
        normalize: bool = True,
        layer_sizes: Optional[List[int]] = None
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel-frequency bands
            min_freq: Minimum frequency for mel bands
            max_freq: Maximum frequency for mel bands
            normalize: Whether to apply normalization
            layer_sizes: List of neuron counts for each neural layer
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.normalize = normalize
        
        # Pre-compute mel filterbank
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=n_mels,
            fmin=min_freq,
            fmax=max_freq
        )
        
        # Configure network layers if not specified
        if layer_sizes is None:
            # Set up a standard hierarchy for auditory processing
            layer_sizes = [
                150,  # First-level features (frequency band detectors)
                75,   # Mid-level features (spectral patterns)
                40    # High-level features (sound objects)
            ]
        
        # Create the neural pathway for audio processing
        self.audio_pathway = NeuralPathway(
            name="audio",
            input_size=self.n_mels,
            layer_sizes=layer_sizes,
            use_recurrent=False  # Disable recurrent connections to improve performance and fix errors
        )
        
        # Tracking statistics
        self.frame_count = 0
        self.current_spectrogram = None
        self.current_frame = None
        self.frame_energy = None
        self.temporal_features = None
        
        # Buffer to handle real-time audio processing
        self.audio_buffer = np.zeros(window_size * 2)  # Double the window size for overlap
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Process a chunk of audio data through the auditory pathway.
        
        Args:
            audio_chunk: Raw audio chunk (mono)
            time_step: Current simulation time step
            
        Returns:
            Activation vector of the top layer of the auditory pathway
        """
        # Update audio buffer with new chunk
        self._update_buffer(audio_chunk)
        
        # Compute spectrogram
        spec_frame = self._compute_spectrogram(self.audio_buffer)
        
        # Process spectrogram through pathway
        activations = self._process_spectrogram_frame(spec_frame, time_step)
        
        # Increment frame count
        self.frame_count += 1
        
        return activations
    
    def process_waveform(self, waveform: np.ndarray, time_step: Optional[int] = None) -> List[np.ndarray]:
        """
        Process an entire audio waveform.
        
        Args:
            waveform: Raw audio waveform (mono)
            time_step: Starting time step
            
        Returns:
            List of activation vectors, one per audio frame
        """
        # Compute full spectrogram
        spectrogram = self._compute_full_spectrogram(waveform)
        self.current_spectrogram = spectrogram
        
        # Process each spectrogram frame
        activations = []
        for i in range(spectrogram.shape[1]):
            spec_frame = spectrogram[:, i]
            frame_activations = self._process_spectrogram_frame(spec_frame, None if time_step is None else time_step + i)
            activations.append(frame_activations)
            
            # Update frame count
            self.frame_count += 1
        
        return activations
    
    def learn(self, learning_rule: str = 'oja') -> None:
        """
        Apply learning to the auditory pathway.
        
        Args:
            learning_rule: Learning rule to use ('hebbian', 'oja', or 'stdp')
        """
        self.audio_pathway.learn(learning_rule)
    
    def _update_buffer(self, audio_chunk: np.ndarray) -> None:
        """
        Update the audio buffer with a new chunk.
        
        Args:
            audio_chunk: New audio data to add to buffer
        """
        # Shift buffer contents
        buffer_size = self.audio_buffer.shape[0]
        chunk_size = min(audio_chunk.shape[0], buffer_size)
        
        # Roll buffer and insert new data
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_size)
        self.audio_buffer[-chunk_size:] = audio_chunk[:chunk_size]
    
    def _compute_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram for a single audio frame.
        
        Args:
            audio_data: Audio data buffer
            
        Returns:
            Mel-spectrogram frame
        """
        # Apply window function
        windowed = audio_data * np.hanning(len(audio_data))
        
        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(windowed, n=self.window_size))
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_fb, spectrum)
        
        # Convert to decibel scale and normalize
        log_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        if self.normalize:
            log_spec = (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec) + 1e-10)
        
        return log_spec
    
    def _compute_full_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram for an entire audio waveform.
        
        Args:
            waveform: Complete audio waveform
            
        Returns:
            Mel-spectrogram (n_mels x n_frames)
        """
        # Compute STFT
        stft = librosa.stft(
            y=waveform,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            window='hann'
        )
        
        # Compute magnitude
        magnitude = np.abs(stft)
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_fb, magnitude)
        
        # Convert to decibel scale
        log_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        if self.normalize:
            log_spec = (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec) + 1e-10)
        
        return log_spec
    
    def _process_spectrogram_frame(self, spec_frame: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Process a single spectrogram frame through the auditory pathway.
        
        Args:
            spec_frame: Mel-spectrogram frame
            time_step: Current simulation time step
            
        Returns:
            Activation vector of the top layer
        """
        # Update current frame
        self.current_frame = spec_frame
        
        # Calculate frame energy (simple sum of spectrogram values)
        self.frame_energy = np.sum(spec_frame)
        
        # Ensure proper shape
        if spec_frame.shape[0] != self.n_mels:
            raise ValueError(f"Spectrogram frame shape {spec_frame.shape} doesn't match expected {self.n_mels}")
        
        # Process through auditory pathway
        activations = self.audio_pathway.process(spec_frame, time_step)
        
        return activations
    
    def _compute_temporal_features(self, spec_history: np.ndarray) -> np.ndarray:
        """
        Compute temporal features from spectrogram history.
        
        Args:
            spec_history: Matrix of recent spectrogram frames
            
        Returns:
            Temporal feature vector
        """
        # Simplified temporal features (spectral flux)
        # Calculate frame-to-frame differences
        diff = np.diff(spec_history, axis=1)
        
        # Take absolute value to capture both onsets and offsets
        abs_diff = np.abs(diff)
        
        # Average across time to get one feature per frequency band
        temporal_features = np.mean(abs_diff, axis=1)
        
        return temporal_features
    
    def get_receptive_fields(self, layer_idx: int = 0) -> np.ndarray:
        """
        Get receptive fields of neurons in the specified layer.
        
        Args:
            layer_idx: Index of the layer to get receptive fields from
            
        Returns:
            Matrix of receptive fields, one per row
        """
        return self.audio_pathway.layers[layer_idx].get_all_receptive_fields()
    
    def visualize_receptive_field(self, layer_idx: int, neuron_idx: int) -> np.ndarray:
        """
        Create a visual representation of a neuron's receptive field.
        
        Args:
            layer_idx: Index of the layer
            neuron_idx: Index of the neuron within the layer
            
        Returns:
            1D frequency response vector
        """
        # Check if indices are valid
        if layer_idx < 0 or layer_idx >= len(self.audio_pathway.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer = self.audio_pathway.layers[layer_idx]
        if neuron_idx < 0 or neuron_idx >= layer.layer_size:
            raise ValueError(f"Invalid neuron index: {neuron_idx}")
        
        # Get the receptive field
        rf = layer.neurons[neuron_idx].get_receptive_field()
        
        # For layer 0, rf directly maps to frequency bands
        if layer_idx == 0:
            return rf  # This is already a frequency response
        else:
            # For higher layers, this is more complex as they respond to patterns
            # This is a simplified placeholder
            return None
    
    def get_pathway_state(self) -> Dict:
        """
        Get the current state of the auditory pathway.
        
        Returns:
            Dictionary with pathway state information
        """
        return self.audio_pathway.get_pathway_state()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the audio processor.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            "frames_processed": self.frame_count,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "window_size": self.window_size,
            "freq_range": f"{self.min_freq}-{self.max_freq}Hz",
            "layer_sizes": [layer.layer_size for layer in self.audio_pathway.layers]
        }
    
    def __repr__(self) -> str:
        return f"AudioProcessor(n_mels={self.n_mels}, freq_range={self.min_freq}-{self.max_freq}Hz)" 