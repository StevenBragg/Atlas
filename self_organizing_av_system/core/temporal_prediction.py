"""
Temporal prediction mechanisms for the self-organizing AV system.

This implements sequence learning, prediction generation, and
predictive coding as described in the architecture document.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    """Prediction modes for temporal learning"""
    FORWARD = "forward"         # Predict future states
    BACKWARD = "backward"       # Predict previous states
    BIDIRECTIONAL = "bidir"     # Predict both directions


class TemporalPrediction:
    """
    Implements temporal prediction mechanisms for sequence learning.
    
    This module enables:
    1. Learning temporal sequences from multimodal input
    2. Predicting future states based on current and past states
    3. Bidirectional prediction for recall/recognition
    4. Integration of prediction errors for continuous learning
    """
    
    def __init__(
        self,
        representation_size: int,
        sequence_length: int = 5,
        prediction_horizon: int = 3,
        learning_rate: float = 0.01,
        prediction_mode: Union[str, PredictionMode] = PredictionMode.FORWARD,
        use_eligibility_trace: bool = True,
        trace_decay: float = 0.7,
        confidence_threshold: float = 0.3,
        enable_surprise_detection: bool = True,
        td_lambda: float = 0.8,
        enable_recurrent_connections: bool = True,
        regularization_strength: float = 0.001,
        surprise_threshold: float = 0.5,
        confidence_learning_rate: float = 0.005,
        prediction_decay: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize temporal prediction module.
        
        Args:
            representation_size: Size of neural representation vectors
            sequence_length: Number of past states to consider for prediction
            prediction_horizon: Number of future states to predict
            learning_rate: Learning rate for temporal weights
            prediction_mode: Forward, backward, or bidirectional prediction
            use_eligibility_trace: Whether to use eligibility traces for learning
            trace_decay: Decay factor for eligibility traces
            confidence_threshold: Threshold for confident predictions
            enable_surprise_detection: Whether to detect surprising events
            td_lambda: TD(λ) parameter for temporal difference learning
            enable_recurrent_connections: Whether to use recurrent connections
            regularization_strength: L2 regularization strength
            surprise_threshold: Threshold for surprise detection
            confidence_learning_rate: Learning rate for confidence estimation
            prediction_decay: Decay factor for prediction influence
            random_seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.representation_size = representation_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        
        # Convert string to enum if needed
        if isinstance(prediction_mode, str):
            prediction_mode = PredictionMode(prediction_mode)
        self.prediction_mode = prediction_mode
        
        self.use_eligibility_trace = use_eligibility_trace
        self.trace_decay = trace_decay
        self.confidence_threshold = confidence_threshold
        self.enable_surprise_detection = enable_surprise_detection
        self.td_lambda = td_lambda
        self.enable_recurrent_connections = enable_recurrent_connections
        self.regularization_strength = regularization_strength
        self.surprise_threshold = surprise_threshold
        self.confidence_learning_rate = confidence_learning_rate
        self.prediction_decay = prediction_decay
        
        # Initialize temporal weights
        # Forward prediction weights: predict from t to t+n
        self.forward_weights = {}
        for t in range(1, self.prediction_horizon + 1):
            # Initialize with small random values
            self.forward_weights[t] = np.random.normal(
                0, 0.01, (self.representation_size, self.representation_size)
            )
        
        # Backward prediction weights if needed
        self.backward_weights = {}
        if self.prediction_mode in [PredictionMode.BACKWARD, PredictionMode.BIDIRECTIONAL]:
            for t in range(1, self.sequence_length + 1):
                self.backward_weights[t] = np.random.normal(
                    0, 0.01, (self.representation_size, self.representation_size)
                )
        
        # Recurrent connections (state -> state)
        self.recurrent_weights = None
        if self.enable_recurrent_connections:
            self.recurrent_weights = np.random.normal(
                0, 0.01, (self.representation_size, self.representation_size)
            )
            # Zero out the diagonal (no self-connections)
            np.fill_diagonal(self.recurrent_weights, 0)
        
        # Confidence estimation weights
        # Used to predict confidence of each prediction
        self.confidence_weights = {}
        for t in range(1, self.prediction_horizon + 1):
            self.confidence_weights[t] = np.random.normal(
                0, 0.01, (self.representation_size, 1)
            )
        
        # Initialize memory buffers
        self.state_buffer = []  # Store past states
        self.prediction_buffer = []  # Store past predictions
        self.confidence_buffer = []  # Store confidence values
        self.surprise_buffer = []  # Store surprise signals
        
        # Eligibility traces
        self.eligibility_traces = {}
        if self.use_eligibility_trace:
            for t in range(1, self.prediction_horizon + 1):
                self.eligibility_traces[t] = np.zeros((self.representation_size, self.representation_size))
        
        # Performance tracking
        self.prediction_errors = []  # Track prediction errors
        self.mean_prediction_error = 0.0
        self.surprise_count = 0
        self.update_count = 0
        
        logger.info(f"Initialized temporal prediction: "
                   f"representation_size={representation_size}, "
                   f"sequence_length={sequence_length}, "
                   f"prediction_horizon={prediction_horizon}, "
                   f"mode={prediction_mode.value}")
    
    def update(
        self,
        current_state: np.ndarray,
        target_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Update the temporal prediction model with a new state.
        
        Args:
            current_state: Current neural representation
            target_state: Optional target state for supervised training
            
        Returns:
            Dictionary with prediction results
        """
        self.update_count += 1
        
        # Ensure the state is the right shape
        if len(current_state.shape) == 1:
            current_state = current_state.reshape(-1)
        
        # Normalize the state if needed
        if np.sum(current_state) > 0:
            normalized_state = current_state / np.max([np.max(np.abs(current_state)), 1e-10])
        else:
            normalized_state = current_state.copy()
        
        # Update state buffer
        self.state_buffer.append(normalized_state)
        if len(self.state_buffer) > self.sequence_length + self.prediction_horizon:
            self.state_buffer.pop(0)
        
        # Initialize results dictionary
        result = {
            "predictions": {},
            "confidence": {},
            "surprise": 0.0,
            "prediction_error": 0.0,
            "is_surprising": False
        }
        
        # Generate predictions for future states
        predictions = {}
        confidence = {}
        
        # Only predict if we have enough states in the buffer
        if len(self.state_buffer) > 1:
            # Generate predictions for each horizon
            for t in range(1, self.prediction_horizon + 1):
                if t < len(self.forward_weights) + 1:
                    # Predict future state
                    pred, conf = self._predict_state(normalized_state, t)
                    predictions[t] = pred
                    confidence[t] = conf
                    
                    # Store in result
                    result["predictions"][t] = pred
                    result["confidence"][t] = float(conf)
            
            # Store predictions for later evaluation
            self.prediction_buffer.append(predictions)
            self.confidence_buffer.append(confidence)
            
            # Trim buffers if too long
            if len(self.prediction_buffer) > self.sequence_length + self.prediction_horizon:
                self.prediction_buffer.pop(0)
            if len(self.confidence_buffer) > self.sequence_length + self.prediction_horizon:
                self.confidence_buffer.pop(0)
            
            # Update the model by comparing past predictions with actual states
            prediction_error = self._update_weights()
            result["prediction_error"] = float(prediction_error)
            
            # Detect surprising events
            if self.enable_surprise_detection:
                surprise_signal = self._detect_surprise(normalized_state)
                result["surprise"] = float(surprise_signal)
                result["is_surprising"] = surprise_signal > self.surprise_threshold
                
                # Store surprise signal
                self.surprise_buffer.append(surprise_signal)
                if len(self.surprise_buffer) > self.sequence_length:
                    self.surprise_buffer.pop(0)
                
                # Count surprising events
                if result["is_surprising"]:
                    self.surprise_count += 1
        
        return result
    
    def _predict_state(
        self,
        current_state: np.ndarray,
        time_offset: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict state at t+time_offset based on current state.
        
        Args:
            current_state: Current neural representation
            time_offset: Time steps into future to predict
            
        Returns:
            Tuple of (predicted_state, confidence)
        """
        # Basic prediction using weights
        if current_state.shape[0] != self.forward_weights[time_offset].shape[1]:
            logger.warning(f"Shape mismatch in _predict_state: current_state shape {current_state.shape}, "
                         f"weights shape {self.forward_weights[time_offset].shape}")
            
            # Resize current_state to match weight dimensions
            if current_state.shape[0] < self.forward_weights[time_offset].shape[1]:
                # Pad with zeros
                current_state = np.pad(current_state, (0, self.forward_weights[time_offset].shape[1] - current_state.shape[0]))
            else:
                # Truncate
                current_state = current_state[:self.forward_weights[time_offset].shape[1]]
        
        prediction = np.dot(self.forward_weights[time_offset], current_state)
        
        # Apply recurrent connections if enabled
        if self.enable_recurrent_connections and self.recurrent_weights is not None:
            # Check dimensions
            if current_state.shape[0] != self.recurrent_weights.shape[1]:
                logger.warning(f"Recurrent weight mismatch in _predict_state: current_state shape {current_state.shape}, "
                             f"recurrent_weights shape {self.recurrent_weights.shape}")
                
                # Resize current_state to match recurrent weight dimensions
                if current_state.shape[0] < self.recurrent_weights.shape[1]:
                    # Pad with zeros
                    current_state = np.pad(current_state, (0, self.recurrent_weights.shape[1] - current_state.shape[0]))
                else:
                    # Truncate
                    current_state = current_state[:self.recurrent_weights.shape[1]]
            
            recurrent_contribution = np.dot(self.recurrent_weights, current_state)
            
            # Check if prediction and recurrent_contribution have compatible shapes
            if prediction.shape[0] != recurrent_contribution.shape[0]:
                logger.warning(f"Shape mismatch between prediction {prediction.shape} and recurrent_contribution {recurrent_contribution.shape}")
                
                # Make them compatible
                min_size = min(prediction.shape[0], recurrent_contribution.shape[0])
                prediction = prediction[:min_size]
                recurrent_contribution = recurrent_contribution[:min_size]
            
            # Mix with base prediction
            prediction = prediction + self.prediction_decay * recurrent_contribution
        
        # Calculate prediction confidence
        if prediction.shape[0] != self.confidence_weights[time_offset].shape[0]:
            logger.warning(f"Confidence weight mismatch: prediction shape {prediction.shape}, "
                         f"confidence_weights shape {self.confidence_weights[time_offset].shape}")
            
            # Resize to match
            min_size = min(prediction.shape[0], self.confidence_weights[time_offset].shape[0])
            confidence_inputs = np.abs(prediction[:min_size])
            confidence_weights = self.confidence_weights[time_offset][:min_size]
            confidence = float(np.tanh(np.mean(np.dot(confidence_inputs, confidence_weights))))
        else:
            confidence_inputs = np.abs(prediction)
            confidence = float(np.tanh(np.mean(np.dot(confidence_inputs, self.confidence_weights[time_offset]))))
        
        # Normalize prediction
        if np.sum(np.abs(prediction)) > 0:
            prediction = prediction / np.max([np.max(np.abs(prediction)), 1e-10])
        
        return prediction, confidence
    
    def _update_weights(self) -> float:
        """
        Update weights based on prediction errors.
        
        Returns:
            Average prediction error
        """
        total_error = 0.0
        update_count = 0
        
        # Check if we have enough history to update
        if len(self.state_buffer) <= self.prediction_horizon or len(self.prediction_buffer) == 0:
            return 0.0
        
        # Update weights for each prediction horizon
        for t in range(1, self.prediction_horizon + 1):
            # Check if we have predictions and actual states for this horizon
            if len(self.state_buffer) > t and len(self.prediction_buffer) >= t:
                # Get the prediction made t steps ago for the current state
                prediction_idx = -t
                state_idx = -1
                
                if abs(prediction_idx) <= len(self.prediction_buffer):
                    past_prediction = self.prediction_buffer[prediction_idx].get(t, None)
                    if past_prediction is not None:
                        # Get actual state
                        actual_state = self.state_buffer[state_idx]
                        
                        # Get prediction made for this state
                        past_prediction = self.prediction_buffer[prediction_idx].get(t, None)
                        
                        # Make sure actual_state and past_prediction have compatible shapes
                        if len(actual_state) != len(past_prediction):
                            logger.warning(f"Shape mismatch between actual_state {actual_state.shape} and past_prediction {past_prediction.shape}")
                            
                            # Resize to make them compatible
                            min_size = min(len(actual_state), len(past_prediction))
                            if len(actual_state) > min_size:
                                actual_state = actual_state[:min_size]
                            if len(past_prediction) > min_size:
                                past_prediction = past_prediction[:min_size]
                        
                        # Calculate error
                        error = actual_state - past_prediction
                        error_magnitude = np.mean(np.abs(error))
                        total_error += error_magnitude
                        update_count += 1
                        
                        # Get state that was used to make prediction
                        input_state = self.state_buffer[state_idx - t]
                        
                        # Check for dimension mismatch in weights
                        if self.forward_weights[t].shape != (len(actual_state), len(input_state)):
                            # Resize weights to match current dimensions
                            logger.warning(f"Dimension mismatch in weights during update: "
                                          f"weights shape {self.forward_weights[t].shape}, "
                                          f"should be ({len(actual_state)}, {len(input_state)})")
                            
                            # Create new weights with proper dimensions
                            new_weights = np.random.normal(
                                0, 0.01, (len(actual_state), len(input_state))
                            )
                            
                            # Copy existing weights where possible
                            min_rows = min(self.forward_weights[t].shape[0], len(actual_state))
                            min_cols = min(self.forward_weights[t].shape[1], len(input_state))
                            new_weights[:min_rows, :min_cols] = self.forward_weights[t][:min_rows, :min_cols]
                            
                            # Update weights
                            self.forward_weights[t] = new_weights
                        
                        # Make sure error and input_state match the dimensions of forward_weights
                        if len(error) != self.forward_weights[t].shape[0]:
                            if len(error) < self.forward_weights[t].shape[0]:
                                # Pad with zeros
                                error = np.pad(error, (0, self.forward_weights[t].shape[0] - len(error)))
                            else:
                                # Truncate
                                error = error[:self.forward_weights[t].shape[0]]
                        
                        if len(input_state) != self.forward_weights[t].shape[1]:
                            if len(input_state) < self.forward_weights[t].shape[1]:
                                # Pad with zeros
                                input_state = np.pad(input_state, (0, self.forward_weights[t].shape[1] - len(input_state)))
                            else:
                                # Truncate
                                input_state = input_state[:self.forward_weights[t].shape[1]]
                        
                        # Update forward weights using error signal
                        delta_w = self.learning_rate * np.outer(error, input_state)
                        
                        # Apply regularization
                        delta_w -= self.regularization_strength * self.forward_weights[t]
                        
                        # Update weights
                        if self.use_eligibility_trace:
                            # Check eligibility traces dimensions
                            if self.eligibility_traces[t].shape != self.forward_weights[t].shape:
                                logger.warning(f"Eligibility trace shape mismatch: "
                                              f"trace shape {self.eligibility_traces[t].shape}, "
                                              f"weights shape {self.forward_weights[t].shape}")
                                
                                # Create new trace with proper dimensions
                                new_trace = np.zeros(self.forward_weights[t].shape)
                                # Copy existing trace where possible
                                min_rows = min(self.eligibility_traces[t].shape[0], self.forward_weights[t].shape[0])
                                min_cols = min(self.eligibility_traces[t].shape[1], self.forward_weights[t].shape[1])
                                new_trace[:min_rows, :min_cols] = self.eligibility_traces[t][:min_rows, :min_cols]
                                self.eligibility_traces[t] = new_trace
                            
                            # Update eligibility trace
                            self.eligibility_traces[t] = (
                                self.trace_decay * self.eligibility_traces[t] + 
                                np.outer(error, input_state)
                            )
                            # Apply update with trace
                            self.forward_weights[t] += self.learning_rate * self.eligibility_traces[t]
                        else:
                            # Direct update
                            self.forward_weights[t] += delta_w
                        
                        # Update confidence weights
                        confidence = self.confidence_buffer[prediction_idx].get(t, 0.0)
                        confidence_error = 1.0 - error_magnitude - confidence
                        delta_conf = self.confidence_learning_rate * confidence_error * np.abs(past_prediction).reshape(-1, 1)
                        self.confidence_weights[t] += delta_conf
        
        # Calculate average error
        if update_count > 0:
            avg_error = total_error / update_count
            
            # Update running average
            alpha = 0.1  # Smoothing factor
            self.mean_prediction_error = (1 - alpha) * self.mean_prediction_error + alpha * avg_error
            
            # Store error
            self.prediction_errors.append((self.update_count, avg_error))
            if len(self.prediction_errors) > 1000:
                self.prediction_errors = self.prediction_errors[-1000:]
            
            return avg_error
        
        return 0.0
    
    def _detect_surprise(self, current_state: np.ndarray) -> float:
        """
        Detect surprising events based on prediction errors.
        
        Args:
            current_state: Current neural representation
            
        Returns:
            Surprise signal (0-1)
        """
        # Check if we have predictions to compare with the current state
        if len(self.prediction_buffer) == 0:
            return 0.0
        
        # Get predictions for the current state
        surprise_signals = []
        
        # Check predictions for different horizons
        for t in range(1, min(self.prediction_horizon, len(self.prediction_buffer)) + 1):
            prediction_idx = -t
            
            if abs(prediction_idx) < len(self.prediction_buffer):
                # Get prediction made t steps ago
                past_prediction = self.prediction_buffer[prediction_idx].get(t, None)
                
                if past_prediction is not None:
                    # Check for dimension mismatch
                    if len(current_state) != len(past_prediction):
                        logger.warning(f"Shape mismatch in _detect_surprise: current_state shape {current_state.shape}, past_prediction shape {past_prediction.shape}")
                        
                        # Resize to make them compatible
                        min_size = min(len(current_state), len(past_prediction))
                        if len(current_state) > min_size:
                            current_state_resized = current_state[:min_size]
                        else:
                            current_state_resized = current_state
                            
                        if len(past_prediction) > min_size:
                            past_prediction = past_prediction[:min_size]
                    else:
                        current_state_resized = current_state
                    
                    # Calculate difference between prediction and actual state
                    prediction_error = np.mean(np.abs(current_state_resized - past_prediction))
                    
                    # Get confidence for this prediction
                    confidence = self.confidence_buffer[prediction_idx].get(t, 0.5)
                    
                    # Higher confidence and higher error = more surprise
                    surprise = confidence * prediction_error
                    surprise_signals.append(surprise)
        
        # Return maximum surprise across different horizons
        if surprise_signals:
            return float(np.max(surprise_signals))
        
        return 0.0
    
    def predict_future(
        self,
        current_state: np.ndarray,
        steps: int = None,
        include_confidence: bool = True
    ) -> Dict[int, Union[np.ndarray, Tuple[np.ndarray, float]]]:
        """
        Predict future states from current state.
        
        Args:
            current_state: Current neural representation
            steps: Number of steps to predict (None for all possible)
            include_confidence: Whether to include confidence values
            
        Returns:
            Dictionary mapping steps to predictions (and confidence if requested)
        """
        if steps is None:
            steps = self.prediction_horizon
        
        # Ensure state is normalized
        if np.sum(current_state) > 0:
            state = current_state / np.max([np.max(np.abs(current_state)), 1e-10])
        else:
            state = current_state.copy()
        
        predictions = {}
        
        # Direct prediction for each horizon
        for t in range(1, min(steps, self.prediction_horizon) + 1):
            pred, conf = self._predict_state(state, t)
            
            if include_confidence:
                predictions[t] = (pred, float(conf))
            else:
                predictions[t] = pred
        
        return predictions
    
    def predict_sequence(
        self,
        initial_state: np.ndarray,
        length: int,
        include_confidence: bool = True
    ) -> List[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        """
        Generate a sequence of predictions by feeding predictions back as input.
        
        Args:
            initial_state: Initial neural representation
            length: Length of sequence to generate
            include_confidence: Whether to include confidence values
            
        Returns:
            List of predicted states (and confidences if requested)
        """
        sequence = []
        current = initial_state.copy()
        
        # Normalize initial state
        if np.sum(current) > 0:
            current = current / np.max([np.max(np.abs(current)), 1e-10])
        
        # Generate sequence by feeding each prediction back as input
        for _ in range(length):
            # Predict next state
            pred, conf = self._predict_state(current, 1)
            
            if include_confidence:
                sequence.append((pred, float(conf)))
            else:
                sequence.append(pred)
            
            # Use prediction as next input
            current = pred
        
        return sequence
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about prediction performance.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate prediction accuracy
        recent_errors = [err for _, err in self.prediction_errors[-100:]] if self.prediction_errors else [0]
        
        # Calculate average accuracy
        avg_error = np.mean(recent_errors) if recent_errors else 0
        error_trend = 0
        
        if len(recent_errors) > 10:
            # Calculate trend (positive means improving)
            first_half = np.mean(recent_errors[:len(recent_errors)//2])
            second_half = np.mean(recent_errors[len(recent_errors)//2:])
            error_trend = first_half - second_half
        
        stats = {
            'representation_size': self.representation_size,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'prediction_mode': self.prediction_mode.value,
            'mean_prediction_error': float(self.mean_prediction_error),
            'surprise_rate': self.surprise_count / max(1, self.update_count),
            'error_trend': float(error_trend),
            'update_count': self.update_count,
            'buffer_size': len(self.state_buffer)
        }
        
        return stats
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the temporal prediction state for saving.
        
        Returns:
            Dictionary with serialized state
        """
        # Convert weights and traces to lists
        forward_weights_serialized = {}
        for t, w in self.forward_weights.items():
            forward_weights_serialized[str(t)] = w.tolist()
        
        backward_weights_serialized = {}
        for t, w in self.backward_weights.items():
            backward_weights_serialized[str(t)] = w.tolist()
        
        confidence_weights_serialized = {}
        for t, w in self.confidence_weights.items():
            confidence_weights_serialized[str(t)] = w.tolist()
        
        eligibility_traces_serialized = {}
        for t, e in self.eligibility_traces.items():
            eligibility_traces_serialized[str(t)] = e.tolist()
        
        # Create serialized data
        data = {
            'representation_size': self.representation_size,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'learning_rate': self.learning_rate,
            'prediction_mode': self.prediction_mode.value,
            'use_eligibility_trace': self.use_eligibility_trace,
            'trace_decay': self.trace_decay,
            'confidence_threshold': self.confidence_threshold,
            'enable_surprise_detection': self.enable_surprise_detection,
            'forward_weights': forward_weights_serialized,
            'backward_weights': backward_weights_serialized,
            'confidence_weights': confidence_weights_serialized,
            'eligibility_traces': eligibility_traces_serialized,
            'mean_prediction_error': float(self.mean_prediction_error),
            'surprise_count': self.surprise_count,
            'update_count': self.update_count,
            'recurrent_weights': self.recurrent_weights.tolist() if self.recurrent_weights is not None else None
        }
        
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'TemporalPrediction':
        """
        Create a temporal prediction instance from serialized data.
        
        Args:
            data: Dictionary with serialized state
            
        Returns:
            TemporalPrediction instance
        """
        instance = cls(
            representation_size=data['representation_size'],
            sequence_length=data['sequence_length'],
            prediction_horizon=data['prediction_horizon'],
            learning_rate=data['learning_rate'],
            prediction_mode=data['prediction_mode'],
            use_eligibility_trace=data['use_eligibility_trace'],
            trace_decay=data['trace_decay'],
            confidence_threshold=data['confidence_threshold'],
            enable_surprise_detection=data['enable_surprise_detection']
        )
        
        # Restore weights
        for t_str, w_list in data['forward_weights'].items():
            t = int(t_str)
            instance.forward_weights[t] = np.array(w_list)
        
        for t_str, w_list in data['backward_weights'].items():
            t = int(t_str)
            instance.backward_weights[t] = np.array(w_list)
        
        for t_str, w_list in data['confidence_weights'].items():
            t = int(t_str)
            instance.confidence_weights[t] = np.array(w_list)
        
        if data.get('eligibility_traces'):
            for t_str, e_list in data['eligibility_traces'].items():
                t = int(t_str)
                instance.eligibility_traces[t] = np.array(e_list)
        
        if data.get('recurrent_weights'):
            instance.recurrent_weights = np.array(data['recurrent_weights'])
        
        # Restore state
        instance.mean_prediction_error = data['mean_prediction_error']
        instance.surprise_count = data['surprise_count']
        instance.update_count = data['update_count']
        
        return instance 