"""
Image Classification Learning System for ATLAS

Implements autonomous image classification learning using biologically-plausible
mechanisms including:
- Unsupervised feature learning through competitive Hebbian learning
- Prototype-based classification with self-organizing category representations
- Meta-learning for optimal strategy selection
- Curriculum learning for progressive difficulty

The system learns to classify images without backpropagation, using only
local learning rules (Hebbian, Oja, competitive) that operate at each
neuron independently.
"""

import numpy as np
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time

from .layer import NeuralLayer
from .pathway import NeuralPathway
from .meta_learning import MetaLearner, LearningStrategy, LearningExperience

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of classifying an image"""
    predicted_class: int
    predicted_label: str
    confidence: float
    class_probabilities: Dict[int, float]
    feature_activations: np.ndarray


@dataclass
class LearningMetrics:
    """Metrics from a learning epoch"""
    epoch: int
    accuracy: float
    loss: float
    feature_sparsity: float
    prototype_distances: Dict[int, float]
    learning_rate: float
    strategy: str
    duration: float


class DatasetType(Enum):
    """Supported dataset types"""
    NUMPY = "numpy"           # .npy or .npz files
    IMAGE_FOLDER = "folder"   # Folder structure with subfolders per class
    PICKLE = "pickle"         # Pickled dataset
    MNIST_FORMAT = "mnist"    # MNIST-style idx format
    CUSTOM = "custom"         # Custom loader


class ImageDataset:
    """
    Dataset handler for image classification.

    Supports loading from various formats and provides iteration,
    batching, and augmentation capabilities.
    """

    def __init__(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        flatten: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            images: Array of images (N, H, W) or (N, H, W, C)
            labels: Array of integer labels
            class_names: Optional list of class names
            normalize: Whether to normalize pixel values to [0, 1]
            flatten: Whether to flatten images to 1D vectors
        """
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.normalize = normalize
        self.flatten = flatten

        if images is not None:
            self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare the data for training."""
        if self.images is None:
            return

        # Ensure float type
        self.images = self.images.astype(np.float32)

        # Normalize if needed
        if self.normalize and self.images.max() > 1.0:
            self.images = self.images / 255.0

        # Store original shape
        self.original_shape = self.images.shape[1:]

        # Flatten if needed
        if self.flatten and len(self.images.shape) > 2:
            n_samples = self.images.shape[0]
            self.images = self.images.reshape(n_samples, -1)

        # Determine number of classes
        if self.labels is not None:
            self.num_classes = len(np.unique(self.labels))
            if self.class_names is None:
                self.class_names = [str(i) for i in range(self.num_classes)]
        else:
            self.num_classes = 0

        logger.info(f"Dataset prepared: {len(self.images)} samples, "
                   f"{self.num_classes} classes, shape {self.images.shape}")

    @classmethod
    def from_numpy(
        cls,
        images_path: str,
        labels_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> 'ImageDataset':
        """
        Load dataset from numpy files.

        Args:
            images_path: Path to .npy or .npz file containing images
            labels_path: Path to .npy file containing labels
            class_names: Optional list of class names

        Returns:
            ImageDataset instance
        """
        # Load images
        if images_path.endswith('.npz'):
            data = np.load(images_path)
            images = data['images'] if 'images' in data else data[data.files[0]]
            if labels_path is None and 'labels' in data:
                labels = data['labels']
            else:
                labels = None
        else:
            images = np.load(images_path)
            labels = None

        # Load labels if separate file
        if labels_path is not None:
            labels = np.load(labels_path)

        return cls(images=images, labels=labels, class_names=class_names)

    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        image_size: Tuple[int, int] = (28, 28),
        grayscale: bool = True,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp'),
    ) -> 'ImageDataset':
        """
        Load dataset from folder structure.

        Expected structure:
            folder_path/
                class_0/
                    image1.png
                    image2.png
                class_1/
                    image3.png
                ...

        Args:
            folder_path: Root folder containing class subfolders
            image_size: Target size (height, width) for images
            grayscale: Whether to convert to grayscale
            extensions: Tuple of valid file extensions

        Returns:
            ImageDataset instance
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for loading images from folders")

        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        # Get class folders
        class_folders = sorted([d for d in folder.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_folders]

        images_list = []
        labels_list = []

        for class_idx, class_folder in enumerate(class_folders):
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in extensions:
                    # Load image
                    if grayscale:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(str(img_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if img is not None:
                        # Resize
                        img = cv2.resize(img, (image_size[1], image_size[0]))
                        images_list.append(img)
                        labels_list.append(class_idx)

        images = np.array(images_list)
        labels = np.array(labels_list)

        return cls(images=images, labels=labels, class_names=class_names)

    @classmethod
    def from_pickle(cls, pickle_path: str) -> 'ImageDataset':
        """Load dataset from pickle file."""
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            images = data.get('images') or data.get('data') or data.get('x')
            labels = data.get('labels') or data.get('targets') or data.get('y')
            class_names = data.get('class_names')
        else:
            images, labels = data[:2]
            class_names = data[2] if len(data) > 2 else None

        return cls(images=images, labels=labels, class_names=class_names)

    @classmethod
    def generate_synthetic(
        cls,
        num_samples_per_class: int = 100,
        num_classes: int = 5,
        image_size: Tuple[int, int] = (28, 28),
        pattern_type: str = "shapes",
    ) -> 'ImageDataset':
        """
        Generate a synthetic dataset for testing.

        Args:
            num_samples_per_class: Number of samples per class
            num_classes: Number of classes
            image_size: Size of generated images
            pattern_type: Type of patterns ("shapes", "gradients", "noise")

        Returns:
            ImageDataset with synthetic data
        """
        h, w = image_size
        images = []
        labels = []
        class_names = []

        for class_idx in range(num_classes):
            class_names.append(f"class_{class_idx}")

            for _ in range(num_samples_per_class):
                img = np.zeros((h, w), dtype=np.float32)

                if pattern_type == "shapes":
                    # Different geometric patterns per class
                    if class_idx % 5 == 0:
                        # Horizontal lines
                        num_lines = np.random.randint(2, 5)
                        for _ in range(num_lines):
                            y = np.random.randint(0, h)
                            thickness = np.random.randint(1, 4)
                            y_start = max(0, y - thickness // 2)
                            y_end = min(h, y + thickness // 2 + 1)
                            img[y_start:y_end, :] = 1.0
                    elif class_idx % 5 == 1:
                        # Vertical lines
                        num_lines = np.random.randint(2, 5)
                        for _ in range(num_lines):
                            x = np.random.randint(0, w)
                            thickness = np.random.randint(1, 4)
                            x_start = max(0, x - thickness // 2)
                            x_end = min(w, x + thickness // 2 + 1)
                            img[:, x_start:x_end] = 1.0
                    elif class_idx % 5 == 2:
                        # Circles/blobs
                        num_blobs = np.random.randint(1, 4)
                        for _ in range(num_blobs):
                            cy, cx = np.random.randint(5, h-5), np.random.randint(5, w-5)
                            r = np.random.randint(3, min(8, min(cy, h-cy, cx, w-cx)))
                            y, x = np.ogrid[:h, :w]
                            mask = (x - cx)**2 + (y - cy)**2 <= r**2
                            img[mask] = 1.0
                    elif class_idx % 5 == 3:
                        # Diagonal patterns
                        for i in range(h):
                            for j in range(w):
                                if (i + j) % (class_idx + 3) < 2:
                                    img[i, j] = 1.0
                    else:
                        # Corners/edges
                        corner = np.random.randint(0, 4)
                        size = np.random.randint(5, 15)
                        if corner == 0:
                            img[:size, :size] = 1.0
                        elif corner == 1:
                            img[:size, -size:] = 1.0
                        elif corner == 2:
                            img[-size:, :size] = 1.0
                        else:
                            img[-size:, -size:] = 1.0

                elif pattern_type == "gradients":
                    # Different gradient directions per class
                    angle = (class_idx / num_classes) * 2 * np.pi
                    y, x = np.mgrid[:h, :w]
                    img = (np.cos(angle) * x / w + np.sin(angle) * y / h + 1) / 2

                else:  # noise
                    # Different noise patterns
                    np.random.seed(class_idx * 1000 + len(images))
                    base_pattern = np.random.randn(h // 4, w // 4)
                    from scipy.ndimage import zoom
                    try:
                        img = zoom(base_pattern, 4, order=1)[:h, :w]
                    except ImportError:
                        img = np.random.randn(h, w)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Add some noise
                noise = np.random.randn(h, w) * 0.1
                img = np.clip(img + noise, 0, 1)

                images.append(img)
                labels.append(class_idx)

        images = np.array(images)
        labels = np.array(labels)

        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

        return cls(images=images, labels=labels, class_names=class_names)

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple['ImageDataset', 'ImageDataset']:
        """
        Split dataset into training and test sets.

        Args:
            train_ratio: Ratio of data for training
            shuffle: Whether to shuffle before splitting

        Returns:
            (train_dataset, test_dataset)
        """
        n = len(self.images)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(n * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        train_dataset = ImageDataset(
            images=self.images[train_idx],
            labels=self.labels[train_idx] if self.labels is not None else None,
            class_names=self.class_names,
            normalize=False,  # Already normalized
            flatten=False,    # Already processed
        )
        train_dataset.original_shape = self.original_shape
        train_dataset.num_classes = self.num_classes

        test_dataset = ImageDataset(
            images=self.images[test_idx],
            labels=self.labels[test_idx] if self.labels is not None else None,
            class_names=self.class_names,
            normalize=False,
            flatten=False,
        )
        test_dataset.original_shape = self.original_shape
        test_dataset.num_classes = self.num_classes

        return train_dataset, test_dataset

    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get a random batch of samples."""
        if shuffle:
            indices = np.random.choice(len(self.images), size=batch_size, replace=False)
        else:
            indices = np.arange(min(batch_size, len(self.images)))

        batch_images = self.images[indices]
        batch_labels = self.labels[indices] if self.labels is not None else None

        return batch_images, batch_labels

    def __len__(self) -> int:
        return len(self.images) if self.images is not None else 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int]]:
        label = self.labels[idx] if self.labels is not None else None
        return self.images[idx], label


class ClassPrototypeLayer:
    """
    A layer that learns class prototypes through competitive learning.

    Each neuron in this layer becomes a prototype for a class.
    The layer uses winner-take-all competition and learns to
    associate prototypes with class labels.
    """

    def __init__(
        self,
        input_size: int,
        num_prototypes_per_class: int = 3,
        num_classes: int = 10,
        learning_rate: float = 0.01,
        neighborhood_size: float = 2.0,
        neighborhood_decay: float = 0.99,
    ):
        """
        Initialize prototype layer.

        Args:
            input_size: Size of input feature vectors
            num_prototypes_per_class: Number of prototypes per class
            num_classes: Number of classes
            learning_rate: Base learning rate
            neighborhood_size: Initial SOM neighborhood size
            neighborhood_decay: Rate of neighborhood decay
        """
        self.input_size = input_size
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_classes = num_classes
        self.total_prototypes = num_prototypes_per_class * num_classes
        self.learning_rate = learning_rate
        self.neighborhood_size = neighborhood_size
        self.neighborhood_decay = neighborhood_decay

        # Initialize prototypes randomly
        self.prototypes = np.random.randn(self.total_prototypes, input_size) * 0.1
        self._normalize_prototypes()

        # Prototype-to-class mapping (initially assigned evenly)
        self.prototype_labels = np.repeat(
            np.arange(num_classes), num_prototypes_per_class
        )

        # Confidence scores for each prototype (how well it represents its class)
        self.prototype_confidence = np.ones(self.total_prototypes) * 0.5

        # Activation counts for load balancing
        self.activation_counts = np.zeros(self.total_prototypes)

        # Learning statistics
        self.total_updates = 0
        self.class_accuracies = {i: [] for i in range(num_classes)}

    def _normalize_prototypes(self) -> None:
        """Normalize all prototypes to unit length."""
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.prototypes = self.prototypes / (norms + 1e-8)

    def find_winner(self, input_vector: np.ndarray) -> Tuple[int, float]:
        """
        Find the winning prototype for an input.

        Args:
            input_vector: Input feature vector

        Returns:
            (winner_index, distance)
        """
        # Normalize input
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)

        # Compute similarities (cosine similarity = dot product for normalized vectors)
        similarities = np.dot(self.prototypes, input_norm)

        # Find winner
        winner_idx = np.argmax(similarities)

        return winner_idx, similarities[winner_idx]

    def classify(self, input_vector: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
        """
        Classify an input vector.

        Args:
            input_vector: Input feature vector

        Returns:
            (predicted_class, confidence, class_probabilities)
        """
        # Normalize input
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)

        # Compute similarities to all prototypes
        similarities = np.dot(self.prototypes, input_norm)

        # Convert to probabilities using softmax
        exp_sim = np.exp(similarities * 5)  # Temperature scaling
        probabilities = exp_sim / (exp_sim.sum() + 1e-8)

        # Aggregate by class (weighted by prototype confidence)
        class_probs = {}
        for class_idx in range(self.num_classes):
            mask = self.prototype_labels == class_idx
            class_prob = np.sum(probabilities[mask] * self.prototype_confidence[mask])
            class_probs[class_idx] = float(class_prob)

        # Normalize class probabilities
        total_prob = sum(class_probs.values())
        if total_prob > 0:
            class_probs = {k: v / total_prob for k, v in class_probs.items()}

        # Find predicted class
        predicted_class = max(class_probs, key=class_probs.get)
        confidence = class_probs[predicted_class]

        return predicted_class, confidence, class_probs

    def learn(
        self,
        input_vector: np.ndarray,
        true_label: int,
        learning_rate: Optional[float] = None,
    ) -> bool:
        """
        Learn from a labeled example using competitive learning.

        Args:
            input_vector: Input feature vector
            true_label: True class label
            learning_rate: Optional learning rate override

        Returns:
            True if prediction was correct before learning
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        # Normalize input
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)

        # Find winning prototype
        winner_idx, similarity = self.find_winner(input_vector)
        predicted_label = self.prototype_labels[winner_idx]
        correct = predicted_label == true_label

        # Update activation count
        self.activation_counts[winner_idx] += 1

        # Always strengthen the correct class prototype (supervised signal)
        correct_class_mask = self.prototype_labels == true_label
        correct_prototypes = self.prototypes[correct_class_mask]
        correct_similarities = np.dot(correct_prototypes, input_norm)
        best_correct_local = np.argmax(correct_similarities)
        best_correct_global = np.where(correct_class_mask)[0][best_correct_local]

        # Move the best correct prototype toward the input
        update_strength = learning_rate * 2.0 if not correct else learning_rate
        self.prototypes[best_correct_global] += update_strength * (
            input_norm - self.prototypes[best_correct_global]
        )

        if correct:
            # Increase confidence
            self.prototype_confidence[winner_idx] = min(1.0,
                self.prototype_confidence[winner_idx] + 0.02)

            # Also update other prototypes of the same class (weaker)
            same_class_mask = self.prototype_labels == true_label
            neighborhood = self._get_neighborhood(winner_idx)
            for idx in np.where(same_class_mask)[0]:
                if idx != winner_idx and idx != best_correct_global:
                    neighbor_factor = neighborhood[idx] * 0.2
                    self.prototypes[idx] += learning_rate * neighbor_factor * (
                        input_norm - self.prototypes[idx]
                    )
        else:
            # Winner was wrong - decrease its confidence
            self.prototype_confidence[winner_idx] = max(0.1,
                self.prototype_confidence[winner_idx] - 0.03)

            # Increase confidence of correct prototype
            self.prototype_confidence[best_correct_global] = min(1.0,
                self.prototype_confidence[best_correct_global] + 0.01)

            # Push the wrong winner away from the input (contrastive learning)
            self.prototypes[winner_idx] -= learning_rate * 0.5 * (
                input_norm - self.prototypes[winner_idx]
            )

        # Normalize prototypes
        self._normalize_prototypes()

        # Decay neighborhood
        self.neighborhood_size *= self.neighborhood_decay
        self.total_updates += 1

        return correct

    def _get_neighborhood(self, center_idx: int) -> np.ndarray:
        """Get neighborhood function values for all prototypes."""
        # Simple exponential neighborhood based on index distance
        indices = np.arange(self.total_prototypes)
        distances = np.abs(indices - center_idx)
        return np.exp(-distances**2 / (2 * self.neighborhood_size**2))

    def get_class_prototypes(self, class_idx: int) -> np.ndarray:
        """Get all prototypes for a class."""
        mask = self.prototype_labels == class_idx
        return self.prototypes[mask]

    def reassign_prototypes(self) -> None:
        """
        Reassign prototypes based on activation patterns.

        Low-activation prototypes are moved to underrepresented classes.
        """
        # Calculate class representation
        class_activations = {}
        for class_idx in range(self.num_classes):
            mask = self.prototype_labels == class_idx
            class_activations[class_idx] = np.sum(self.activation_counts[mask])

        # Find overrepresented and underrepresented classes
        mean_activation = np.mean(list(class_activations.values()))

        for class_idx, activation in class_activations.items():
            if activation < mean_activation * 0.5:
                # This class needs more prototypes
                # Find the prototype with lowest activation from another class
                other_mask = self.prototype_labels != class_idx
                if np.any(other_mask):
                    other_activations = self.activation_counts.copy()
                    other_activations[~other_mask] = np.inf
                    victim_idx = np.argmin(other_activations)

                    if self.activation_counts[victim_idx] < mean_activation * 0.3:
                        # Reassign this prototype
                        self.prototype_labels[victim_idx] = class_idx
                        # Reset to average of class prototypes
                        class_protos = self.get_class_prototypes(class_idx)
                        if len(class_protos) > 1:
                            self.prototypes[victim_idx] = np.mean(class_protos, axis=0)
                        self.activation_counts[victim_idx] = 0

                        logger.debug(f"Reassigned prototype {victim_idx} to class {class_idx}")

        self._normalize_prototypes()

    def get_stats(self) -> Dict[str, Any]:
        """Get prototype layer statistics."""
        return {
            'total_prototypes': self.total_prototypes,
            'prototypes_per_class': self.num_prototypes_per_class,
            'num_classes': self.num_classes,
            'total_updates': self.total_updates,
            'mean_confidence': float(np.mean(self.prototype_confidence)),
            'activation_distribution': {
                int(i): int(c) for i, c in enumerate(self.activation_counts)
            },
            'neighborhood_size': float(self.neighborhood_size),
        }


class ImageClassificationLearner:
    """
    Main image classification learning system.

    This system autonomously learns to classify images using:
    1. Unsupervised feature learning (Hebbian/competitive learning)
    2. Prototype-based classification
    3. Meta-learning for strategy optimization
    4. Curriculum learning for progressive difficulty
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        num_classes: int = 10,
        feature_layers: Optional[List[int]] = None,
        num_prototypes_per_class: int = 3,
        learning_rate: float = 0.01,
        use_meta_learning: bool = True,
        use_feature_learning: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the image classification learner.

        Args:
            input_size: Size of flattened input images (auto-detected if None)
            num_classes: Number of classes to learn
            feature_layers: Sizes of feature extraction layers
            num_prototypes_per_class: Prototypes per class for classification
            learning_rate: Base learning rate
            use_meta_learning: Whether to use meta-learning for optimization
            use_feature_learning: Whether to use learned features (vs raw input)
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.use_meta_learning = use_meta_learning
        self.use_feature_learning = use_feature_learning

        # Feature extraction layers (hierarchical)
        if feature_layers is None:
            feature_layers = [256, 128, 64]
        self.feature_layer_sizes = feature_layers

        # These will be initialized when we know the input size
        self.feature_pathway: Optional[NeuralPathway] = None
        self.prototype_layer: Optional[ClassPrototypeLayer] = None
        self.num_prototypes_per_class = num_prototypes_per_class

        # Flag to initialize prototypes from data
        self.prototypes_initialized_from_data = False

        # Meta-learner for strategy optimization
        if use_meta_learning:
            self.meta_learner = MetaLearner(
                exploration_rate=0.2,
                learning_rate=0.01,
                enable_curriculum=True,
            )
        else:
            self.meta_learner = None

        # Current learning strategy
        self.current_strategy = LearningStrategy.OJA
        self.current_hyperparameters = {
            'learning_rate': learning_rate,
            'k_winners': 0.1,
            'sparsity': 0.1,
        }

        # Training state
        self.is_initialized = False
        self.epochs_trained = 0
        self.samples_seen = 0
        self.class_names: List[str] = []

        # Metrics history
        self.training_history: List[LearningMetrics] = []
        self.accuracy_history: List[float] = []

        logger.info("ImageClassificationLearner created")

    def _initialize_networks(self, input_size: int) -> None:
        """Initialize the neural networks with the correct input size."""
        self.input_size = input_size

        if self.use_feature_learning:
            # Create feature extraction pathway
            self.feature_pathway = NeuralPathway(
                name="classification_features",
                input_size=input_size,
                layer_sizes=self.feature_layer_sizes,
                learning_rates=[self.learning_rate] * len(self.feature_layer_sizes),
                use_recurrent=False,
            )
            # Prototype layer uses feature pathway output size
            feature_output_size = self.feature_layer_sizes[-1]
        else:
            # No feature learning - use raw input directly
            self.feature_pathway = None
            feature_output_size = input_size

        # Create prototype layer
        self.prototype_layer = ClassPrototypeLayer(
            input_size=feature_output_size,
            num_prototypes_per_class=self.num_prototypes_per_class,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
        )

        self.is_initialized = True
        logger.info(f"Networks initialized: input={input_size}, "
                   f"features={'learned' if self.use_feature_learning else 'raw'}, "
                   f"prototypes={self.num_prototypes_per_class}x{self.num_classes}")

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image using the feature pathway."""
        # Normalize input
        image_norm = image / (np.linalg.norm(image) + 1e-8)

        if self.feature_pathway is not None:
            # Process through feature pathway
            features = self.feature_pathway.process(image_norm)
        else:
            # Use normalized raw input as features
            features = image_norm

        return features

    def _initialize_prototypes_from_data(
        self,
        dataset: 'ImageDataset',
        samples_per_class: int = 5,
    ) -> None:
        """
        Initialize prototypes using actual samples from the dataset.

        This helps the classifier start with meaningful prototypes instead
        of random vectors.
        """
        if self.prototypes_initialized_from_data:
            return

        # Collect samples per class
        class_samples = {i: [] for i in range(self.num_classes)}

        for i in range(len(dataset)):
            image, label = dataset[i]
            if len(class_samples[label]) < samples_per_class:
                features = self._extract_features(image)
                class_samples[label].append(features)

            # Check if we have enough for all classes
            if all(len(s) >= samples_per_class for s in class_samples.values()):
                break

        # Initialize prototypes from collected samples
        for class_idx in range(self.num_classes):
            samples = class_samples.get(class_idx, [])
            if samples:
                # Get prototypes for this class
                class_mask = self.prototype_layer.prototype_labels == class_idx
                class_proto_indices = np.where(class_mask)[0]

                for i, proto_idx in enumerate(class_proto_indices):
                    if i < len(samples):
                        # Use actual sample as prototype
                        self.prototype_layer.prototypes[proto_idx] = samples[i].copy()
                    elif samples:
                        # Use average of samples
                        self.prototype_layer.prototypes[proto_idx] = np.mean(samples, axis=0)

        # Normalize all prototypes
        self.prototype_layer._normalize_prototypes()
        self.prototypes_initialized_from_data = True

        logger.info("Prototypes initialized from data samples")

    def _get_task_characteristics(self) -> Dict[str, float]:
        """Get current task characteristics for meta-learning."""
        return {
            'input_size': min(1.0, self.input_size / 1000) if self.input_size else 0.5,
            'num_classes': min(1.0, self.num_classes / 100),
            'samples_seen': min(1.0, self.samples_seen / 10000),
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.5,
            'epochs_trained': min(1.0, self.epochs_trained / 100),
        }

    def train_on_dataset(
        self,
        dataset: ImageDataset,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: bool = True,
        callback: Optional[Callable[[LearningMetrics], None]] = None,
    ) -> List[LearningMetrics]:
        """
        Train the classifier on a dataset.

        Args:
            dataset: ImageDataset to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Whether to print progress
            callback: Optional callback called after each epoch

        Returns:
            List of LearningMetrics for each epoch
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Store class names
        self.class_names = dataset.class_names or [str(i) for i in range(dataset.num_classes)]
        self.num_classes = dataset.num_classes

        # Initialize networks if needed
        if not self.is_initialized:
            self._initialize_networks(dataset.images.shape[1])
            # Reinitialize prototype layer with correct number of classes
            if self.use_feature_learning:
                feature_output_size = self.feature_layer_sizes[-1]
            else:
                feature_output_size = dataset.images.shape[1]
            self.prototype_layer = ClassPrototypeLayer(
                input_size=feature_output_size,
                num_prototypes_per_class=self.num_prototypes_per_class,
                num_classes=self.num_classes,
                learning_rate=self.learning_rate,
            )

        # Initialize prototypes from actual data samples (much better than random)
        if not self.prototypes_initialized_from_data:
            self._initialize_prototypes_from_data(dataset)

        # Split into train/validation
        if validation_split > 0:
            train_data, val_data = dataset.split(1.0 - validation_split)
        else:
            train_data = dataset
            val_data = None

        metrics_list = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # Select learning strategy using meta-learning
            if self.meta_learner is not None:
                task_chars = self._get_task_characteristics()
                self.current_strategy, self.current_hyperparameters = \
                    self.meta_learner.select_strategy(task_chars)

            # Map strategy to learning rule
            learning_rule = self._strategy_to_rule(self.current_strategy)
            effective_lr = self.current_hyperparameters.get('learning_rate', self.learning_rate)

            # Training loop
            correct = 0
            total = 0
            total_loss = 0.0

            # Shuffle training data
            indices = np.random.permutation(len(train_data))

            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i + batch_size]

                for idx in batch_indices:
                    image, label = train_data[idx]

                    # Extract features (with learning)
                    features = self._extract_features(image)

                    # Learn features (unsupervised) - only if using feature learning
                    if self.feature_pathway is not None:
                        self.feature_pathway.learn(learning_rule)

                    # Learn classification (supervised)
                    was_correct = self.prototype_layer.learn(
                        features, label, learning_rate=effective_lr
                    )

                    if was_correct:
                        correct += 1
                    total += 1

                    # Calculate loss (negative log likelihood)
                    _, confidence, probs = self.prototype_layer.classify(features)
                    if label in probs:
                        total_loss += -np.log(probs[label] + 1e-8)

                    self.samples_seen += 1

            # Calculate epoch metrics
            train_accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else 0.0

            # Validation
            if val_data is not None:
                val_accuracy = self.evaluate(val_data)
            else:
                val_accuracy = train_accuracy

            # Calculate feature sparsity
            if self.feature_pathway and len(self.feature_pathway.layers) > 0:
                last_layer = self.feature_pathway.layers[-1]
                sparsity = float(np.mean(last_layer.activations > 0))
            else:
                sparsity = 0.0

            # Create metrics
            epoch_duration = time.time() - epoch_start
            metrics = LearningMetrics(
                epoch=self.epochs_trained + epoch + 1,
                accuracy=float(val_accuracy),
                loss=float(avg_loss),
                feature_sparsity=sparsity,
                prototype_distances={},
                learning_rate=effective_lr,
                strategy=self.current_strategy.value,
                duration=epoch_duration,
            )

            metrics_list.append(metrics)
            self.training_history.append(metrics)
            self.accuracy_history.append(val_accuracy)

            # Update meta-learner
            if self.meta_learner is not None:
                self.meta_learner.update(
                    self.current_strategy,
                    self.current_hyperparameters,
                    self._get_task_characteristics(),
                    {
                        'accuracy': val_accuracy,
                        'sparsity': sparsity,
                        'reconstruction_error': avg_loss,
                    }
                )

            # Print progress
            if verbose:
                print(f"Epoch {self.epochs_trained + epoch + 1}/{self.epochs_trained + epochs}: "
                      f"acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}, "
                      f"loss={avg_loss:.4f}, strategy={self.current_strategy.value}, "
                      f"time={epoch_duration:.2f}s")

            # Callback
            if callback is not None:
                callback(metrics)

            # Periodically reassign prototypes
            if (epoch + 1) % 5 == 0:
                self.prototype_layer.reassign_prototypes()

        self.epochs_trained += epochs

        return metrics_list

    def _strategy_to_rule(self, strategy: LearningStrategy) -> str:
        """Convert meta-learning strategy to learning rule name."""
        mapping = {
            LearningStrategy.HEBBIAN: 'hebbian',
            LearningStrategy.OJA: 'oja',
            LearningStrategy.STDP: 'stdp',
            LearningStrategy.BCM: 'oja',  # Use Oja as fallback
            LearningStrategy.COMPETITIVE: 'oja',
            LearningStrategy.COOPERATIVE: 'hebbian',
            LearningStrategy.ANTI_HEBBIAN: 'oja',
        }
        return mapping.get(strategy, 'oja')

    def train_single(
        self,
        image: np.ndarray,
        label: int,
    ) -> bool:
        """
        Train on a single image-label pair.

        Args:
            image: Flattened image vector
            label: Class label

        Returns:
            True if prediction was correct before learning
        """
        # Initialize if needed
        if not self.is_initialized:
            self._initialize_networks(len(image))

        # Extract features
        features = self._extract_features(image)

        # Learn features
        learning_rule = self._strategy_to_rule(self.current_strategy)
        self.feature_pathway.learn(learning_rule)

        # Learn classification
        correct = self.prototype_layer.learn(features, label)

        self.samples_seen += 1

        return correct

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify an image.

        Args:
            image: Flattened image vector

        Returns:
            ClassificationResult with prediction details
        """
        if not self.is_initialized:
            raise RuntimeError("Classifier not initialized. Train on data first.")

        # Extract features
        features = self._extract_features(image)

        # Classify
        predicted_class, confidence, class_probs = self.prototype_layer.classify(features)

        # Get label name
        if predicted_class < len(self.class_names):
            predicted_label = self.class_names[predicted_class]
        else:
            predicted_label = str(predicted_class)

        return ClassificationResult(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_activations=features,
        )

    def evaluate(
        self,
        dataset: ImageDataset,
        verbose: bool = False,
    ) -> float:
        """
        Evaluate accuracy on a dataset.

        Args:
            dataset: Dataset to evaluate on
            verbose: Whether to print per-class accuracy

        Returns:
            Overall accuracy
        """
        if not self.is_initialized:
            raise RuntimeError("Classifier not initialized. Train on data first.")

        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(self.num_classes)}
        class_total = {i: 0 for i in range(self.num_classes)}

        for i in range(len(dataset)):
            image, label = dataset[i]
            result = self.classify(image)

            if result.predicted_class == label:
                correct += 1
                class_correct[label] += 1

            class_total[label] += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        if verbose:
            print(f"\nOverall accuracy: {accuracy:.4f}")
            print("\nPer-class accuracy:")
            for class_idx in range(self.num_classes):
                if class_total[class_idx] > 0:
                    class_acc = class_correct[class_idx] / class_total[class_idx]
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)
                    print(f"  {class_name}: {class_acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

        return accuracy

    def get_confusion_matrix(self, dataset: ImageDataset) -> np.ndarray:
        """
        Compute confusion matrix for a dataset.

        Args:
            dataset: Dataset to evaluate

        Returns:
            Confusion matrix (num_classes x num_classes)
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for i in range(len(dataset)):
            image, true_label = dataset[i]
            result = self.classify(image)
            matrix[true_label, result.predicted_class] += 1

        return matrix

    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        stats = {
            'is_initialized': self.is_initialized,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'feature_layers': self.feature_layer_sizes,
            'epochs_trained': self.epochs_trained,
            'samples_seen': self.samples_seen,
            'current_strategy': self.current_strategy.value,
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
        }

        if self.prototype_layer is not None:
            stats['prototype_stats'] = self.prototype_layer.get_stats()

        if self.meta_learner is not None:
            stats['meta_learning_stats'] = self.meta_learner.get_stats()

        return stats

    def save(self, filepath: str) -> None:
        """Save the classifier to a file."""
        state = {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'feature_layer_sizes': self.feature_layer_sizes,
            'num_prototypes_per_class': self.num_prototypes_per_class,
            'learning_rate': self.learning_rate,
            'class_names': self.class_names,
            'epochs_trained': self.epochs_trained,
            'samples_seen': self.samples_seen,
            'accuracy_history': self.accuracy_history,
            'is_initialized': self.is_initialized,
        }

        # Save prototype layer state
        if self.prototype_layer is not None:
            state['prototypes'] = self.prototype_layer.prototypes
            state['prototype_labels'] = self.prototype_layer.prototype_labels
            state['prototype_confidence'] = self.prototype_layer.prototype_confidence

        # Save feature pathway weights
        if self.feature_pathway is not None:
            state['feature_weights'] = []
            for layer in self.feature_pathway.layers:
                state['feature_weights'].append(layer.get_weight_matrix())

        # Save meta-learner state
        if self.meta_learner is not None:
            state['meta_learner'] = self.meta_learner.serialize()

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Classifier saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ImageClassificationLearner':
        """Load a classifier from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance
        learner = cls(
            input_size=state['input_size'],
            num_classes=state['num_classes'],
            feature_layers=state['feature_layer_sizes'],
            num_prototypes_per_class=state['num_prototypes_per_class'],
            learning_rate=state['learning_rate'],
            use_meta_learning='meta_learner' in state,
        )

        # Restore state
        learner.class_names = state['class_names']
        learner.epochs_trained = state['epochs_trained']
        learner.samples_seen = state['samples_seen']
        learner.accuracy_history = state['accuracy_history']

        # Initialize networks
        if state['is_initialized']:
            learner._initialize_networks(state['input_size'])

            # Restore prototype layer
            if 'prototypes' in state:
                learner.prototype_layer.prototypes = state['prototypes']
                learner.prototype_layer.prototype_labels = state['prototype_labels']
                learner.prototype_layer.prototype_confidence = state['prototype_confidence']

            # Restore feature weights
            if 'feature_weights' in state:
                for i, weights in enumerate(state['feature_weights']):
                    if i < len(learner.feature_pathway.layers):
                        for j, neuron in enumerate(learner.feature_pathway.layers[i].neurons):
                            if j < len(weights):
                                neuron.weights = weights[j]

            # Restore meta-learner
            if 'meta_learner' in state:
                learner.meta_learner = MetaLearner.deserialize(state['meta_learner'])

        logger.info(f"Classifier loaded from {filepath}")
        return learner

    def visualize_prototypes(
        self,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[int, List[np.ndarray]]:
        """
        Visualize the learned prototypes.

        This shows what patterns the classifier has learned for each class.

        Args:
            image_shape: Original image shape for reshaping

        Returns:
            Dictionary mapping class index to list of prototype visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("Classifier not initialized")

        visualizations = {}

        for class_idx in range(self.num_classes):
            prototypes = self.prototype_layer.get_class_prototypes(class_idx)
            class_vis = []

            for proto in prototypes:
                # The prototype is in feature space, so we need to project back
                # For simplicity, we'll just return the raw prototype
                # In a full implementation, you'd use gradient ascent or deconvolution
                class_vis.append(proto)

            visualizations[class_idx] = class_vis

        return visualizations


def quick_train_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    epochs: int = 10,
    verbose: bool = True,
) -> ImageClassificationLearner:
    """
    Convenience function to quickly train a classifier.

    Args:
        images: Array of images (N, H, W) or (N, H, W, C) or (N, D)
        labels: Array of integer labels
        class_names: Optional list of class names
        epochs: Number of training epochs
        verbose: Whether to print progress

    Returns:
        Trained ImageClassificationLearner
    """
    # Create dataset
    dataset = ImageDataset(
        images=images,
        labels=labels,
        class_names=class_names,
    )

    # Create and train classifier
    num_classes = len(np.unique(labels))
    classifier = ImageClassificationLearner(
        num_classes=num_classes,
        use_meta_learning=True,
    )

    classifier.train_on_dataset(dataset, epochs=epochs, verbose=verbose)

    return classifier
