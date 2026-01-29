"""
Comprehensive tests for the image classification learning system.

Tests cover:
- ImageClassificationLearner initialization and training
- ImageDataset creation, management, splitting, and batching
- ClassPrototypeLayer competitive learning and classification
- Classification with synthetic data
- LearningMetrics tracking
- ClassificationResult dataclass
- quick_train_classifier convenience function

All tests use small 8x8 images and fixed random seeds for determinism and speed.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.image_classification import (
    ImageClassificationLearner,
    ImageDataset,
    ClassificationResult,
    LearningMetrics,
    ClassPrototypeLayer,
    quick_train_classifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_images(num_per_class=10, num_classes=3, size=8, seed=42):
    """Create small deterministic synthetic images and labels for testing."""
    rng = np.random.RandomState(seed)
    images = []
    labels = []
    for c in range(num_classes):
        for _ in range(num_per_class):
            img = np.zeros((size, size), dtype=np.float32)
            if c == 0:
                # Horizontal stripe pattern
                img[size // 3 : 2 * size // 3, :] = 1.0
            elif c == 1:
                # Vertical stripe pattern
                img[:, size // 3 : 2 * size // 3] = 1.0
            else:
                # Diagonal checkerboard pattern
                for i in range(size):
                    for j in range(size):
                        if (i + j) % 2 == 0:
                            img[i, j] = 1.0
            # Add tiny deterministic noise so samples are not identical
            img += rng.randn(size, size).astype(np.float32) * 0.05
            img = np.clip(img, 0.0, 1.0)
            images.append(img)
            labels.append(c)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def _make_flat_images(num_per_class=10, num_classes=3, dim=64, seed=42):
    """Create pre-flattened deterministic feature vectors per class."""
    rng = np.random.RandomState(seed)
    images = []
    labels = []
    for c in range(num_classes):
        center = np.zeros(dim, dtype=np.float32)
        # Each class gets a distinct centroid region
        start = c * (dim // num_classes)
        end = start + (dim // num_classes)
        center[start:end] = 1.0
        for _ in range(num_per_class):
            vec = center + rng.randn(dim).astype(np.float32) * 0.1
            images.append(vec)
            labels.append(c)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


# ===========================================================================
# ImageDataset tests
# ===========================================================================

class TestImageDatasetCreation(unittest.TestCase):
    """Tests for creating ImageDataset instances."""

    def test_create_with_images_and_labels(self):
        """Dataset should store images, labels, and derive num_classes."""
        images, labels = _make_synthetic_images(num_per_class=5, num_classes=3)
        ds = ImageDataset(images=images, labels=labels)
        self.assertEqual(len(ds), 15)
        self.assertEqual(ds.num_classes, 3)

    def test_create_empty(self):
        """Dataset created without data should have length 0."""
        ds = ImageDataset()
        self.assertEqual(len(ds), 0)

    def test_images_flattened_by_default(self):
        """2D+ images should be flattened to (N, D) when flatten=True."""
        images, labels = _make_synthetic_images(num_per_class=4, num_classes=2, size=8)
        ds = ImageDataset(images=images, labels=labels, flatten=True)
        self.assertEqual(ds.images.shape, (8, 64))

    def test_images_not_flattened(self):
        """When flatten=False, images should keep their spatial shape."""
        images, labels = _make_synthetic_images(num_per_class=4, num_classes=2, size=8)
        ds = ImageDataset(images=images, labels=labels, flatten=False)
        # Already 2D per image (8,8) -> stored as (N, 8, 8) which is 2D per sample
        # The flatten check is len(shape) > 2, so (N, 8, 8) has len 3 => won't flatten
        # with flatten=False
        self.assertEqual(ds.images.shape, (8, 8, 8))

    def test_normalization(self):
        """Pixel values >1 should be normalized to [0,1] by default."""
        images = np.ones((4, 8, 8), dtype=np.float32) * 255.0
        labels = np.array([0, 0, 1, 1])
        ds = ImageDataset(images=images, labels=labels)
        self.assertLessEqual(ds.images.max(), 1.0)

    def test_no_normalization_if_already_01(self):
        """Images already in [0,1] should not be divided by 255 again."""
        images = np.ones((4, 8, 8), dtype=np.float32) * 0.5
        labels = np.array([0, 0, 1, 1])
        ds = ImageDataset(images=images, labels=labels, normalize=True)
        np.testing.assert_allclose(ds.images.max(), 0.5)

    def test_class_names_auto_generated(self):
        """If class_names is None, they should be auto-generated as strings."""
        images, labels = _make_synthetic_images(num_per_class=3, num_classes=2)
        ds = ImageDataset(images=images, labels=labels)
        self.assertEqual(ds.class_names, ["0", "1"])

    def test_class_names_custom(self):
        """Custom class_names should be preserved."""
        images, labels = _make_synthetic_images(num_per_class=3, num_classes=2)
        ds = ImageDataset(images=images, labels=labels,
                          class_names=["horizontal", "vertical"])
        self.assertEqual(ds.class_names, ["horizontal", "vertical"])

    def test_original_shape_stored(self):
        """original_shape should reflect the spatial dimensions before flatten."""
        images, labels = _make_synthetic_images(num_per_class=2, num_classes=2, size=8)
        ds = ImageDataset(images=images, labels=labels, flatten=True)
        self.assertEqual(ds.original_shape, (8, 8))

    def test_float32_dtype(self):
        """Images should be converted to float32."""
        images = np.zeros((4, 8, 8), dtype=np.uint8)
        labels = np.array([0, 0, 1, 1])
        ds = ImageDataset(images=images, labels=labels)
        self.assertEqual(ds.images.dtype, np.float32)


class TestImageDatasetOperations(unittest.TestCase):
    """Tests for ImageDataset indexing, batching, and splitting."""

    def setUp(self):
        np.random.seed(99)
        self.images, self.labels = _make_synthetic_images(
            num_per_class=10, num_classes=3, size=8
        )
        self.ds = ImageDataset(images=self.images, labels=self.labels)

    def test_len(self):
        """__len__ should return total number of samples."""
        self.assertEqual(len(self.ds), 30)

    def test_getitem(self):
        """__getitem__ should return (image, label) tuple."""
        img, lbl = self.ds[0]
        self.assertIsInstance(img, np.ndarray)
        self.assertIsNotNone(lbl)

    def test_get_batch_size(self):
        """get_batch should return arrays of the requested batch size."""
        batch_imgs, batch_lbls = self.ds.get_batch(5)
        self.assertEqual(batch_imgs.shape[0], 5)
        self.assertEqual(batch_lbls.shape[0], 5)

    def test_get_batch_no_shuffle(self):
        """get_batch with shuffle=False should return the first N samples."""
        batch_imgs, batch_lbls = self.ds.get_batch(3, shuffle=False)
        np.testing.assert_array_equal(batch_imgs, self.ds.images[:3])

    def test_split_ratio(self):
        """split should produce train and test sets with correct sizes."""
        np.random.seed(10)
        train, test = self.ds.split(train_ratio=0.8, shuffle=False)
        self.assertEqual(len(train), 24)
        self.assertEqual(len(test), 6)

    def test_split_preserves_num_classes(self):
        """Both splits should preserve num_classes."""
        train, test = self.ds.split(train_ratio=0.7, shuffle=False)
        self.assertEqual(train.num_classes, 3)
        self.assertEqual(test.num_classes, 3)

    def test_split_preserves_class_names(self):
        """Both splits should carry forward class_names."""
        train, test = self.ds.split(train_ratio=0.7, shuffle=False)
        self.assertEqual(train.class_names, self.ds.class_names)
        self.assertEqual(test.class_names, self.ds.class_names)


class TestImageDatasetSyntheticGeneration(unittest.TestCase):
    """Tests for the generate_synthetic class method."""

    def test_generate_shapes(self):
        """generate_synthetic with shapes pattern should return valid dataset."""
        np.random.seed(7)
        # Use 16x16 because the shapes pattern's circle generator needs h >= 11
        ds = ImageDataset.generate_synthetic(
            num_samples_per_class=5,
            num_classes=3,
            image_size=(16, 16),
            pattern_type="shapes",
        )
        self.assertEqual(len(ds), 15)
        self.assertEqual(ds.num_classes, 3)
        self.assertEqual(ds.images.shape[1], 256)  # flattened 16*16

    def test_generate_gradients(self):
        """generate_synthetic with gradients pattern should work."""
        np.random.seed(8)
        ds = ImageDataset.generate_synthetic(
            num_samples_per_class=4,
            num_classes=2,
            image_size=(8, 8),
            pattern_type="gradients",
        )
        self.assertEqual(len(ds), 8)
        self.assertEqual(ds.num_classes, 2)

    def test_generate_class_names(self):
        """Synthetic datasets should have auto-generated class names."""
        np.random.seed(9)
        # Use gradients pattern to avoid shape constraints at small sizes
        ds = ImageDataset.generate_synthetic(
            num_samples_per_class=3,
            num_classes=4,
            image_size=(8, 8),
            pattern_type="gradients",
        )
        self.assertEqual(len(ds.class_names), 4)
        self.assertEqual(ds.class_names[0], "class_0")
        self.assertEqual(ds.class_names[3], "class_3")

    def test_generate_values_in_range(self):
        """All pixel values should be in [0, 1] after generation."""
        np.random.seed(11)
        ds = ImageDataset.generate_synthetic(
            num_samples_per_class=5,
            num_classes=2,
            image_size=(8, 8),
            pattern_type="shapes",
        )
        self.assertGreaterEqual(ds.images.min(), 0.0)
        self.assertLessEqual(ds.images.max(), 1.0)


# ===========================================================================
# ClassPrototypeLayer tests
# ===========================================================================

class TestClassPrototypeLayerInit(unittest.TestCase):
    """Tests for ClassPrototypeLayer initialization."""

    def setUp(self):
        np.random.seed(42)
        self.layer = ClassPrototypeLayer(
            input_size=64,
            num_prototypes_per_class=2,
            num_classes=3,
            learning_rate=0.01,
        )

    def test_total_prototypes(self):
        """total_prototypes should be num_prototypes_per_class * num_classes."""
        self.assertEqual(self.layer.total_prototypes, 6)

    def test_prototype_shape(self):
        """Prototypes array should be (total_prototypes, input_size)."""
        self.assertEqual(self.layer.prototypes.shape, (6, 64))

    def test_prototypes_normalized(self):
        """All prototypes should be approximately unit-length after init."""
        norms = np.linalg.norm(self.layer.prototypes, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_prototype_labels_assignment(self):
        """prototype_labels should evenly assign classes."""
        expected = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(self.layer.prototype_labels, expected)

    def test_initial_confidence(self):
        """All prototype confidences should start at 0.5."""
        np.testing.assert_allclose(self.layer.prototype_confidence, 0.5)

    def test_initial_activation_counts(self):
        """All activation counts should start at zero."""
        np.testing.assert_array_equal(
            self.layer.activation_counts, np.zeros(6)
        )

    def test_initial_total_updates(self):
        """total_updates should start at 0."""
        self.assertEqual(self.layer.total_updates, 0)


class TestClassPrototypeLayerFindWinner(unittest.TestCase):
    """Tests for find_winner method."""

    def setUp(self):
        np.random.seed(42)
        self.layer = ClassPrototypeLayer(
            input_size=16,
            num_prototypes_per_class=2,
            num_classes=3,
            learning_rate=0.01,
        )

    def test_find_winner_returns_tuple(self):
        """find_winner should return (index, similarity)."""
        vec = np.random.RandomState(0).randn(16).astype(np.float32)
        result = self.layer.find_winner(vec)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_find_winner_index_in_range(self):
        """Winner index must be in [0, total_prototypes)."""
        vec = np.random.RandomState(1).randn(16).astype(np.float32)
        idx, _ = self.layer.find_winner(vec)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, self.layer.total_prototypes)

    def test_find_winner_similarity_bounded(self):
        """Cosine similarity should be in [-1, 1]."""
        vec = np.random.RandomState(2).randn(16).astype(np.float32)
        _, sim = self.layer.find_winner(vec)
        self.assertGreaterEqual(sim, -1.0 - 1e-6)
        self.assertLessEqual(sim, 1.0 + 1e-6)

    def test_find_winner_deterministic(self):
        """Same input should yield the same winner every time."""
        vec = np.random.RandomState(3).randn(16).astype(np.float32)
        idx1, sim1 = self.layer.find_winner(vec)
        idx2, sim2 = self.layer.find_winner(vec)
        self.assertEqual(idx1, idx2)
        self.assertAlmostEqual(sim1, sim2, places=10)


class TestClassPrototypeLayerClassify(unittest.TestCase):
    """Tests for the classify method."""

    def setUp(self):
        np.random.seed(42)
        self.layer = ClassPrototypeLayer(
            input_size=16,
            num_prototypes_per_class=2,
            num_classes=3,
            learning_rate=0.01,
        )

    def test_classify_returns_correct_types(self):
        """classify should return (int, float, dict)."""
        vec = np.random.RandomState(0).randn(16).astype(np.float32)
        pred, conf, probs = self.layer.classify(vec)
        self.assertIsInstance(pred, (int, np.integer))
        self.assertIsInstance(conf, float)
        self.assertIsInstance(probs, dict)

    def test_classify_pred_in_range(self):
        """Predicted class should be in [0, num_classes)."""
        vec = np.random.RandomState(1).randn(16).astype(np.float32)
        pred, _, _ = self.layer.classify(vec)
        self.assertGreaterEqual(pred, 0)
        self.assertLess(pred, self.layer.num_classes)

    def test_classify_probabilities_sum_to_one(self):
        """Class probabilities should sum to approximately 1."""
        vec = np.random.RandomState(2).randn(16).astype(np.float32)
        _, _, probs = self.layer.classify(vec)
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_classify_all_classes_present(self):
        """Probability dict should have an entry for every class."""
        vec = np.random.RandomState(3).randn(16).astype(np.float32)
        _, _, probs = self.layer.classify(vec)
        for c in range(self.layer.num_classes):
            self.assertIn(c, probs)

    def test_classify_confidence_equals_max_prob(self):
        """Confidence should equal the probability of the predicted class."""
        vec = np.random.RandomState(4).randn(16).astype(np.float32)
        pred, conf, probs = self.layer.classify(vec)
        self.assertAlmostEqual(conf, probs[pred], places=8)


class TestClassPrototypeLayerLearn(unittest.TestCase):
    """Tests for the learn method."""

    def setUp(self):
        np.random.seed(42)
        self.layer = ClassPrototypeLayer(
            input_size=16,
            num_prototypes_per_class=2,
            num_classes=3,
            learning_rate=0.05,
        )

    def test_learn_returns_bool(self):
        """learn should return a boolean (was the prediction correct?)."""
        vec = np.random.RandomState(0).randn(16).astype(np.float32)
        result = self.layer.learn(vec, true_label=0)
        self.assertIsInstance(result, (bool, np.bool_))

    def test_learn_increments_total_updates(self):
        """Each learn call should increment total_updates by 1."""
        vec = np.random.RandomState(1).randn(16).astype(np.float32)
        before = self.layer.total_updates
        self.layer.learn(vec, true_label=1)
        self.assertEqual(self.layer.total_updates, before + 1)

    def test_learn_increments_activation_count(self):
        """Winner's activation count should increase after learn."""
        total_before = self.layer.activation_counts.sum()
        vec = np.random.RandomState(2).randn(16).astype(np.float32)
        self.layer.learn(vec, true_label=0)
        total_after = self.layer.activation_counts.sum()
        self.assertGreater(total_after, total_before)

    def test_prototypes_remain_normalized_after_learn(self):
        """All prototypes should stay approximately unit-length after learning."""
        vec = np.random.RandomState(3).randn(16).astype(np.float32)
        self.layer.learn(vec, true_label=2)
        norms = np.linalg.norm(self.layer.prototypes, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_learn_with_custom_learning_rate(self):
        """learn should accept a custom learning_rate override."""
        vec = np.random.RandomState(4).randn(16).astype(np.float32)
        # Should not raise
        self.layer.learn(vec, true_label=1, learning_rate=0.001)
        self.assertEqual(self.layer.total_updates, 1)

    def test_neighborhood_decays(self):
        """Neighborhood size should decrease after learning."""
        initial = self.layer.neighborhood_size
        vec = np.random.RandomState(5).randn(16).astype(np.float32)
        self.layer.learn(vec, true_label=0)
        self.assertLess(self.layer.neighborhood_size, initial)

    def test_repeated_learning_changes_prototypes(self):
        """Prototypes should change after several rounds of learning."""
        initial_prototypes = self.layer.prototypes.copy()
        rng = np.random.RandomState(6)
        for _ in range(20):
            vec = rng.randn(16).astype(np.float32)
            label = rng.randint(0, 3)
            self.layer.learn(vec, true_label=label)
        self.assertFalse(np.allclose(self.layer.prototypes, initial_prototypes))


class TestClassPrototypeLayerGetStats(unittest.TestCase):
    """Tests for get_stats and get_class_prototypes methods."""

    def setUp(self):
        np.random.seed(42)
        self.layer = ClassPrototypeLayer(
            input_size=16,
            num_prototypes_per_class=2,
            num_classes=3,
        )

    def test_get_stats_keys(self):
        """get_stats should contain expected keys."""
        stats = self.layer.get_stats()
        for key in ('total_prototypes', 'prototypes_per_class', 'num_classes',
                     'total_updates', 'mean_confidence', 'activation_distribution',
                     'neighborhood_size'):
            self.assertIn(key, stats)

    def test_get_stats_values_initial(self):
        """Initial stats should reflect default state."""
        stats = self.layer.get_stats()
        self.assertEqual(stats['total_prototypes'], 6)
        self.assertEqual(stats['prototypes_per_class'], 2)
        self.assertEqual(stats['num_classes'], 3)
        self.assertEqual(stats['total_updates'], 0)
        self.assertAlmostEqual(stats['mean_confidence'], 0.5)

    def test_get_class_prototypes_shape(self):
        """get_class_prototypes should return correct number per class."""
        for c in range(3):
            protos = self.layer.get_class_prototypes(c)
            self.assertEqual(protos.shape, (2, 16))

    def test_reassign_prototypes_no_error(self):
        """reassign_prototypes should run without error."""
        # Do some learning first so activation counts are nonzero
        rng = np.random.RandomState(42)
        for _ in range(10):
            vec = rng.randn(16).astype(np.float32)
            self.layer.learn(vec, rng.randint(0, 3))
        self.layer.reassign_prototypes()
        # Prototypes should still be normalized
        norms = np.linalg.norm(self.layer.prototypes, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ===========================================================================
# ClassificationResult tests
# ===========================================================================

class TestClassificationResult(unittest.TestCase):
    """Tests for the ClassificationResult dataclass."""

    def test_create_instance(self):
        """ClassificationResult should store all provided fields."""
        result = ClassificationResult(
            predicted_class=2,
            predicted_label="cat",
            confidence=0.85,
            class_probabilities={0: 0.05, 1: 0.10, 2: 0.85},
            feature_activations=np.zeros(16),
        )
        self.assertEqual(result.predicted_class, 2)
        self.assertEqual(result.predicted_label, "cat")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertEqual(len(result.class_probabilities), 3)
        self.assertEqual(result.feature_activations.shape, (16,))

    def test_fields_accessible(self):
        """All dataclass fields should be directly accessible."""
        activations = np.ones(8, dtype=np.float32)
        result = ClassificationResult(
            predicted_class=0,
            predicted_label="dog",
            confidence=0.92,
            class_probabilities={0: 0.92, 1: 0.08},
            feature_activations=activations,
        )
        self.assertIs(result.feature_activations, activations)


# ===========================================================================
# LearningMetrics tests
# ===========================================================================

class TestLearningMetrics(unittest.TestCase):
    """Tests for the LearningMetrics dataclass."""

    def test_create_instance(self):
        """LearningMetrics should store all provided fields."""
        m = LearningMetrics(
            epoch=1,
            accuracy=0.75,
            loss=0.42,
            feature_sparsity=0.15,
            prototype_distances={0: 0.3, 1: 0.5},
            learning_rate=0.01,
            strategy="oja",
            duration=1.23,
        )
        self.assertEqual(m.epoch, 1)
        self.assertAlmostEqual(m.accuracy, 0.75)
        self.assertAlmostEqual(m.loss, 0.42)
        self.assertAlmostEqual(m.feature_sparsity, 0.15)
        self.assertEqual(m.strategy, "oja")
        self.assertAlmostEqual(m.duration, 1.23)
        self.assertEqual(len(m.prototype_distances), 2)

    def test_metrics_mutable(self):
        """LearningMetrics fields should be mutable (standard dataclass)."""
        m = LearningMetrics(
            epoch=1, accuracy=0.5, loss=1.0, feature_sparsity=0.0,
            prototype_distances={}, learning_rate=0.01,
            strategy="hebbian", duration=0.0,
        )
        m.accuracy = 0.9
        self.assertAlmostEqual(m.accuracy, 0.9)


# ===========================================================================
# ImageClassificationLearner tests
# ===========================================================================

class TestImageClassificationLearnerInit(unittest.TestCase):
    """Tests for ImageClassificationLearner initialization."""

    def test_default_init(self):
        """Learner should initialize with sensible defaults."""
        np.random.seed(42)
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        self.assertFalse(learner.is_initialized)
        self.assertEqual(learner.num_classes, 3)
        self.assertEqual(learner.epochs_trained, 0)
        self.assertEqual(learner.samples_seen, 0)

    def test_init_with_input_size(self):
        """Providing input_size should store it but not initialize networks yet."""
        learner = ImageClassificationLearner(
            input_size=64, num_classes=3, random_seed=42
        )
        self.assertEqual(learner.input_size, 64)
        self.assertFalse(learner.is_initialized)

    def test_init_without_meta_learning(self):
        """Setting use_meta_learning=False should leave meta_learner as None."""
        learner = ImageClassificationLearner(
            num_classes=3, use_meta_learning=False, random_seed=42
        )
        self.assertIsNone(learner.meta_learner)

    def test_init_with_meta_learning(self):
        """Setting use_meta_learning=True should create a meta_learner."""
        learner = ImageClassificationLearner(
            num_classes=3, use_meta_learning=True, random_seed=42
        )
        self.assertIsNotNone(learner.meta_learner)

    def test_custom_feature_layers(self):
        """Custom feature_layers should be stored."""
        learner = ImageClassificationLearner(
            num_classes=3, feature_layers=[32, 16], random_seed=42
        )
        self.assertEqual(learner.feature_layer_sizes, [32, 16])

    def test_default_feature_layers(self):
        """Default feature layers should be [256, 128, 64]."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        self.assertEqual(learner.feature_layer_sizes, [256, 128, 64])

    def test_training_history_empty(self):
        """Training history should start empty."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        self.assertEqual(len(learner.training_history), 0)
        self.assertEqual(len(learner.accuracy_history), 0)


class TestImageClassificationLearnerNetworkInit(unittest.TestCase):
    """Tests for internal network initialization."""

    def test_initialize_networks_sets_flag(self):
        """_initialize_networks should set is_initialized to True."""
        np.random.seed(42)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            random_seed=42,
        )
        learner._initialize_networks(64)
        self.assertTrue(learner.is_initialized)

    def test_initialize_creates_prototype_layer(self):
        """After init, prototype_layer should exist."""
        np.random.seed(42)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            random_seed=42,
        )
        learner._initialize_networks(64)
        self.assertIsNotNone(learner.prototype_layer)

    def test_initialize_with_feature_learning(self):
        """Feature pathway should be created when use_feature_learning=True."""
        np.random.seed(42)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_feature_learning=True,
            random_seed=42,
        )
        learner._initialize_networks(64)
        self.assertIsNotNone(learner.feature_pathway)

    def test_initialize_without_feature_learning(self):
        """Feature pathway should be None when use_feature_learning=False."""
        np.random.seed(42)
        learner = ImageClassificationLearner(
            num_classes=3,
            use_feature_learning=False,
            random_seed=42,
        )
        learner._initialize_networks(64)
        self.assertIsNone(learner.feature_pathway)
        self.assertIsNotNone(learner.prototype_layer)


class TestImageClassificationLearnerClassify(unittest.TestCase):
    """Tests for classify before and after training."""

    def test_classify_before_init_raises(self):
        """Calling classify before initialization should raise RuntimeError."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        with self.assertRaises(RuntimeError):
            learner.classify(np.zeros(64))

    def test_classify_after_init_returns_result(self):
        """Classify should return ClassificationResult after initialization."""
        np.random.seed(42)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_feature_learning=False,
            random_seed=42,
        )
        learner._initialize_networks(64)
        learner.class_names = ["a", "b", "c"]
        vec = np.random.RandomState(0).randn(64).astype(np.float32)
        result = learner.classify(vec)
        self.assertIsInstance(result, ClassificationResult)
        self.assertIn(result.predicted_class, [0, 1, 2])
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(len(result.class_probabilities), 3)


class TestImageClassificationLearnerTraining(unittest.TestCase):
    """Tests for train_on_dataset and evaluate."""

    def setUp(self):
        np.random.seed(42)
        self.images, self.labels = _make_flat_images(
            num_per_class=10, num_classes=3, dim=64, seed=42
        )
        # Pre-flatten: images are already (N, 64)
        self.dataset = ImageDataset(
            images=self.images,
            labels=self.labels,
            class_names=["classA", "classB", "classC"],
            normalize=False,
            flatten=False,
        )

    def test_train_returns_metrics_list(self):
        """train_on_dataset should return a list of LearningMetrics."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.1,
        )
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 2)
        for m in metrics:
            self.assertIsInstance(m, LearningMetrics)

    def test_train_increments_epochs(self):
        """epochs_trained should increase after training."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(
            self.dataset, epochs=3, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertEqual(learner.epochs_trained, 3)

    def test_train_increments_samples_seen(self):
        """samples_seen should increase after training."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(
            self.dataset, epochs=1, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertGreater(learner.samples_seen, 0)

    def test_train_populates_history(self):
        """Training should populate training_history and accuracy_history."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertEqual(len(learner.training_history), 2)
        self.assertEqual(len(learner.accuracy_history), 2)

    def test_train_accuracy_nonneg(self):
        """All reported accuracies should be non-negative."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        for m in metrics:
            self.assertGreaterEqual(m.accuracy, 0.0)
            self.assertLessEqual(m.accuracy, 1.0)

    def test_evaluate_after_training(self):
        """evaluate should return a float accuracy in [0, 1]."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        accuracy = learner.evaluate(self.dataset)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_evaluate_before_init_raises(self):
        """evaluate should raise RuntimeError if not initialized."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        with self.assertRaises(RuntimeError):
            learner.evaluate(self.dataset)

    def test_train_on_empty_dataset_raises(self):
        """Training on an empty dataset should raise ValueError."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        empty_ds = ImageDataset()
        with self.assertRaises(ValueError):
            learner.train_on_dataset(empty_ds, epochs=1, verbose=False)

    def test_train_with_callback(self):
        """Callback should be called once per epoch."""
        callback_epochs = []

        def my_callback(metrics):
            callback_epochs.append(metrics.epoch)

        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(
            self.dataset, epochs=3, batch_size=10, verbose=False,
            validation_split=0.0, callback=my_callback,
        )
        self.assertEqual(len(callback_epochs), 3)


class TestImageClassificationLearnerGetStats(unittest.TestCase):
    """Tests for get_stats method."""

    def test_stats_before_init(self):
        """get_stats should work before initialization."""
        learner = ImageClassificationLearner(num_classes=3, random_seed=42)
        stats = learner.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertFalse(stats['is_initialized'])
        self.assertEqual(stats['num_classes'], 3)
        self.assertEqual(stats['epochs_trained'], 0)

    def test_stats_after_training(self):
        """get_stats should reflect post-training state."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=5, num_classes=2, dim=32, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=2,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=1, batch_size=5, verbose=False,
                                 validation_split=0.0)
        stats = learner.get_stats()
        self.assertTrue(stats['is_initialized'])
        self.assertEqual(stats['epochs_trained'], 1)
        self.assertGreater(stats['samples_seen'], 0)
        self.assertIn('prototype_stats', stats)


class TestImageClassificationLearnerConfusionMatrix(unittest.TestCase):
    """Tests for get_confusion_matrix."""

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be (num_classes, num_classes)."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=5, num_classes=3, dim=32, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=1, batch_size=5, verbose=False,
                                 validation_split=0.0)
        cm = learner.get_confusion_matrix(ds)
        self.assertEqual(cm.shape, (3, 3))

    def test_confusion_matrix_sums_to_total(self):
        """Confusion matrix entries should sum to total samples."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=5, num_classes=3, dim=32, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=1, batch_size=5, verbose=False,
                                 validation_split=0.0)
        cm = learner.get_confusion_matrix(ds)
        self.assertEqual(cm.sum(), len(ds))


class TestImageClassificationLearnerVisualize(unittest.TestCase):
    """Tests for visualize_prototypes."""

    def test_visualize_returns_dict(self):
        """visualize_prototypes should return a dict mapping class to prototypes."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=5, num_classes=2, dim=32, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=2,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=1, batch_size=5, verbose=False,
                                 validation_split=0.0)
        vis = learner.visualize_prototypes()
        self.assertIsInstance(vis, dict)
        self.assertIn(0, vis)
        self.assertIn(1, vis)
        for v in vis.values():
            self.assertIsInstance(v, list)
            self.assertGreater(len(v), 0)


# ===========================================================================
# Classification with synthetic data (end-to-end)
# ===========================================================================

class TestClassificationWithSyntheticData(unittest.TestCase):
    """End-to-end classification tests with synthetic data."""

    def test_train_and_classify_synthetic(self):
        """Train on synthetic data and verify classify returns valid results."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=15, num_classes=3, dim=64, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[32, 16],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=3, batch_size=10, verbose=False,
                                 validation_split=0.0)

        # Classify a sample from each class
        for c in range(3):
            idx = c * 15  # first sample of each class
            result = learner.classify(images[idx])
            self.assertIsInstance(result, ClassificationResult)
            self.assertIn(result.predicted_class, [0, 1, 2])
            self.assertGreater(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_accuracy_above_random(self):
        """After training, accuracy should beat random guessing on well-separated data."""
        np.random.seed(42)
        images, labels = _make_flat_images(
            num_per_class=20, num_classes=3, dim=64, seed=42
        )
        ds = ImageDataset(images=images, labels=labels, normalize=False, flatten=False)
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[32, 16],
            num_prototypes_per_class=3,
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        learner.train_on_dataset(ds, epochs=5, batch_size=10, verbose=False,
                                 validation_split=0.0)
        accuracy = learner.evaluate(ds)
        # Random chance is ~0.33 for 3 classes; trained model should exceed that
        self.assertGreater(accuracy, 0.33)

    def test_generate_synthetic_and_train(self):
        """Using generate_synthetic followed by training should work end-to-end."""
        np.random.seed(42)
        # Use 16x16 because the shapes pattern's circle generator needs h >= 11
        ds = ImageDataset.generate_synthetic(
            num_samples_per_class=10,
            num_classes=3,
            image_size=(16, 16),
            pattern_type="shapes",
        )
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[32, 16],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            ds, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertEqual(len(metrics), 2)
        self.assertTrue(learner.is_initialized)


# ===========================================================================
# quick_train_classifier tests
# ===========================================================================

class TestQuickTrainClassifier(unittest.TestCase):
    """Tests for the quick_train_classifier convenience function."""

    def test_returns_trained_learner(self):
        """quick_train_classifier should return an initialized learner."""
        np.random.seed(42)
        images, labels = _make_synthetic_images(
            num_per_class=8, num_classes=3, size=8
        )
        learner = quick_train_classifier(
            images=images,
            labels=labels,
            class_names=["a", "b", "c"],
            epochs=2,
            verbose=False,
        )
        self.assertIsInstance(learner, ImageClassificationLearner)
        self.assertTrue(learner.is_initialized)
        self.assertEqual(learner.epochs_trained, 2)

    def test_quick_train_num_classes(self):
        """Learner should detect the correct number of classes."""
        np.random.seed(42)
        images, labels = _make_synthetic_images(
            num_per_class=5, num_classes=4, size=8
        )
        learner = quick_train_classifier(
            images=images, labels=labels, epochs=1, verbose=False,
        )
        self.assertEqual(learner.num_classes, 4)

    def test_quick_train_can_classify(self):
        """Learner from quick_train should be able to classify new inputs."""
        np.random.seed(42)
        images, labels = _make_synthetic_images(
            num_per_class=8, num_classes=2, size=8
        )
        learner = quick_train_classifier(
            images=images, labels=labels, epochs=2, verbose=False,
        )
        # Classify the first image
        flat_img = images[0].reshape(-1).astype(np.float32)
        # Normalize as the dataset would
        flat_img = flat_img / 255.0 if flat_img.max() > 1.0 else flat_img
        result = learner.classify(flat_img)
        self.assertIsInstance(result, ClassificationResult)

    def test_quick_train_with_class_names(self):
        """Class names should be preserved in the trained learner."""
        np.random.seed(42)
        images, labels = _make_synthetic_images(
            num_per_class=5, num_classes=2, size=8
        )
        learner = quick_train_classifier(
            images=images, labels=labels,
            class_names=["cat", "dog"],
            epochs=1, verbose=False,
        )
        self.assertEqual(learner.class_names, ["cat", "dog"])


# ===========================================================================
# LearningMetrics tracking integration
# ===========================================================================

class TestLearningMetricsTracking(unittest.TestCase):
    """Tests that LearningMetrics are properly tracked during training."""

    def setUp(self):
        np.random.seed(42)
        self.images, self.labels = _make_flat_images(
            num_per_class=10, num_classes=3, dim=64, seed=42
        )
        self.dataset = ImageDataset(
            images=self.images, labels=self.labels,
            normalize=False, flatten=False,
        )

    def test_metrics_epoch_numbers(self):
        """Each metric should have incrementing epoch numbers."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=3, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        for i, m in enumerate(metrics):
            self.assertEqual(m.epoch, i + 1)

    def test_metrics_have_valid_loss(self):
        """Loss values should be finite non-negative floats."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        for m in metrics:
            self.assertTrue(np.isfinite(m.loss))
            self.assertGreaterEqual(m.loss, 0.0)

    def test_metrics_have_valid_duration(self):
        """Duration should be a positive finite float."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=2, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        for m in metrics:
            self.assertGreater(m.duration, 0.0)
            self.assertTrue(np.isfinite(m.duration))

    def test_metrics_strategy_is_string(self):
        """Strategy field should be a non-empty string."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=1, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertIsInstance(metrics[0].strategy, str)
        self.assertGreater(len(metrics[0].strategy), 0)

    def test_metrics_learning_rate_positive(self):
        """Learning rate should be a positive float."""
        learner = ImageClassificationLearner(
            num_classes=3,
            feature_layers=[16, 8],
            use_meta_learning=False,
            use_feature_learning=False,
            random_seed=42,
        )
        metrics = learner.train_on_dataset(
            self.dataset, epochs=1, batch_size=10, verbose=False,
            validation_split=0.0,
        )
        self.assertGreater(metrics[0].learning_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
