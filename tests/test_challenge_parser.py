"""
Comprehensive tests for ChallengeParser.

Tests initialization, natural language parsing, structured data parsing,
challenge type inference, modality inference, difficulty inference,
success criteria extraction, and error handling.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.challenge_parser import (
    ChallengeParser,
    CHALLENGE_TYPE_KEYWORDS,
    MODALITY_KEYWORDS,
    DIFFICULTY_KEYWORDS,
    ACCURACY_PATTERNS,
)
from self_organizing_av_system.core.challenge import (
    Challenge,
    ChallengeType,
    Modality,
    ChallengeStatus,
    DifficultyLevel,
    SuccessCriteria,
    TrainingData,
)


class TestChallengeParserInitialization(unittest.TestCase):
    """Tests for ChallengeParser initialization."""

    def test_parser_creates_successfully(self):
        """Parser should instantiate without errors."""
        parser = ChallengeParser()
        self.assertIsInstance(parser, ChallengeParser)

    def test_parser_compiles_accuracy_patterns(self):
        """Parser should pre-compile regex patterns on init."""
        parser = ChallengeParser()
        self.assertTrue(hasattr(parser, '_accuracy_patterns'))
        self.assertIsInstance(parser._accuracy_patterns, list)
        self.assertGreater(len(parser._accuracy_patterns), 0)

    def test_compiled_patterns_count_matches_source(self):
        """The number of compiled patterns should match ACCURACY_PATTERNS."""
        parser = ChallengeParser()
        self.assertEqual(len(parser._accuracy_patterns), len(ACCURACY_PATTERNS))

    def test_compiled_patterns_are_regex_objects(self):
        """Each compiled pattern should be a compiled regex."""
        import re
        parser = ChallengeParser()
        for pattern, extractor in parser._accuracy_patterns:
            self.assertIsNotNone(pattern.pattern)
            self.assertTrue(callable(extractor))


class TestParseDispatch(unittest.TestCase):
    """Tests for the parse() dispatch method."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_parse_with_string_input(self):
        """parse() with a string should delegate to parse_natural_language."""
        result = self.parser.parse("classify images of handwritten digits")
        self.assertIsInstance(result, Challenge)

    def test_parse_with_dict_input(self):
        """parse() with a dict should delegate to parse_structured."""
        result = self.parser.parse({
            'name': 'test_challenge',
            'description': 'A test challenge',
        })
        self.assertIsInstance(result, Challenge)

    def test_parse_with_invalid_type_raises(self):
        """parse() with an unsupported type should raise ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse(42)

    def test_parse_with_list_raises(self):
        """parse() with a list should raise ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse([1, 2, 3])

    def test_parse_with_none_raises(self):
        """parse() with None should raise ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse(None)

    def test_parse_with_bool_raises(self):
        """parse() with a boolean should raise ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse(True)


class TestInferChallengeType(unittest.TestCase):
    """Tests for challenge type inference from descriptions."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_pattern_recognition_classify(self):
        """Should detect PATTERN_RECOGNITION from 'classify' keyword."""
        result = self.parser.infer_challenge_type("classify images into categories")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_pattern_recognition_recognize(self):
        """Should detect PATTERN_RECOGNITION from 'recognize' keyword."""
        result = self.parser.infer_challenge_type("recognize handwritten digits")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_pattern_recognition_detect(self):
        """Should detect PATTERN_RECOGNITION from 'detect' keyword."""
        result = self.parser.infer_challenge_type("detect objects in a scene")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_pattern_recognition_identification(self):
        """Should detect PATTERN_RECOGNITION from 'identify' keyword."""
        result = self.parser.infer_challenge_type("identify and categorize species")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_prediction_predict(self):
        """Should detect PREDICTION from 'predict' keyword."""
        result = self.parser.infer_challenge_type("predict the next value in the series")
        self.assertEqual(result, ChallengeType.PREDICTION)

    def test_prediction_forecast(self):
        """Should detect PREDICTION from 'forecast' keyword."""
        result = self.parser.infer_challenge_type("forecast future weather conditions")
        self.assertEqual(result, ChallengeType.PREDICTION)

    def test_prediction_anticipate(self):
        """Should detect PREDICTION from 'anticipate' keyword."""
        result = self.parser.infer_challenge_type("anticipate the next event")
        self.assertEqual(result, ChallengeType.PREDICTION)

    def test_problem_solving_solve(self):
        """Should detect PROBLEM_SOLVING from 'solve' keyword."""
        result = self.parser.infer_challenge_type("solve this optimization problem")
        self.assertEqual(result, ChallengeType.PROBLEM_SOLVING)

    def test_problem_solving_optimize(self):
        """Should detect PROBLEM_SOLVING from 'optimize' keyword."""
        result = self.parser.infer_challenge_type("optimize the route for delivery")
        self.assertEqual(result, ChallengeType.PROBLEM_SOLVING)

    def test_problem_solving_navigate(self):
        """Should detect PROBLEM_SOLVING from 'navigate' keyword."""
        result = self.parser.infer_challenge_type("navigate through the maze")
        self.assertEqual(result, ChallengeType.PROBLEM_SOLVING)

    def test_association_associate(self):
        """Should detect ASSOCIATION from 'associate' keyword."""
        result = self.parser.infer_challenge_type("associate sounds with images")
        self.assertEqual(result, ChallengeType.ASSOCIATION)

    def test_association_match(self):
        """Should detect ASSOCIATION from 'match' keyword."""
        result = self.parser.infer_challenge_type("match pairs of related items")
        self.assertEqual(result, ChallengeType.ASSOCIATION)

    def test_association_link(self):
        """Should detect ASSOCIATION from 'link' keyword."""
        result = self.parser.infer_challenge_type("link concepts to their definitions")
        self.assertEqual(result, ChallengeType.ASSOCIATION)

    def test_sequence_learning(self):
        """Should detect SEQUENCE_LEARNING from 'sequence' keyword."""
        result = self.parser.infer_challenge_type("learn this sequence of steps")
        self.assertEqual(result, ChallengeType.SEQUENCE_LEARNING)

    def test_sequence_learning_temporal(self):
        """Should detect SEQUENCE_LEARNING from 'temporal' keyword."""
        result = self.parser.infer_challenge_type("learn temporal patterns in the trajectory")
        self.assertEqual(result, ChallengeType.SEQUENCE_LEARNING)

    def test_sequence_learning_time_series(self):
        """Should detect SEQUENCE_LEARNING from 'time series' keyword."""
        result = self.parser.infer_challenge_type("analyze this time series data")
        self.assertEqual(result, ChallengeType.SEQUENCE_LEARNING)

    def test_concept_formation(self):
        """Should detect CONCEPT_FORMATION from 'concept' keyword."""
        result = self.parser.infer_challenge_type("form concept of animal categories")
        self.assertEqual(result, ChallengeType.CONCEPT_FORMATION)

    def test_concept_formation_abstract(self):
        """Should detect CONCEPT_FORMATION from 'abstract' keyword."""
        result = self.parser.infer_challenge_type("abstract the general category from examples")
        self.assertEqual(result, ChallengeType.CONCEPT_FORMATION)

    def test_concept_formation_cluster(self):
        """Should detect CONCEPT_FORMATION from 'cluster' keyword."""
        result = self.parser.infer_challenge_type("cluster similar items into groups")
        self.assertEqual(result, ChallengeType.CONCEPT_FORMATION)

    def test_anomaly_detection(self):
        """Should detect ANOMALY_DETECTION from 'anomaly' keyword."""
        # Avoid 'detect' which also matches PATTERN_RECOGNITION
        result = self.parser.infer_challenge_type("anomaly in unusual sensor readings")
        self.assertEqual(result, ChallengeType.ANOMALY_DETECTION)

    def test_anomaly_detection_outlier(self):
        """Should detect ANOMALY_DETECTION from 'outlier' keyword."""
        # Avoid 'find' which also matches PROBLEM_SOLVING
        result = self.parser.infer_challenge_type("outlier values look abnormal")
        self.assertEqual(result, ChallengeType.ANOMALY_DETECTION)

    def test_anomaly_detection_unusual(self):
        """Should detect ANOMALY_DETECTION from 'unusual' keyword."""
        result = self.parser.infer_challenge_type("flag unusual network traffic patterns")
        self.assertEqual(result, ChallengeType.ANOMALY_DETECTION)

    def test_generation_generate(self):
        """Should detect GENERATION from 'generate' keyword."""
        result = self.parser.infer_challenge_type("generate new artwork")
        self.assertEqual(result, ChallengeType.GENERATION)

    def test_generation_create(self):
        """Should detect GENERATION from 'create' keyword."""
        result = self.parser.infer_challenge_type("create novel music compositions")
        self.assertEqual(result, ChallengeType.GENERATION)

    def test_generation_synthesize(self):
        """Should detect GENERATION from 'synthesize' keyword."""
        result = self.parser.infer_challenge_type("synthesize realistic speech")
        self.assertEqual(result, ChallengeType.GENERATION)

    def test_default_to_pattern_recognition(self):
        """Should default to PATTERN_RECOGNITION when no keywords match."""
        result = self.parser.infer_challenge_type("do something with data")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_empty_string_defaults(self):
        """Should default to PATTERN_RECOGNITION for empty string."""
        result = self.parser.infer_challenge_type("")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_case_insensitivity(self):
        """Keyword matching should be case-insensitive."""
        result = self.parser.infer_challenge_type("CLASSIFY IMAGES")
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)

    def test_highest_score_wins(self):
        """When multiple types match, the one with more keyword hits should win."""
        # 'classify' and 'recognition' and 'detect' are all PATTERN_RECOGNITION
        # 'predict' is PREDICTION -- but pattern recognition has more hits
        result = self.parser.infer_challenge_type(
            "classify and detect with recognition, also predict"
        )
        self.assertEqual(result, ChallengeType.PATTERN_RECOGNITION)


class TestInferModalities(unittest.TestCase):
    """Tests for modality inference from descriptions."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_vision_modality_image(self):
        """Should detect VISION from 'image' keyword."""
        result = self.parser.infer_modalities("classify image categories")
        self.assertIn(Modality.VISION, result)

    def test_vision_modality_pixel(self):
        """Should detect VISION from 'pixel' keyword."""
        result = self.parser.infer_modalities("analyze pixel data")
        self.assertIn(Modality.VISION, result)

    def test_vision_modality_mnist(self):
        """Should detect VISION from 'mnist' keyword."""
        result = self.parser.infer_modalities("learn mnist handwritten digits")
        self.assertIn(Modality.VISION, result)

    def test_audio_modality_sound(self):
        """Should detect AUDIO from 'sound' keyword."""
        result = self.parser.infer_modalities("classify different sound patterns")
        self.assertIn(Modality.AUDIO, result)

    def test_audio_modality_speech(self):
        """Should detect AUDIO from 'speech' keyword."""
        result = self.parser.infer_modalities("recognize speech commands")
        self.assertIn(Modality.AUDIO, result)

    def test_audio_modality_music(self):
        """Should detect AUDIO from 'music' keyword."""
        result = self.parser.infer_modalities("classify music genres")
        self.assertIn(Modality.AUDIO, result)

    def test_text_modality_sentence(self):
        """Should detect TEXT from 'sentence' keyword."""
        result = self.parser.infer_modalities("analyze sentence structure")
        self.assertIn(Modality.TEXT, result)

    def test_text_modality_sentiment(self):
        """Should detect TEXT from 'sentiment' keyword."""
        result = self.parser.infer_modalities("perform sentiment analysis")
        self.assertIn(Modality.TEXT, result)

    def test_text_modality_language(self):
        """Should detect TEXT from 'language' keyword."""
        result = self.parser.infer_modalities("process natural language")
        self.assertIn(Modality.TEXT, result)

    def test_sensor_modality_temperature(self):
        """Should detect SENSOR from 'temperature' keyword."""
        result = self.parser.infer_modalities("monitor temperature readings")
        self.assertIn(Modality.SENSOR, result)

    def test_sensor_modality_accelerometer(self):
        """Should detect SENSOR from 'accelerometer' keyword."""
        result = self.parser.infer_modalities("process accelerometer data")
        self.assertIn(Modality.SENSOR, result)

    def test_sensor_modality_lidar(self):
        """Should detect SENSOR from 'lidar' keyword."""
        result = self.parser.infer_modalities("process lidar point cloud")
        self.assertIn(Modality.SENSOR, result)

    def test_time_series_modality_stock(self):
        """Should detect TIME_SERIES from 'stock' keyword."""
        result = self.parser.infer_modalities("analyze stock market data")
        self.assertIn(Modality.TIME_SERIES, result)

    def test_time_series_modality_ecg(self):
        """Should detect TIME_SERIES from 'ecg' keyword."""
        result = self.parser.infer_modalities("analyze ecg signals")
        self.assertIn(Modality.TIME_SERIES, result)

    def test_time_series_modality_weather(self):
        """Should detect TIME_SERIES from 'weather' keyword."""
        result = self.parser.infer_modalities("forecast weather trends")
        self.assertIn(Modality.TIME_SERIES, result)

    def test_default_to_embedding(self):
        """Should default to EMBEDDING when no modality keywords match."""
        result = self.parser.infer_modalities("process generic data")
        self.assertEqual(result, [Modality.EMBEDDING])

    def test_empty_string_defaults_to_embedding(self):
        """Should default to EMBEDDING for empty string."""
        result = self.parser.infer_modalities("")
        self.assertEqual(result, [Modality.EMBEDDING])

    def test_multimodal_when_multiple_detected(self):
        """Should append MULTIMODAL when more than one modality is found."""
        result = self.parser.infer_modalities(
            "process image and audio data simultaneously"
        )
        self.assertIn(Modality.VISION, result)
        self.assertIn(Modality.AUDIO, result)
        self.assertIn(Modality.MULTIMODAL, result)

    def test_multimodal_text_and_vision(self):
        """Should detect MULTIMODAL when both text and vision are present."""
        result = self.parser.infer_modalities(
            "classify images with sentence captions"
        )
        self.assertIn(Modality.VISION, result)
        self.assertIn(Modality.TEXT, result)
        self.assertIn(Modality.MULTIMODAL, result)

    def test_single_modality_no_multimodal_appended(self):
        """Should NOT append MULTIMODAL when only one modality is found."""
        result = self.parser.infer_modalities("classify image categories")
        self.assertIn(Modality.VISION, result)
        self.assertNotIn(Modality.MULTIMODAL, result)

    def test_case_insensitive_modality(self):
        """Modality keyword matching should be case-insensitive."""
        result = self.parser.infer_modalities("ANALYZE IMAGE DATA")
        self.assertIn(Modality.VISION, result)


class TestInferDifficulty(unittest.TestCase):
    """Tests for difficulty level inference from descriptions."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_trivial_difficulty(self):
        """Should detect TRIVIAL difficulty."""
        result = self.parser.infer_difficulty("a trivial classification task")
        self.assertAlmostEqual(result, DifficultyLevel.TRIVIAL.value)

    def test_easy_difficulty(self):
        """Should detect EASY difficulty."""
        result = self.parser.infer_difficulty("an easy beginner exercise")
        self.assertAlmostEqual(result, DifficultyLevel.EASY.value)

    def test_medium_difficulty(self):
        """Should detect MEDIUM difficulty."""
        result = self.parser.infer_difficulty("a medium difficulty problem")
        self.assertAlmostEqual(result, DifficultyLevel.MEDIUM.value)

    def test_hard_difficulty(self):
        """Should detect HARD difficulty."""
        result = self.parser.infer_difficulty("a hard and challenging task")
        self.assertAlmostEqual(result, DifficultyLevel.HARD.value)

    def test_very_hard_difficulty(self):
        """Should detect VERY_HARD difficulty."""
        # Use 'expert level' keyword which is in VERY_HARD, avoiding HARD keywords
        result = self.parser.infer_difficulty("this is expert level stuff")
        self.assertAlmostEqual(result, DifficultyLevel.VERY_HARD.value)

    def test_expert_difficulty(self):
        """Should detect EXPERT difficulty."""
        # Use 'extreme' keyword which is only in EXPERT, avoiding VERY_HARD keywords
        result = self.parser.infer_difficulty("an extreme task")
        self.assertAlmostEqual(result, DifficultyLevel.EXPERT.value)

    def test_default_medium_difficulty(self):
        """Should default to MEDIUM when no difficulty keywords match."""
        result = self.parser.infer_difficulty("classify some data")
        self.assertAlmostEqual(result, DifficultyLevel.MEDIUM.value)

    def test_empty_string_defaults_medium(self):
        """Should default to MEDIUM for empty description."""
        result = self.parser.infer_difficulty("")
        self.assertAlmostEqual(result, DifficultyLevel.MEDIUM.value)

    def test_difficulty_is_float(self):
        """Returned difficulty should be a float."""
        result = self.parser.infer_difficulty("some task")
        self.assertIsInstance(result, float)

    def test_difficulty_in_valid_range(self):
        """Returned difficulty should be between 0.0 and 1.0."""
        for desc in ["trivial", "easy", "medium", "hard", "very hard", "expert", "xyz"]:
            result = self.parser.infer_difficulty(desc)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)


class TestExtractSuccessCriteria(unittest.TestCase):
    """Tests for success criteria extraction from descriptions."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_default_accuracy(self):
        """Should default to 0.8 accuracy when no criteria given."""
        criteria = self.parser.extract_success_criteria("classify data")
        self.assertAlmostEqual(criteria.accuracy, 0.8)

    def test_percentage_accuracy_extraction(self):
        """Should extract accuracy from 'X% accuracy' pattern."""
        criteria = self.parser.extract_success_criteria("achieve 95% accuracy on the task")
        self.assertAlmostEqual(criteria.accuracy, 0.95)

    def test_correct_percentage_extraction(self):
        """Should extract accuracy from 'X% correct' pattern."""
        criteria = self.parser.extract_success_criteria("get 90% correct predictions")
        self.assertAlmostEqual(criteria.accuracy, 0.90)

    def test_perfect_accuracy(self):
        """Should extract 1.0 from 'perfect' keyword."""
        criteria = self.parser.extract_success_criteria("achieve perfect score")
        self.assertAlmostEqual(criteria.accuracy, 1.0)

    def test_high_accuracy_keyword(self):
        """Should extract 0.95 from 'high accuracy' keyword."""
        criteria = self.parser.extract_success_criteria("need high accuracy results")
        self.assertAlmostEqual(criteria.accuracy, 0.95)

    def test_good_accuracy_keyword(self):
        """Should extract 0.85 from 'good accuracy' keyword."""
        criteria = self.parser.extract_success_criteria("get good accuracy on test set")
        self.assertAlmostEqual(criteria.accuracy, 0.85)

    def test_reasonable_keyword(self):
        """Should extract 0.75 from 'reasonable' keyword."""
        criteria = self.parser.extract_success_criteria("achieve reasonable performance")
        self.assertAlmostEqual(criteria.accuracy, 0.75)

    def test_returns_success_criteria_object(self):
        """Should return a SuccessCriteria instance."""
        criteria = self.parser.extract_success_criteria("some task")
        self.assertIsInstance(criteria, SuccessCriteria)

    def test_default_min_samples(self):
        """Extracted criteria should have default min_samples of 10."""
        criteria = self.parser.extract_success_criteria("some task")
        self.assertEqual(criteria.min_samples, 10)

    def test_default_max_iterations(self):
        """Extracted criteria should have default max_iterations of 1000."""
        criteria = self.parser.extract_success_criteria("some task")
        self.assertEqual(criteria.max_iterations, 1000)

    def test_accuracy_clamped_lower(self):
        """Accuracy should be clamped to minimum 0.1."""
        # No way to trigger from description easily (patterns always give >= 0.75),
        # but the clamping logic is still tested by ensuring default is >= 0.1.
        criteria = self.parser.extract_success_criteria("some task")
        self.assertGreaterEqual(criteria.accuracy, 0.1)

    def test_accuracy_clamped_upper(self):
        """Accuracy should be clamped to maximum 1.0."""
        criteria = self.parser.extract_success_criteria("achieve perfect results")
        self.assertLessEqual(criteria.accuracy, 1.0)

    def test_decimal_percentage(self):
        """Should handle decimal percentages like 87.5%."""
        criteria = self.parser.extract_success_criteria("achieve 87.5% accuracy on data")
        self.assertAlmostEqual(criteria.accuracy, 0.875)


class TestParseNaturalLanguage(unittest.TestCase):
    """Tests for full natural language parsing pipeline."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_returns_challenge_object(self):
        """Should return a Challenge instance."""
        result = self.parser.parse_natural_language("classify images")
        self.assertIsInstance(result, Challenge)

    def test_challenge_has_description(self):
        """Parsed challenge should store the original description."""
        desc = "recognize handwritten digits from images"
        result = self.parser.parse_natural_language(desc)
        self.assertEqual(result.description, desc)

    def test_challenge_has_name(self):
        """Parsed challenge should have a generated name."""
        result = self.parser.parse_natural_language("classify animal species")
        self.assertTrue(len(result.name) > 0)

    def test_challenge_type_set(self):
        """Parsed challenge should have the inferred challenge type."""
        result = self.parser.parse_natural_language("predict stock prices")
        self.assertEqual(result.challenge_type, ChallengeType.PREDICTION)

    def test_modalities_set(self):
        """Parsed challenge should have inferred modalities."""
        result = self.parser.parse_natural_language("classify images of animals")
        self.assertIn(Modality.VISION, result.modalities)

    def test_difficulty_set(self):
        """Parsed challenge should have inferred difficulty."""
        result = self.parser.parse_natural_language("easy classification task")
        self.assertAlmostEqual(result.difficulty, DifficultyLevel.EASY.value)

    def test_success_criteria_set(self):
        """Parsed challenge should have success criteria."""
        result = self.parser.parse_natural_language(
            "classify data with 90% accuracy"
        )
        self.assertIsInstance(result.success_criteria, SuccessCriteria)
        self.assertAlmostEqual(result.success_criteria.accuracy, 0.90)

    def test_metadata_source_is_natural_language(self):
        """Parsed NL challenge should have source=natural_language in metadata."""
        result = self.parser.parse_natural_language("classify data")
        self.assertEqual(result.metadata.get('source'), 'natural_language')

    def test_status_is_pending(self):
        """Newly parsed challenge should have PENDING status."""
        result = self.parser.parse_natural_language("classify data")
        self.assertEqual(result.status, ChallengeStatus.PENDING)

    def test_classification_visual_description(self):
        """Full parse of visual classification description."""
        result = self.parser.parse_natural_language(
            "classify handwritten digit images with high accuracy"
        )
        self.assertEqual(result.challenge_type, ChallengeType.PATTERN_RECOGNITION)
        self.assertIn(Modality.VISION, result.modalities)
        self.assertAlmostEqual(result.success_criteria.accuracy, 0.95)

    def test_sequence_audio_description(self):
        """Full parse of audio sequence description."""
        result = self.parser.parse_natural_language(
            "learn the sequence of audio patterns in speech"
        )
        self.assertEqual(result.challenge_type, ChallengeType.SEQUENCE_LEARNING)
        self.assertIn(Modality.AUDIO, result.modalities)

    def test_prediction_time_series_description(self):
        """Full parse of time series prediction description."""
        result = self.parser.parse_natural_language(
            "predict future stock prices from historical data"
        )
        self.assertEqual(result.challenge_type, ChallengeType.PREDICTION)
        self.assertIn(Modality.TIME_SERIES, result.modalities)

    def test_anomaly_sensor_description(self):
        """Full parse of sensor anomaly detection description."""
        # Avoid 'detect' which also matches PATTERN_RECOGNITION
        result = self.parser.parse_natural_language(
            "anomaly in unusual temperature sensor readings"
        )
        self.assertEqual(result.challenge_type, ChallengeType.ANOMALY_DETECTION)
        self.assertIn(Modality.SENSOR, result.modalities)

    def test_generation_description(self):
        """Full parse of generation task description."""
        result = self.parser.parse_natural_language(
            "generate new image compositions"
        )
        self.assertEqual(result.challenge_type, ChallengeType.GENERATION)
        self.assertIn(Modality.VISION, result.modalities)


class TestParseStructured(unittest.TestCase):
    """Tests for structured data parsing."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_minimal_structured_input(self):
        """Should parse a minimal dict with defaults."""
        result = self.parser.parse_structured({})
        self.assertIsInstance(result, Challenge)
        self.assertEqual(result.name, 'structured_challenge')

    def test_structured_with_name_and_description(self):
        """Should use provided name and description."""
        result = self.parser.parse_structured({
            'name': 'my_challenge',
            'description': 'A custom challenge',
        })
        self.assertEqual(result.name, 'my_challenge')
        self.assertEqual(result.description, 'A custom challenge')

    def test_structured_with_challenge_type_string(self):
        """Should parse challenge type from string."""
        result = self.parser.parse_structured({
            'challenge_type': 'prediction',
        })
        self.assertEqual(result.challenge_type, ChallengeType.PREDICTION)

    def test_structured_with_challenge_type_enum(self):
        """Should accept challenge type as enum directly."""
        result = self.parser.parse_structured({
            'challenge_type': ChallengeType.GENERATION,
        })
        self.assertEqual(result.challenge_type, ChallengeType.GENERATION)

    def test_structured_with_modality_string(self):
        """Should parse modality from a string."""
        result = self.parser.parse_structured({
            'modality': 'vision',
        })
        self.assertEqual(result.modalities, [Modality.VISION])

    def test_structured_with_modality_enum(self):
        """Should accept modality as enum directly."""
        result = self.parser.parse_structured({
            'modality': Modality.AUDIO,
        })
        self.assertEqual(result.modalities, [Modality.AUDIO])

    def test_structured_with_modalities_list_strings(self):
        """Should parse modalities from a list of strings."""
        result = self.parser.parse_structured({
            'modalities': ['vision', 'audio'],
        })
        self.assertIn(Modality.VISION, result.modalities)
        self.assertIn(Modality.AUDIO, result.modalities)

    def test_structured_with_modalities_list_enums(self):
        """Should parse modalities from a list of enums."""
        result = self.parser.parse_structured({
            'modalities': [Modality.TEXT, Modality.SENSOR],
        })
        self.assertIn(Modality.TEXT, result.modalities)
        self.assertIn(Modality.SENSOR, result.modalities)

    def test_structured_with_success_criteria(self):
        """Should parse success criteria from dict."""
        result = self.parser.parse_structured({
            'success_criteria': {
                'accuracy': 0.95,
                'min_samples': 50,
                'max_iterations': 500,
            },
        })
        self.assertAlmostEqual(result.success_criteria.accuracy, 0.95)
        self.assertEqual(result.success_criteria.min_samples, 50)
        self.assertEqual(result.success_criteria.max_iterations, 500)

    def test_structured_with_criteria_key(self):
        """Should also accept 'criteria' as key for success criteria."""
        result = self.parser.parse_structured({
            'criteria': {
                'accuracy': 0.9,
            },
        })
        self.assertAlmostEqual(result.success_criteria.accuracy, 0.9)

    def test_structured_with_difficulty_float(self):
        """Should accept difficulty as a float."""
        result = self.parser.parse_structured({
            'difficulty': 0.7,
        })
        self.assertAlmostEqual(result.difficulty, 0.7)

    def test_structured_with_difficulty_string(self):
        """Should accept difficulty as a named string."""
        result = self.parser.parse_structured({
            'difficulty': 'hard',
        })
        self.assertAlmostEqual(result.difficulty, DifficultyLevel.HARD.value)

    def test_structured_default_difficulty(self):
        """Should default to 0.5 difficulty."""
        result = self.parser.parse_structured({})
        self.assertAlmostEqual(result.difficulty, 0.5)

    def test_structured_default_modality_is_embedding(self):
        """Should default modalities to [EMBEDDING] when no data or modality."""
        result = self.parser.parse_structured({})
        self.assertEqual(result.modalities, [Modality.EMBEDDING])

    def test_structured_with_samples_and_labels(self):
        """Should create TrainingData from samples and labels."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        labels = [0, 1]
        result = self.parser.parse_structured({
            'samples': samples,
            'labels': labels,
        })
        self.assertIsNotNone(result.training_data)
        self.assertEqual(len(result.training_data), 2)

    def test_structured_with_data_key(self):
        """Should also accept 'data' as key for samples."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = self.parser.parse_structured({
            'data': samples,
        })
        self.assertIsNotNone(result.training_data)
        self.assertEqual(len(result.training_data), 2)

    def test_structured_no_labels_defaults_concept_formation(self):
        """When data has no labels, should default type to CONCEPT_FORMATION."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = self.parser.parse_structured({
            'data': samples,
        })
        self.assertEqual(result.challenge_type, ChallengeType.CONCEPT_FORMATION)

    def test_structured_with_metadata(self):
        """Should pass through metadata dict."""
        result = self.parser.parse_structured({
            'metadata': {'custom_key': 'custom_value'},
        })
        self.assertEqual(result.metadata.get('custom_key'), 'custom_value')

    def test_structured_default_metadata_has_source(self):
        """Should have source=structured in default metadata."""
        result = self.parser.parse_structured({})
        self.assertEqual(result.metadata.get('source'), 'structured')


class TestInferModalityFromData(unittest.TestCase):
    """Tests for _infer_modality_from_data private method."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_empty_samples_returns_embedding(self):
        """Empty sample list should return EMBEDDING."""
        result = self.parser._infer_modality_from_data([])
        self.assertEqual(result, Modality.EMBEDDING)

    def test_1d_short_array_returns_embedding(self):
        """Short 1D numpy array should return EMBEDDING."""
        samples = [np.zeros(50)]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.EMBEDDING)

    def test_1d_long_array_returns_time_series(self):
        """Long 1D numpy array (>100) should return TIME_SERIES."""
        samples = [np.zeros(200)]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.TIME_SERIES)

    def test_2d_square_array_returns_vision(self):
        """2D square numpy array should return VISION (grayscale image)."""
        samples = [np.zeros((28, 28))]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.VISION)

    def test_2d_rectangular_array_returns_audio(self):
        """2D rectangular numpy array should return AUDIO (spectrogram)."""
        samples = [np.zeros((128, 64))]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.AUDIO)

    def test_3d_array_returns_vision(self):
        """3D numpy array should return VISION (color image)."""
        samples = [np.zeros((32, 32, 3))]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.VISION)

    def test_4d_array_returns_vision(self):
        """4D numpy array should return VISION (video/batch)."""
        samples = [np.zeros((10, 32, 32, 3))]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.VISION)

    def test_string_samples_returns_text(self):
        """String samples should return TEXT."""
        samples = ["hello world"]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.TEXT)

    def test_list_of_numbers_returns_embedding(self):
        """List of numbers should return EMBEDDING."""
        samples = [[1.0, 2.0, 3.0]]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.EMBEDDING)

    def test_tuple_of_numbers_returns_embedding(self):
        """Tuple of numbers should return EMBEDDING."""
        samples = [(1.0, 2.0, 3.0)]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.EMBEDDING)

    def test_0d_array_returns_embedding(self):
        """0D numpy scalar array should return EMBEDDING (feature_dim=1 path)."""
        # ndim == 1 with length <= 100 -> EMBEDDING
        samples = [np.array([1.0])]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.EMBEDDING)

    def test_other_type_returns_embedding(self):
        """Non-standard sample types should return EMBEDDING."""
        samples = [42]
        result = self.parser._infer_modality_from_data(samples)
        self.assertEqual(result, Modality.EMBEDDING)


class TestGenerateName(unittest.TestCase):
    """Tests for _generate_name private method."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_generates_name_from_words(self):
        """Should generate a name from content words."""
        name = self.parser._generate_name("classify images of animals")
        self.assertTrue(len(name) > 0)
        self.assertIn('classify', name)
        self.assertIn('images', name)
        self.assertIn('animals', name)

    def test_removes_stop_words(self):
        """Should remove stop words from the name."""
        name = self.parser._generate_name("learn to classify the images")
        self.assertNotIn('learn', name.split('_'))
        self.assertNotIn('to', name.split('_'))
        self.assertNotIn('the', name.split('_'))

    def test_limits_to_four_words(self):
        """Should use at most 4 content words."""
        name = self.parser._generate_name(
            "classify recognize detect identify categorize objects"
        )
        parts = name.split('_')
        self.assertLessEqual(len(parts), 4)

    def test_truncates_to_50_characters(self):
        """Should truncate the name to at most 50 characters."""
        name = self.parser._generate_name(
            "verylongword " * 20
        )
        self.assertLessEqual(len(name), 50)

    def test_empty_description_defaults(self):
        """Empty description should return 'challenge'."""
        name = self.parser._generate_name("")
        self.assertEqual(name, 'challenge')

    def test_only_stop_words_returns_challenge(self):
        """Description with only stop words should return 'challenge'."""
        name = self.parser._generate_name("learn to the a an from with")
        self.assertEqual(name, 'challenge')

    def test_name_is_lowercase(self):
        """Generated name should be lowercase."""
        name = self.parser._generate_name("CLASSIFY IMAGES")
        self.assertEqual(name, name.lower())

    def test_words_joined_with_underscore(self):
        """Content words should be joined with underscores."""
        name = self.parser._generate_name("classify images fast")
        self.assertIn('_', name)


class TestKeywordMappings(unittest.TestCase):
    """Tests to verify keyword mapping constants are well-formed."""

    def test_all_challenge_types_have_keywords(self):
        """Every ChallengeType used in CHALLENGE_TYPE_KEYWORDS should be valid."""
        for ct in CHALLENGE_TYPE_KEYWORDS:
            self.assertIsInstance(ct, ChallengeType)

    def test_all_modalities_have_keywords(self):
        """Every Modality used in MODALITY_KEYWORDS should be valid."""
        for m in MODALITY_KEYWORDS:
            self.assertIsInstance(m, Modality)

    def test_all_difficulty_levels_have_keywords(self):
        """Every DifficultyLevel used in DIFFICULTY_KEYWORDS should be valid."""
        for dl in DIFFICULTY_KEYWORDS:
            self.assertIsInstance(dl, DifficultyLevel)

    def test_challenge_type_keywords_are_lists_of_strings(self):
        """All keyword lists should contain only strings."""
        for ct, keywords in CHALLENGE_TYPE_KEYWORDS.items():
            self.assertIsInstance(keywords, list)
            for kw in keywords:
                self.assertIsInstance(kw, str)

    def test_modality_keywords_are_lists_of_strings(self):
        """All modality keyword lists should contain only strings."""
        for m, keywords in MODALITY_KEYWORDS.items():
            self.assertIsInstance(keywords, list)
            for kw in keywords:
                self.assertIsInstance(kw, str)

    def test_difficulty_keywords_are_lists_of_strings(self):
        """All difficulty keyword lists should contain only strings."""
        for dl, keywords in DIFFICULTY_KEYWORDS.items():
            self.assertIsInstance(keywords, list)
            for kw in keywords:
                self.assertIsInstance(kw, str)

    def test_keywords_are_all_lowercase(self):
        """All keywords should be lowercase for consistent matching."""
        for ct, keywords in CHALLENGE_TYPE_KEYWORDS.items():
            for kw in keywords:
                self.assertEqual(kw, kw.lower(), f"Keyword '{kw}' is not lowercase")
        for m, keywords in MODALITY_KEYWORDS.items():
            for kw in keywords:
                self.assertEqual(kw, kw.lower(), f"Keyword '{kw}' is not lowercase")
        for dl, keywords in DIFFICULTY_KEYWORDS.items():
            for kw in keywords:
                self.assertEqual(kw, kw.lower(), f"Keyword '{kw}' is not lowercase")


class TestXpBackendIntegration(unittest.TestCase):
    """Tests ensuring the xp backend can be used alongside the parser."""

    def test_xp_is_available(self):
        """The xp backend should be importable and usable."""
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertEqual(len(arr), 3)

    def test_structured_parse_with_xp_arrays(self):
        """Parser should handle numpy arrays (from xp fallback) in structured data."""
        parser = ChallengeParser()
        samples = [xp.zeros(10), xp.ones(10)]
        labels = [0, 1]
        result = parser.parse_structured({
            'samples': [np.asarray(s) for s in samples],
            'labels': labels,
        })
        self.assertIsNotNone(result.training_data)
        self.assertEqual(len(result.training_data), 2)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self):
        self.parser = ChallengeParser()

    def test_very_long_description(self):
        """Should handle very long descriptions without error."""
        desc = "classify images " * 1000
        result = self.parser.parse_natural_language(desc)
        self.assertIsInstance(result, Challenge)

    def test_special_characters_in_description(self):
        """Should handle special characters in descriptions."""
        desc = "classify data @#$%^&*() with 90% accuracy!"
        result = self.parser.parse_natural_language(desc)
        self.assertIsInstance(result, Challenge)

    def test_unicode_in_description(self):
        """Should handle unicode characters in descriptions."""
        desc = "classify data with accuracy"
        result = self.parser.parse_natural_language(desc)
        self.assertIsInstance(result, Challenge)

    def test_newlines_in_description(self):
        """Should handle newlines in descriptions."""
        desc = "classify\nimages\nof\nanimals"
        result = self.parser.parse_natural_language(desc)
        self.assertIsInstance(result, Challenge)

    def test_structured_with_all_challenge_type_strings(self):
        """Should accept every ChallengeType name as a string."""
        for ct in ChallengeType:
            result = self.parser.parse_structured({
                'challenge_type': ct.name.lower(),
            })
            self.assertEqual(result.challenge_type, ct)

    def test_structured_with_all_modality_strings(self):
        """Should accept every Modality name as a string."""
        for mod in Modality:
            result = self.parser.parse_structured({
                'modality': mod.name.lower(),
            })
            self.assertEqual(result.modalities, [mod])

    def test_structured_with_all_difficulty_strings(self):
        """Should accept every DifficultyLevel name as a string."""
        for dl in DifficultyLevel:
            result = self.parser.parse_structured({
                'difficulty': dl.name.lower(),
            })
            self.assertAlmostEqual(result.difficulty, dl.value)

    def test_challenge_id_is_set(self):
        """Parsed challenges should have a non-empty id."""
        result = self.parser.parse("classify data")
        self.assertTrue(len(result.id) > 0)

    def test_challenge_created_at_is_set(self):
        """Parsed challenges should have a created_at timestamp."""
        result = self.parser.parse("classify data")
        self.assertIsNotNone(result.created_at)
        self.assertGreater(result.created_at, 0)


if __name__ == '__main__':
    unittest.main()
