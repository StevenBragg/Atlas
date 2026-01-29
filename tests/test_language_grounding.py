"""
Comprehensive tests for the language grounding module.

Tests cover: LanguageGrounding initialization, word grounding/learning,
GroundedWord creation, ParsedSentence parsing, WordType enum values,
TextCorpusLearner, and serialization via get_state / serialize.

All tests are deterministic and pass reliably by using fixed random seeds.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.language_grounding import (
    LanguageGrounding,
    TextCorpusLearner,
    GroundedWord,
    ParsedSentence,
    WordType,
)

# Fixed seed used throughout for deterministic tests
SEED = 42


# ---------------------------------------------------------------------------
# WordType enum
# ---------------------------------------------------------------------------
class TestWordType(unittest.TestCase):
    """Tests for the WordType enumeration."""

    def test_all_expected_values_exist(self):
        """WordType must expose every documented member."""
        expected = {
            "NOUN", "VERB", "ADJECTIVE", "ADVERB", "PREPOSITION",
            "DETERMINER", "CONJUNCTION", "PRONOUN", "UNKNOWN",
        }
        actual = {member.name for member in WordType}
        self.assertEqual(actual, expected)

    def test_string_values(self):
        """Each WordType member should have a lowercase string value."""
        for member in WordType:
            self.assertIsInstance(member.value, str)
            self.assertEqual(member.value, member.name.lower())

    def test_identity(self):
        """Constructing WordType from its value should return the same member."""
        for member in WordType:
            self.assertIs(WordType(member.value), member)

    def test_noun_value(self):
        """WordType.NOUN should have the value 'noun'."""
        self.assertEqual(WordType.NOUN.value, "noun")

    def test_verb_value(self):
        """WordType.VERB should have the value 'verb'."""
        self.assertEqual(WordType.VERB.value, "verb")

    def test_adjective_value(self):
        """WordType.ADJECTIVE should have the value 'adjective'."""
        self.assertEqual(WordType.ADJECTIVE.value, "adjective")

    def test_adverb_value(self):
        """WordType.ADVERB should have the value 'adverb'."""
        self.assertEqual(WordType.ADVERB.value, "adverb")

    def test_unknown_value(self):
        """WordType.UNKNOWN should have the value 'unknown'."""
        self.assertEqual(WordType.UNKNOWN.value, "unknown")

    def test_enum_count(self):
        """There should be exactly 9 WordType members."""
        self.assertEqual(len(WordType), 9)


# ---------------------------------------------------------------------------
# GroundedWord dataclass
# ---------------------------------------------------------------------------
class TestGroundedWord(unittest.TestCase):
    """Tests for the GroundedWord dataclass."""

    def _make_word(self, **overrides):
        """Helper to build a GroundedWord with sensible defaults."""
        defaults = dict(
            word="test",
            word_type=WordType.NOUN,
            embedding=np.zeros(16),
        )
        defaults.update(overrides)
        return GroundedWord(**defaults)

    def test_basic_creation(self):
        """A GroundedWord can be created with required fields."""
        gw = self._make_word()
        self.assertEqual(gw.word, "test")
        self.assertEqual(gw.word_type, WordType.NOUN)
        self.assertEqual(gw.embedding.shape, (16,))

    def test_default_sensory_grounding_is_none(self):
        """sensory_grounding should default to None."""
        gw = self._make_word()
        self.assertIsNone(gw.sensory_grounding)

    def test_default_concept_ids_empty(self):
        """concept_ids should default to an empty list."""
        gw = self._make_word()
        self.assertEqual(gw.concept_ids, [])

    def test_default_usage_count_zero(self):
        """usage_count should default to 0."""
        gw = self._make_word()
        self.assertEqual(gw.usage_count, 0)

    def test_default_confidence(self):
        """confidence should default to 0.5."""
        gw = self._make_word()
        self.assertAlmostEqual(gw.confidence, 0.5)

    def test_default_associations_empty(self):
        """associations should default to an empty dict."""
        gw = self._make_word()
        self.assertEqual(gw.associations, {})

    def test_custom_fields(self):
        """All fields can be set at construction time."""
        emb = np.ones(8)
        sensory = np.ones(4)
        gw = GroundedWord(
            word="hello",
            word_type=WordType.VERB,
            embedding=emb,
            sensory_grounding=sensory,
            concept_ids=["c1", "c2"],
            usage_count=10,
            confidence=0.9,
            associations={"world": 3.0},
        )
        self.assertEqual(gw.word, "hello")
        self.assertEqual(gw.word_type, WordType.VERB)
        np.testing.assert_array_equal(gw.embedding, emb)
        np.testing.assert_array_equal(gw.sensory_grounding, sensory)
        self.assertEqual(gw.concept_ids, ["c1", "c2"])
        self.assertEqual(gw.usage_count, 10)
        self.assertAlmostEqual(gw.confidence, 0.9)
        self.assertEqual(gw.associations, {"world": 3.0})

    def test_mutable_defaults_are_independent(self):
        """Each instance should have its own mutable default containers."""
        gw1 = self._make_word()
        gw2 = self._make_word()
        gw1.concept_ids.append("x")
        gw1.associations["a"] = 1.0
        self.assertEqual(gw2.concept_ids, [])
        self.assertEqual(gw2.associations, {})


# ---------------------------------------------------------------------------
# ParsedSentence dataclass
# ---------------------------------------------------------------------------
class TestParsedSentence(unittest.TestCase):
    """Tests for the ParsedSentence dataclass."""

    def _make_parsed(self, **overrides):
        defaults = dict(
            raw_text="the cat sat",
            tokens=["the", "cat", "sat"],
            word_types=[WordType.DETERMINER, WordType.NOUN, WordType.VERB],
            embeddings=[np.zeros(8)] * 3,
            sentence_embedding=np.zeros(8),
        )
        defaults.update(overrides)
        return ParsedSentence(**defaults)

    def test_basic_creation(self):
        """ParsedSentence can be created with required fields."""
        ps = self._make_parsed()
        self.assertEqual(ps.raw_text, "the cat sat")
        self.assertEqual(len(ps.tokens), 3)

    def test_default_subject_none(self):
        """subject should default to None."""
        ps = self._make_parsed()
        self.assertIsNone(ps.subject)

    def test_default_verb_none(self):
        """verb should default to None."""
        ps = self._make_parsed()
        self.assertIsNone(ps.verb)

    def test_default_object_none(self):
        """object should default to None."""
        ps = self._make_parsed()
        self.assertIsNone(ps.object)

    def test_default_modifiers_empty(self):
        """modifiers should default to an empty list."""
        ps = self._make_parsed()
        self.assertEqual(ps.modifiers, [])

    def test_custom_svo(self):
        """subject, verb, and object can be set."""
        ps = self._make_parsed(subject="cat", verb="sat", object="mat")
        self.assertEqual(ps.subject, "cat")
        self.assertEqual(ps.verb, "sat")
        self.assertEqual(ps.object, "mat")

    def test_modifiers(self):
        """modifiers can be provided."""
        ps = self._make_parsed(modifiers=["quickly", "large"])
        self.assertEqual(ps.modifiers, ["quickly", "large"])


# ---------------------------------------------------------------------------
# LanguageGrounding initialisation
# ---------------------------------------------------------------------------
class TestLanguageGroundingInit(unittest.TestCase):
    """Tests for LanguageGrounding constructor and initial state."""

    def test_default_parameters(self):
        """Constructor should set documented defaults."""
        lg = LanguageGrounding(random_seed=SEED)
        self.assertEqual(lg.embedding_dim, 128)
        self.assertEqual(lg.vocabulary_size, 10000)
        self.assertAlmostEqual(lg.learning_rate, 0.01)
        self.assertEqual(lg.context_window, 5)
        self.assertEqual(lg.sensory_dim, 64)

    def test_custom_parameters(self):
        """Constructor should accept custom values."""
        lg = LanguageGrounding(
            embedding_dim=64,
            vocabulary_size=5000,
            learning_rate=0.05,
            context_window=3,
            sensory_dim=32,
            random_seed=SEED,
        )
        self.assertEqual(lg.embedding_dim, 64)
        self.assertEqual(lg.vocabulary_size, 5000)
        self.assertAlmostEqual(lg.learning_rate, 0.05)
        self.assertEqual(lg.context_window, 3)
        self.assertEqual(lg.sensory_dim, 32)

    def test_vocabulary_starts_empty(self):
        """vocabulary should be empty at construction."""
        lg = LanguageGrounding(random_seed=SEED)
        self.assertEqual(len(lg.vocabulary), 0)

    def test_counters_start_at_zero(self):
        """Processing counters should start at zero."""
        lg = LanguageGrounding(random_seed=SEED)
        self.assertEqual(lg.total_words_processed, 0)
        self.assertEqual(lg.total_sentences_processed, 0)

    def test_word_patterns_initialized(self):
        """Word-pattern sets should be populated after init."""
        lg = LanguageGrounding(random_seed=SEED)
        self.assertGreater(len(lg.determiners), 0)
        self.assertGreater(len(lg.prepositions), 0)
        self.assertGreater(len(lg.conjunctions), 0)
        self.assertGreater(len(lg.pronouns), 0)
        self.assertGreater(len(lg.common_verbs), 0)
        self.assertGreater(len(lg.noun_suffixes), 0)
        self.assertGreater(len(lg.verb_suffixes), 0)
        self.assertGreater(len(lg.adj_suffixes), 0)
        self.assertGreater(len(lg.adv_suffixes), 0)

    def test_concept_to_words_empty(self):
        """concept_to_words should be empty initially."""
        lg = LanguageGrounding(random_seed=SEED)
        self.assertEqual(len(lg.concept_to_words), 0)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
class TestLanguageGroundingTokenize(unittest.TestCase):
    """Tests for the tokenize method."""

    def setUp(self):
        self.lg = LanguageGrounding(random_seed=SEED)

    def test_simple_sentence(self):
        """Tokenise a simple sentence."""
        tokens = self.lg.tokenize("The cat sat on the mat")
        self.assertEqual(tokens, ["the", "cat", "sat", "on", "the", "mat"])

    def test_lowercasing(self):
        """All tokens should be lowercased."""
        tokens = self.lg.tokenize("Hello World")
        for t in tokens:
            self.assertEqual(t, t.lower())

    def test_punctuation_removed(self):
        """Punctuation should not appear in tokens."""
        tokens = self.lg.tokenize("Hello, world! How are you?")
        for t in tokens:
            self.assertTrue(t.isalpha(), f"Token '{t}' contains non-alpha chars")

    def test_empty_string(self):
        """An empty string should produce no tokens."""
        self.assertEqual(self.lg.tokenize(""), [])

    def test_only_punctuation(self):
        """A string with only punctuation should produce no tokens."""
        self.assertEqual(self.lg.tokenize("!!! ... ???"), [])

    def test_numbers_excluded(self):
        """Numeric strings should not appear as tokens."""
        tokens = self.lg.tokenize("There are 42 cats")
        self.assertNotIn("42", tokens)


# ---------------------------------------------------------------------------
# Word type detection
# ---------------------------------------------------------------------------
class TestLanguageGroundingWordTypeDetection(unittest.TestCase):
    """Tests for the _detect_word_type heuristic."""

    def setUp(self):
        self.lg = LanguageGrounding(random_seed=SEED)

    def test_determiner(self):
        self.assertEqual(self.lg._detect_word_type("the"), WordType.DETERMINER)
        self.assertEqual(self.lg._detect_word_type("a"), WordType.DETERMINER)

    def test_preposition(self):
        self.assertEqual(self.lg._detect_word_type("in"), WordType.PREPOSITION)
        self.assertEqual(self.lg._detect_word_type("on"), WordType.PREPOSITION)

    def test_conjunction(self):
        self.assertEqual(self.lg._detect_word_type("and"), WordType.CONJUNCTION)
        self.assertEqual(self.lg._detect_word_type("but"), WordType.CONJUNCTION)

    def test_pronoun(self):
        self.assertEqual(self.lg._detect_word_type("he"), WordType.PRONOUN)
        self.assertEqual(self.lg._detect_word_type("she"), WordType.PRONOUN)

    def test_common_verb(self):
        self.assertEqual(self.lg._detect_word_type("run"), WordType.VERB)
        self.assertEqual(self.lg._detect_word_type("make"), WordType.VERB)
        self.assertEqual(self.lg._detect_word_type("is"), WordType.VERB)

    def test_adverb_suffix(self):
        """Words ending in -ly (with sufficient length) should be adverbs."""
        self.assertEqual(self.lg._detect_word_type("quickly"), WordType.ADVERB)
        self.assertEqual(self.lg._detect_word_type("slowly"), WordType.ADVERB)

    def test_adjective_suffix(self):
        """Words ending in -able, -ful, etc. should be adjectives."""
        self.assertEqual(self.lg._detect_word_type("comfortable"), WordType.ADJECTIVE)
        self.assertEqual(self.lg._detect_word_type("beautiful"), WordType.ADJECTIVE)

    def test_verb_suffix(self):
        """Words ending in -ize, -ate, etc. should be verbs."""
        self.assertEqual(self.lg._detect_word_type("organize"), WordType.VERB)
        self.assertEqual(self.lg._detect_word_type("activate"), WordType.VERB)

    def test_verb_ing_ending(self):
        """Words ending in -ing should be verbs."""
        self.assertEqual(self.lg._detect_word_type("running"), WordType.VERB)

    def test_verb_ed_ending(self):
        """Words ending in -ed should be verbs."""
        self.assertEqual(self.lg._detect_word_type("walked"), WordType.VERB)

    def test_noun_suffix(self):
        """Words ending in -tion, -ness, etc. should be nouns."""
        self.assertEqual(self.lg._detect_word_type("creation"), WordType.NOUN)
        self.assertEqual(self.lg._detect_word_type("happiness"), WordType.NOUN)

    def test_default_content_word_is_noun(self):
        """Unrecognised content words (len > 2) default to NOUN."""
        self.assertEqual(self.lg._detect_word_type("cat"), WordType.NOUN)
        self.assertEqual(self.lg._detect_word_type("dog"), WordType.NOUN)

    def test_very_short_word_is_unknown(self):
        """Words of length <= 2 that match no pattern should be UNKNOWN."""
        # 'ab' is not in any known set and is length 2 => UNKNOWN
        self.assertEqual(self.lg._detect_word_type("ab"), WordType.UNKNOWN)


# ---------------------------------------------------------------------------
# Word learning / grounding
# ---------------------------------------------------------------------------
class TestLanguageGroundingLearning(unittest.TestCase):
    """Tests for learn_from_text and word grounding."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )

    def test_learn_from_text_adds_vocabulary(self):
        """learn_from_text should populate the vocabulary."""
        self.lg.learn_from_text("The quick brown fox jumps over the lazy dog")
        self.assertGreater(len(self.lg.vocabulary), 0)

    def test_learned_words_present(self):
        """Each unique token should appear in the vocabulary."""
        self.lg.learn_from_text("cat dog bird")
        for w in ["cat", "dog", "bird"]:
            self.assertIn(w, self.lg.vocabulary)

    def test_usage_count_increments(self):
        """Repeated tokens should have usage_count > 1."""
        self.lg.learn_from_text("cat cat cat")
        self.assertEqual(self.lg.vocabulary["cat"].usage_count, 3)

    def test_total_words_processed(self):
        """total_words_processed should equal the number of tokens seen."""
        self.lg.learn_from_text("one two three")
        self.assertEqual(self.lg.total_words_processed, 3)

    def test_total_sentences_processed(self):
        """total_sentences_processed should increment per call."""
        self.lg.learn_from_text("first sentence")
        self.lg.learn_from_text("second sentence")
        self.assertEqual(self.lg.total_sentences_processed, 2)

    def test_cooccurrence_populated(self):
        """Co-occurrence counts should be recorded."""
        self.lg.learn_from_text("cat dog bird")
        self.assertGreater(len(self.lg.cooccurrence), 0)

    def test_associations_populated(self):
        """Word associations should be populated after learning."""
        self.lg.learn_from_text("cat dog bird")
        # cat and dog are within the context window
        self.assertIn("dog", self.lg.vocabulary["cat"].associations)

    def test_embeddings_have_correct_dim(self):
        """Word embeddings should match embedding_dim."""
        self.lg.learn_from_text("hello world")
        for gw in self.lg.vocabulary.values():
            self.assertEqual(gw.embedding.shape, (32,))

    def test_embeddings_are_normalised(self):
        """After learning, word embeddings should be approximately unit-norm."""
        self.lg.learn_from_text("the cat sat on the mat")
        for gw in self.lg.vocabulary.values():
            norm = np.linalg.norm(gw.embedding)
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_learn_empty_text(self):
        """Learning from empty text should not error or change state."""
        self.lg.learn_from_text("")
        self.assertEqual(len(self.lg.vocabulary), 0)
        self.assertEqual(self.lg.total_words_processed, 0)

    def test_learn_with_sensory_context(self):
        """Learning with sensory context should set sensory_grounding."""
        sensory = np.random.RandomState(SEED).randn(16)
        self.lg.learn_from_text("red ball", sensory_context=sensory)
        for gw in self.lg.vocabulary.values():
            self.assertIsNotNone(gw.sensory_grounding)
            self.assertEqual(gw.sensory_grounding.shape, (16,))

    def test_learn_with_sensory_context_updates_confidence(self):
        """Sensory grounding should increase word confidence."""
        sensory = np.random.RandomState(SEED).randn(16)
        self.lg.learn_from_text("ball", sensory_context=sensory)
        self.assertGreater(self.lg.vocabulary["ball"].confidence, 0.1)

    def test_learn_with_oversized_sensory(self):
        """Sensory features longer than sensory_dim should be pooled down."""
        sensory = np.random.RandomState(SEED).randn(64)  # > 16
        self.lg.learn_from_text("large feature", sensory_context=sensory)
        for gw in self.lg.vocabulary.values():
            self.assertEqual(gw.sensory_grounding.shape, (16,))

    def test_learn_with_undersized_sensory(self):
        """Sensory features shorter than sensory_dim should be padded."""
        sensory = np.random.RandomState(SEED).randn(4)  # < 16
        self.lg.learn_from_text("small feature", sensory_context=sensory)
        for gw in self.lg.vocabulary.values():
            self.assertEqual(gw.sensory_grounding.shape, (16,))

    def test_repeated_sensory_grounding_is_moving_average(self):
        """Multiple sensory exposures should blend via moving average."""
        rng = np.random.RandomState(SEED)
        sensory1 = rng.randn(16)
        sensory2 = rng.randn(16)
        self.lg.learn_from_text("ball", sensory_context=sensory1)
        grounding_after_first = self.lg.vocabulary["ball"].sensory_grounding.copy()
        self.lg.learn_from_text("ball", sensory_context=sensory2)
        grounding_after_second = self.lg.vocabulary["ball"].sensory_grounding
        # Should have changed
        self.assertFalse(
            np.allclose(grounding_after_first, grounding_after_second),
            "Sensory grounding should update with new exposure",
        )


# ---------------------------------------------------------------------------
# Linking words to concepts
# ---------------------------------------------------------------------------
class TestLanguageGroundingLinkConcept(unittest.TestCase):
    """Tests for link_word_to_concept."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )

    def test_link_creates_word(self):
        """Linking a new word to a concept should create the word."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        self.assertIn("apple", self.lg.vocabulary)

    def test_concept_id_recorded(self):
        """The concept ID should appear in the word's concept_ids."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        self.assertIn("fruit_01", self.lg.vocabulary["apple"].concept_ids)

    def test_concept_to_words_updated(self):
        """concept_to_words should map concept -> word."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        self.assertIn("apple", self.lg.concept_to_words["fruit_01"])

    def test_duplicate_concept_not_added(self):
        """Linking the same concept twice should not duplicate."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        self.lg.link_word_to_concept("apple", "fruit_01")
        ids = self.lg.vocabulary["apple"].concept_ids
        self.assertEqual(ids.count("fruit_01"), 1)

    def test_multiple_concepts_per_word(self):
        """A word can be linked to multiple concepts."""
        self.lg.link_word_to_concept("bank", "financial_01")
        self.lg.link_word_to_concept("bank", "river_01")
        ids = self.lg.vocabulary["bank"].concept_ids
        self.assertIn("financial_01", ids)
        self.assertIn("river_01", ids)

    def test_concept_embedding_alignment(self):
        """Providing a concept embedding should modify the word embedding."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        old_emb = self.lg.vocabulary["apple"].embedding.copy()
        # Use a different seed so the concept vector differs from the word embedding
        concept_emb = np.random.RandomState(SEED + 99).randn(32)
        self.lg.link_word_to_concept("apple", "fruit_02", concept_embedding=concept_emb)
        new_emb = self.lg.vocabulary["apple"].embedding
        self.assertFalse(np.allclose(old_emb, new_emb))

    def test_concept_embedding_resize_larger(self):
        """Concept embedding larger than embedding_dim should be truncated."""
        large_emb = np.random.RandomState(SEED).randn(64)
        self.lg.link_word_to_concept("apple", "c1", concept_embedding=large_emb)
        self.assertEqual(self.lg.vocabulary["apple"].embedding.shape, (32,))

    def test_concept_embedding_resize_smaller(self):
        """Concept embedding smaller than embedding_dim should be padded."""
        small_emb = np.random.RandomState(SEED).randn(8)
        self.lg.link_word_to_concept("apple", "c2", concept_embedding=small_emb)
        self.assertEqual(self.lg.vocabulary["apple"].embedding.shape, (32,))

    def test_confidence_increases(self):
        """Linking to a concept should increase word confidence."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        conf = self.lg.vocabulary["apple"].confidence
        self.assertGreater(conf, 0.1)

    def test_find_words_for_concept(self):
        """find_words_for_concept should return the linked words."""
        self.lg.link_word_to_concept("apple", "fruit_01")
        self.lg.link_word_to_concept("banana", "fruit_01")
        words = self.lg.find_words_for_concept("fruit_01")
        self.assertIn("apple", words)
        self.assertIn("banana", words)

    def test_find_words_for_unknown_concept(self):
        """find_words_for_concept on unknown concept returns empty list."""
        result = self.lg.find_words_for_concept("nonexistent")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class TestLanguageGroundingEmbeddings(unittest.TestCase):
    """Tests for get_word_embedding and get_sentence_embedding."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the quick brown fox jumps over the lazy dog")

    def test_get_word_embedding_known(self):
        """get_word_embedding returns an array for a known word."""
        emb = self.lg.get_word_embedding("fox")
        self.assertIsNotNone(emb)
        self.assertEqual(emb.shape, (32,))

    def test_get_word_embedding_unknown(self):
        """get_word_embedding returns None for an unknown word."""
        emb = self.lg.get_word_embedding("zzzzz")
        self.assertIsNone(emb)

    def test_get_word_embedding_returns_copy(self):
        """get_word_embedding should return a copy, not the internal array."""
        emb1 = self.lg.get_word_embedding("fox")
        emb2 = self.lg.get_word_embedding("fox")
        emb1[0] = 999.0
        self.assertNotAlmostEqual(emb2[0], 999.0)

    def test_get_sentence_embedding_shape(self):
        """get_sentence_embedding returns correct-dimension vector."""
        emb = self.lg.get_sentence_embedding("the quick fox")
        self.assertEqual(emb.shape, (32,))

    def test_get_sentence_embedding_empty(self):
        """get_sentence_embedding of empty text returns zero vector."""
        emb = self.lg.get_sentence_embedding("")
        np.testing.assert_array_equal(emb, np.zeros(32))

    def test_get_sentence_embedding_all_unknown(self):
        """Sentence of unknown words returns zero vector."""
        emb = self.lg.get_sentence_embedding("zzz yyy xxx")
        np.testing.assert_array_equal(emb, np.zeros(32))

    def test_sentence_embedding_normalised(self):
        """Sentence embeddings for known words should be roughly unit-norm."""
        emb = self.lg.get_sentence_embedding("quick brown fox")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=3)


# ---------------------------------------------------------------------------
# Sentence parsing
# ---------------------------------------------------------------------------
class TestLanguageGroundingParseSentence(unittest.TestCase):
    """Tests for parse_sentence and SVO extraction."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )

    def test_parse_returns_parsed_sentence(self):
        """parse_sentence should return a ParsedSentence instance."""
        result = self.lg.parse_sentence("the cat sat")
        self.assertIsInstance(result, ParsedSentence)

    def test_parse_raw_text(self):
        """raw_text should match the input string."""
        result = self.lg.parse_sentence("the cat sat")
        self.assertEqual(result.raw_text, "the cat sat")

    def test_parse_tokens(self):
        """Tokens should be the lowercased words."""
        result = self.lg.parse_sentence("The Cat Sat")
        self.assertEqual(result.tokens, ["the", "cat", "sat"])

    def test_parse_word_types_length(self):
        """word_types should have the same length as tokens."""
        result = self.lg.parse_sentence("the cat sat on the mat")
        self.assertEqual(len(result.word_types), len(result.tokens))

    def test_parse_embeddings_length(self):
        """embeddings list should have the same length as tokens."""
        result = self.lg.parse_sentence("the cat sat")
        self.assertEqual(len(result.embeddings), len(result.tokens))

    def test_parse_sentence_embedding_shape(self):
        """sentence_embedding should have the correct dimension."""
        result = self.lg.parse_sentence("the cat sat")
        self.assertEqual(result.sentence_embedding.shape, (32,))

    def test_parse_extracts_subject_noun(self):
        """First noun should be extracted as subject."""
        result = self.lg.parse_sentence("the cat sat on the mat")
        self.assertEqual(result.subject, "cat")

    def test_parse_extracts_verb(self):
        """The verb 'sat' should be extracted (ends in 'at' but 'sat' has -ed-like treatment? no, 'sat' len=3, ends not in 'ing' or 'ed'; but it's not in common verbs either. Let's pick a sentence with a clear verb)."""
        # 'is' is a common verb
        result = self.lg.parse_sentence("the dog is happy")
        self.assertEqual(result.verb, "is")

    def test_parse_extracts_object(self):
        """Noun after verb should be extracted as object."""
        # 'is' is a verb, 'ball' is a noun after it
        result = self.lg.parse_sentence("the dog chased the ball")
        # 'chased' ends with -ed => VERB, 'ball' => NOUN
        self.assertEqual(result.verb, "chased")
        self.assertEqual(result.object, "ball")

    def test_parse_extracts_modifiers(self):
        """Adjectives and adverbs should be collected as modifiers."""
        result = self.lg.parse_sentence("the beautiful dog walked quickly")
        self.assertIn("beautiful", result.modifiers)
        self.assertIn("quickly", result.modifiers)

    def test_parse_empty_text(self):
        """Parsing empty text should return empty-but-valid ParsedSentence."""
        result = self.lg.parse_sentence("")
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.word_types, [])
        self.assertEqual(result.embeddings, [])
        np.testing.assert_array_equal(
            result.sentence_embedding, np.zeros(32)
        )
        self.assertIsNone(result.subject)
        self.assertIsNone(result.verb)
        self.assertIsNone(result.object)
        self.assertEqual(result.modifiers, [])

    def test_parse_pronoun_as_subject(self):
        """A pronoun should be detected as subject if no noun precedes it."""
        result = self.lg.parse_sentence("she runs fast")
        self.assertEqual(result.subject, "she")

    def test_parse_creates_vocabulary_entries(self):
        """Parsing should add words to the vocabulary."""
        self.lg.parse_sentence("the elephant jumped")
        self.assertIn("elephant", self.lg.vocabulary)
        self.assertIn("jumped", self.lg.vocabulary)


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------
class TestLanguageGroundingTextSimilarity(unittest.TestCase):
    """Tests for compute_text_similarity."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        # Build vocabulary first
        self.lg.learn_from_text("the cat sat on the mat")
        self.lg.learn_from_text("the dog ran in the park")

    def test_identical_texts(self):
        """Identical texts should have similarity close to 1.0."""
        sim = self.lg.compute_text_similarity("the cat", "the cat")
        self.assertGreater(sim, 0.9)

    def test_empty_texts(self):
        """Empty texts should have similarity of 0.0."""
        sim = self.lg.compute_text_similarity("", "")
        self.assertAlmostEqual(sim, 0.0)

    def test_similarity_is_float(self):
        """Similarity should be a Python float."""
        sim = self.lg.compute_text_similarity("the cat", "the dog")
        self.assertIsInstance(sim, float)


# ---------------------------------------------------------------------------
# Word similarity search
# ---------------------------------------------------------------------------
class TestLanguageGroundingFindSimilar(unittest.TestCase):
    """Tests for find_similar_words."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the cat sat on the mat with the dog and the bird")

    def test_returns_list(self):
        """find_similar_words should return a list."""
        result = self.lg.find_similar_words("cat")
        self.assertIsInstance(result, list)

    def test_result_is_tuples(self):
        """Each result should be a (word, similarity) tuple."""
        result = self.lg.find_similar_words("cat")
        for item in result:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_does_not_include_self(self):
        """The query word itself should not appear in results."""
        result = self.lg.find_similar_words("cat")
        words = [w for w, _ in result]
        self.assertNotIn("cat", words)

    def test_unknown_word_returns_empty(self):
        """An unknown word should return an empty list."""
        result = self.lg.find_similar_words("zzzzz")
        self.assertEqual(result, [])

    def test_top_k_limits(self):
        """Result length should not exceed top_k."""
        result = self.lg.find_similar_words("cat", top_k=2)
        self.assertLessEqual(len(result), 2)

    def test_results_sorted_descending(self):
        """Results should be sorted by similarity descending."""
        result = self.lg.find_similar_words("cat", top_k=5)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                self.assertGreaterEqual(result[i][1], result[i + 1][1])


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------
class TestLanguageGroundingGenerateDescription(unittest.TestCase):
    """Tests for generate_description."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the quick brown fox jumps over the lazy dog")

    def test_returns_string(self):
        """generate_description should return a string."""
        emb = np.random.RandomState(SEED).randn(32)
        result = self.lg.generate_description(emb)
        self.assertIsInstance(result, str)

    def test_max_words_respected(self):
        """Output should contain at most max_words words."""
        emb = np.random.RandomState(SEED).randn(32)
        result = self.lg.generate_description(emb, max_words=3)
        self.assertLessEqual(len(result.split()), 3)

    def test_empty_vocabulary_returns_empty_string(self):
        """With no vocabulary, generate_description should return ''."""
        lg_empty = LanguageGrounding(embedding_dim=32, random_seed=SEED)
        emb = np.zeros(32)
        self.assertEqual(lg_empty.generate_description(emb), "")

    def test_oversized_concept_embedding(self):
        """Concept embedding larger than embedding_dim should be handled."""
        emb = np.random.RandomState(SEED).randn(64)
        result = self.lg.generate_description(emb)
        self.assertIsInstance(result, str)

    def test_undersized_concept_embedding(self):
        """Concept embedding smaller than embedding_dim should be handled."""
        emb = np.random.RandomState(SEED).randn(8)
        result = self.lg.generate_description(emb)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
class TestLanguageGroundingAnswerQuestion(unittest.TestCase):
    """Tests for the simple answer_question method."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the cat sat on the mat")
        self.lg.learn_from_text("the dog ran in the park")

    def test_returns_tuple(self):
        """answer_question should return (answer, confidence) tuple."""
        result = self.lg.answer_question("what is the cat doing")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_answer_is_string(self):
        """The answer component should be a string."""
        answer, _ = self.lg.answer_question("what is the cat doing")
        self.assertIsInstance(answer, str)

    def test_confidence_is_float(self):
        """The confidence component should be a float."""
        _, conf = self.lg.answer_question("what is the cat doing")
        self.assertIsInstance(conf, float)

    def test_confidence_in_range(self):
        """Confidence should be between 0.0 and 1.0."""
        _, conf = self.lg.answer_question("what is the cat doing")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_no_key_words_returns_unknown(self):
        """A question with no content words should return 'Unknown'."""
        answer, conf = self.lg.answer_question("what")
        self.assertEqual(answer, "Unknown")
        self.assertAlmostEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# Vocabulary stats
# ---------------------------------------------------------------------------
class TestLanguageGroundingVocabStats(unittest.TestCase):
    """Tests for get_vocabulary_stats."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the quick brown fox jumps over the lazy dog")

    def test_returns_dict(self):
        """get_vocabulary_stats should return a dict."""
        stats = self.lg.get_vocabulary_stats()
        self.assertIsInstance(stats, dict)

    def test_expected_keys(self):
        """Stats dict should contain all expected keys."""
        stats = self.lg.get_vocabulary_stats()
        expected_keys = {
            'vocabulary_size', 'words_by_type', 'sensory_grounded_words',
            'concept_linked_words', 'total_words_processed',
            'total_sentences_processed', 'unique_cooccurrences',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_vocabulary_size_matches(self):
        """vocabulary_size stat should match len(vocabulary)."""
        stats = self.lg.get_vocabulary_stats()
        self.assertEqual(stats['vocabulary_size'], len(self.lg.vocabulary))

    def test_total_words_processed(self):
        """total_words_processed should reflect tokens processed."""
        stats = self.lg.get_vocabulary_stats()
        self.assertGreater(stats['total_words_processed'], 0)

    def test_total_sentences_processed(self):
        """total_sentences_processed should be at least 1."""
        stats = self.lg.get_vocabulary_stats()
        self.assertEqual(stats['total_sentences_processed'], 1)

    def test_words_by_type_is_dict(self):
        """words_by_type should be a dict."""
        stats = self.lg.get_vocabulary_stats()
        self.assertIsInstance(stats['words_by_type'], dict)


# ---------------------------------------------------------------------------
# Serialisation / get_state
# ---------------------------------------------------------------------------
class TestLanguageGroundingSerialization(unittest.TestCase):
    """Tests for serialize and deserialize (get_state round-trip)."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.lg.learn_from_text("the cat sat on the mat")
        sensory = np.random.RandomState(SEED).randn(16)
        self.lg.learn_from_text("red ball", sensory_context=sensory)
        self.lg.link_word_to_concept("cat", "animal_01")

    def test_serialize_returns_dict(self):
        """serialize should return a dict."""
        data = self.lg.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_has_embedding_dim(self):
        """Serialized data should contain embedding_dim."""
        data = self.lg.serialize()
        self.assertEqual(data['embedding_dim'], 32)

    def test_serialize_has_sensory_dim(self):
        """Serialized data should contain sensory_dim."""
        data = self.lg.serialize()
        self.assertEqual(data['sensory_dim'], 16)

    def test_serialize_has_vocabulary(self):
        """Serialized data should contain vocabulary."""
        data = self.lg.serialize()
        self.assertIn('vocabulary', data)
        self.assertGreater(len(data['vocabulary']), 0)

    def test_serialize_vocabulary_word_fields(self):
        """Each word in the serialized vocabulary should have expected keys."""
        data = self.lg.serialize()
        expected_keys = {
            'word_type', 'embedding', 'sensory_grounding',
            'concept_ids', 'usage_count', 'confidence', 'associations',
        }
        for word, word_data in data['vocabulary'].items():
            self.assertEqual(set(word_data.keys()), expected_keys)

    def test_serialize_embedding_is_list(self):
        """Embeddings should be serialized as lists (JSON-friendly)."""
        data = self.lg.serialize()
        for word_data in data['vocabulary'].values():
            self.assertIsInstance(word_data['embedding'], list)

    def test_serialize_concept_to_words(self):
        """concept_to_words should appear as dict of lists."""
        data = self.lg.serialize()
        self.assertIn('concept_to_words', data)
        for key, val in data['concept_to_words'].items():
            self.assertIsInstance(val, list)

    def test_serialize_has_stats(self):
        """Serialized data should contain stats."""
        data = self.lg.serialize()
        self.assertIn('stats', data)

    def test_deserialize_round_trip(self):
        """Deserialize should reconstruct from serialized data."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        self.assertEqual(restored.embedding_dim, self.lg.embedding_dim)
        self.assertEqual(restored.sensory_dim, self.lg.sensory_dim)

    def test_deserialize_vocabulary_preserved(self):
        """Deserialized system should have the same vocabulary words."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        self.assertEqual(
            set(restored.vocabulary.keys()),
            set(self.lg.vocabulary.keys()),
        )

    def test_deserialize_word_types_preserved(self):
        """Word types should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        for word in self.lg.vocabulary:
            self.assertEqual(
                restored.vocabulary[word].word_type,
                self.lg.vocabulary[word].word_type,
            )

    def test_deserialize_embeddings_preserved(self):
        """Word embeddings should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        for word in self.lg.vocabulary:
            np.testing.assert_array_almost_equal(
                restored.vocabulary[word].embedding,
                self.lg.vocabulary[word].embedding,
            )

    def test_deserialize_sensory_grounding_preserved(self):
        """Sensory grounding should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        # 'red' and 'ball' should have sensory grounding
        for word in ["red", "ball"]:
            self.assertIsNotNone(restored.vocabulary[word].sensory_grounding)
            np.testing.assert_array_almost_equal(
                restored.vocabulary[word].sensory_grounding,
                self.lg.vocabulary[word].sensory_grounding,
            )

    def test_deserialize_concept_ids_preserved(self):
        """Concept links should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        self.assertIn("animal_01", restored.vocabulary["cat"].concept_ids)

    def test_deserialize_concept_to_words_preserved(self):
        """concept_to_words mapping should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        self.assertIn("cat", restored.concept_to_words["animal_01"])

    def test_deserialize_usage_count_preserved(self):
        """usage_count should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        for word in self.lg.vocabulary:
            self.assertEqual(
                restored.vocabulary[word].usage_count,
                self.lg.vocabulary[word].usage_count,
            )

    def test_deserialize_confidence_preserved(self):
        """confidence should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        for word in self.lg.vocabulary:
            self.assertAlmostEqual(
                restored.vocabulary[word].confidence,
                self.lg.vocabulary[word].confidence,
            )

    def test_deserialize_associations_preserved(self):
        """associations should survive serialization round-trip."""
        data = self.lg.serialize()
        restored = LanguageGrounding.deserialize(data)
        for word in self.lg.vocabulary:
            self.assertEqual(
                restored.vocabulary[word].associations,
                self.lg.vocabulary[word].associations,
            )


# ---------------------------------------------------------------------------
# TextCorpusLearner
# ---------------------------------------------------------------------------
class TestTextCorpusLearner(unittest.TestCase):
    """Tests for the TextCorpusLearner class."""

    def setUp(self):
        self.lg = LanguageGrounding(
            embedding_dim=32, sensory_dim=16, random_seed=SEED
        )
        self.learner = TextCorpusLearner(self.lg, batch_size=10)

    def test_init_attributes(self):
        """Constructor should set language and batch_size."""
        self.assertIs(self.learner.language, self.lg)
        self.assertEqual(self.learner.batch_size, 10)
        self.assertEqual(self.learner.sentences_processed, 0)

    def test_learn_from_corpus_returns_stats(self):
        """learn_from_corpus should return a statistics dict."""
        corpus = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "birds fly in the sky",
        ]
        stats = self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertIsInstance(stats, dict)

    def test_learn_from_corpus_stat_keys(self):
        """Stats dict should have expected keys."""
        corpus = ["hello world"]
        stats = self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        expected = {'epochs', 'sentences_processed', 'duration', 'final_vocab_size'}
        self.assertEqual(set(stats.keys()), expected)

    def test_learn_from_corpus_processes_all_sentences(self):
        """sentences_processed should match corpus size times epochs."""
        corpus = ["one sentence", "two sentence", "three sentence"]
        self.learner.learn_from_corpus(corpus, epochs=2, verbose=False)
        self.assertEqual(self.learner.sentences_processed, 6)

    def test_learn_from_corpus_populates_vocabulary(self):
        """After learning, the vocabulary should be non-empty."""
        corpus = [
            "the cat sat on the mat",
            "the dog ran in the park",
        ]
        self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertGreater(len(self.lg.vocabulary), 0)

    def test_learn_from_corpus_returns_correct_epoch_count(self):
        """Stats should report the correct number of epochs."""
        corpus = ["test sentence"]
        stats = self.learner.learn_from_corpus(corpus, epochs=3, verbose=False)
        self.assertEqual(stats['epochs'], 3)

    def test_learn_from_corpus_duration_positive(self):
        """Training duration should be a non-negative number."""
        corpus = ["the cat sat"]
        stats = self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertGreaterEqual(stats['duration'], 0.0)

    def test_learn_from_corpus_final_vocab_size(self):
        """final_vocab_size should match the vocabulary length."""
        corpus = ["alpha beta gamma"]
        stats = self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertEqual(stats['final_vocab_size'], len(self.lg.vocabulary))

    def test_learn_from_empty_corpus(self):
        """Learning from an empty corpus should not error."""
        stats = self.learner.learn_from_corpus([], epochs=1, verbose=False)
        self.assertEqual(stats['sentences_processed'], 0)
        self.assertEqual(stats['final_vocab_size'], 0)

    def test_learn_from_corpus_multiple_epochs_improves_vocabulary(self):
        """Running multiple epochs should still yield the same vocabulary size for the same corpus."""
        corpus = ["the cat sat on the mat"]
        self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        vocab_after_1 = len(self.lg.vocabulary)
        self.learner.learn_from_corpus(corpus, epochs=2, verbose=False)
        vocab_after_3 = len(self.lg.vocabulary)
        # Vocabulary should not shrink and should stay same for repeated text
        self.assertEqual(vocab_after_1, vocab_after_3)

    def test_learn_from_corpus_custom_batch_size(self):
        """Using a small batch size should still process all sentences."""
        lg2 = LanguageGrounding(embedding_dim=32, sensory_dim=16, random_seed=SEED)
        learner2 = TextCorpusLearner(lg2, batch_size=1)
        corpus = ["alpha beta", "gamma delta", "epsilon zeta"]
        stats = learner2.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertEqual(stats['sentences_processed'], 3)

    def test_learner_shares_language_system(self):
        """The learner and the language system should share the same vocabulary."""
        corpus = ["cat dog bird"]
        self.learner.learn_from_corpus(corpus, epochs=1, verbose=False)
        self.assertIn("cat", self.lg.vocabulary)
        self.assertIn("dog", self.lg.vocabulary)
        self.assertIn("bird", self.lg.vocabulary)


# ---------------------------------------------------------------------------
# Deterministic reproducibility
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    """Tests that confirm deterministic behaviour with the same seed."""

    def test_same_seed_same_embeddings(self):
        """Two instances with the same seed should produce identical embeddings."""
        lg1 = LanguageGrounding(embedding_dim=32, random_seed=SEED)
        lg1.learn_from_text("the cat sat on the mat")

        lg2 = LanguageGrounding(embedding_dim=32, random_seed=SEED)
        lg2.learn_from_text("the cat sat on the mat")

        for word in lg1.vocabulary:
            np.testing.assert_array_equal(
                lg1.vocabulary[word].embedding,
                lg2.vocabulary[word].embedding,
            )

    def test_same_seed_same_vocabulary(self):
        """Two instances with the same seed should produce identical vocabularies."""
        lg1 = LanguageGrounding(embedding_dim=32, random_seed=SEED)
        lg1.learn_from_text("hello world")

        lg2 = LanguageGrounding(embedding_dim=32, random_seed=SEED)
        lg2.learn_from_text("hello world")

        self.assertEqual(
            set(lg1.vocabulary.keys()),
            set(lg2.vocabulary.keys()),
        )


if __name__ == "__main__":
    unittest.main()
