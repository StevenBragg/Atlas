"""
Language Grounding System for ATLAS

Implements symbol-to-concept grounding that connects language (words, sentences)
to sensory-grounded concepts in semantic memory. This enables:
- Learning word meanings from perceptual experience
- Understanding text descriptions
- Generating language from internal representations
- Accessing human knowledge through text

Key insight: Words are grounded in sensory experience, not just other words.
This avoids the "symbol grounding problem" by anchoring symbols in perception.
"""

import numpy as np
import logging
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class WordType(Enum):
    """Types of words for grammatical processing"""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    DETERMINER = "determiner"
    CONJUNCTION = "conjunction"
    PRONOUN = "pronoun"
    UNKNOWN = "unknown"


@dataclass
class GroundedWord:
    """A word grounded in sensory/conceptual space"""
    word: str
    word_type: WordType
    embedding: np.ndarray  # Distributed representation
    sensory_grounding: Optional[np.ndarray] = None  # Link to sensory features
    concept_ids: List[str] = field(default_factory=list)  # Linked semantic concepts
    usage_count: int = 0
    confidence: float = 0.5
    associations: Dict[str, float] = field(default_factory=dict)  # Word associations


@dataclass
class ParsedSentence:
    """A parsed sentence with structure"""
    raw_text: str
    tokens: List[str]
    word_types: List[WordType]
    embeddings: List[np.ndarray]
    sentence_embedding: np.ndarray
    subject: Optional[str] = None
    verb: Optional[str] = None
    object: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)


class LanguageGrounding:
    """
    Language grounding system that connects words to sensory concepts.

    This implements a biologically-plausible approach where:
    1. Words are learned through co-occurrence with perceptual experience
    2. Word meanings are distributed representations grounded in sensory features
    3. Compositional semantics emerge from combining grounded words
    4. Language production generates words from conceptual representations
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        vocabulary_size: int = 10000,
        learning_rate: float = 0.01,
        context_window: int = 5,
        sensory_dim: int = 64,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the language grounding system.

        Args:
            embedding_dim: Dimension of word embeddings
            vocabulary_size: Maximum vocabulary size
            learning_rate: Learning rate for updates
            context_window: Context window for co-occurrence learning
            sensory_dim: Dimension of sensory grounding vectors
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.context_window = context_window
        self.sensory_dim = sensory_dim

        # Vocabulary: word -> GroundedWord
        self.vocabulary: Dict[str, GroundedWord] = {}

        # Reverse mapping: concept_id -> words
        self.concept_to_words: Dict[str, Set[str]] = defaultdict(set)

        # Word co-occurrence matrix (sparse representation)
        self.cooccurrence: Dict[Tuple[str, str], float] = defaultdict(float)

        # Sensory-word associations: sensory_pattern -> words
        self.sensory_word_matrix: Optional[np.ndarray] = None

        # Common word patterns for type detection
        self._init_word_patterns()

        # Statistics
        self.total_words_processed = 0
        self.total_sentences_processed = 0

        logger.info(f"LanguageGrounding initialized: dim={embedding_dim}, vocab={vocabulary_size}")

    def _init_word_patterns(self) -> None:
        """Initialize patterns for word type detection."""
        # Common suffixes for word type detection
        self.noun_suffixes = {'tion', 'ness', 'ment', 'ity', 'er', 'or', 'ist', 'ism'}
        self.verb_suffixes = {'ate', 'ify', 'ize', 'en'}
        self.adj_suffixes = {'able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ic'}
        self.adv_suffixes = {'ly'}

        # Common function words
        self.determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        self.prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over'}
        self.conjunctions = {'and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'because', 'although', 'while', 'if', 'when', 'where', 'unless', 'until'}
        self.pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}

        # Common verbs (base forms)
        self.common_verbs = {'be', 'have', 'do', 'say', 'go', 'get', 'make', 'know', 'think', 'take', 'see', 'come', 'want', 'look', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'is', 'are', 'was', 'were', 'has', 'had', 'does', 'did'}

    def _detect_word_type(self, word: str) -> WordType:
        """Detect the type of a word using heuristics."""
        word_lower = word.lower()

        # Check function words first
        if word_lower in self.determiners:
            return WordType.DETERMINER
        if word_lower in self.prepositions:
            return WordType.PREPOSITION
        if word_lower in self.conjunctions:
            return WordType.CONJUNCTION
        if word_lower in self.pronouns:
            return WordType.PRONOUN
        if word_lower in self.common_verbs:
            return WordType.VERB

        # Check suffixes
        for suffix in self.adv_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return WordType.ADVERB

        for suffix in self.adj_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return WordType.ADJECTIVE

        for suffix in self.verb_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return WordType.VERB

        # Check verb endings
        if word_lower.endswith('ing') or word_lower.endswith('ed'):
            return WordType.VERB

        for suffix in self.noun_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return WordType.NOUN

        # Default to noun for content words
        if len(word_lower) > 2:
            return WordType.NOUN

        return WordType.UNKNOWN

    def _get_or_create_word(self, word: str) -> GroundedWord:
        """Get existing word or create new one."""
        word_lower = word.lower()

        if word_lower not in self.vocabulary:
            # Create new word with random initial embedding
            embedding = np.random.randn(self.embedding_dim) * 0.1
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            word_type = self._detect_word_type(word)

            self.vocabulary[word_lower] = GroundedWord(
                word=word_lower,
                word_type=word_type,
                embedding=embedding,
                sensory_grounding=None,
                concept_ids=[],
                usage_count=0,
                confidence=0.1,
            )

        return self.vocabulary[word_lower]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization: split on whitespace and punctuation
        # Remove punctuation but keep words
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def learn_from_text(
        self,
        text: str,
        sensory_context: Optional[np.ndarray] = None,
    ) -> None:
        """
        Learn word representations from text, optionally with sensory context.

        This implements grounded language learning where words are associated
        with co-occurring sensory experiences.

        Args:
            text: Text to learn from
            sensory_context: Optional sensory features present during text exposure
        """
        tokens = self.tokenize(text)

        if len(tokens) == 0:
            return

        # Process each word
        for i, token in enumerate(tokens):
            word = self._get_or_create_word(token)
            word.usage_count += 1

            # Update co-occurrence with context words
            start = max(0, i - self.context_window)
            end = min(len(tokens), i + self.context_window + 1)

            for j in range(start, end):
                if i != j:
                    context_word = tokens[j]
                    self.cooccurrence[(token, context_word)] += 1.0

                    # Update word associations
                    if context_word not in word.associations:
                        word.associations[context_word] = 0.0
                    word.associations[context_word] += 1.0

            # Ground in sensory context if available
            if sensory_context is not None:
                self._ground_word_in_sensory(word, sensory_context)

        # Update embeddings based on co-occurrence
        self._update_embeddings_from_cooccurrence(tokens)

        self.total_words_processed += len(tokens)
        self.total_sentences_processed += 1

    def _ground_word_in_sensory(
        self,
        word: GroundedWord,
        sensory_features: np.ndarray,
    ) -> None:
        """Ground a word in sensory experience."""
        # Resize sensory features if needed
        if len(sensory_features) != self.sensory_dim:
            # Simple pooling/padding
            if len(sensory_features) > self.sensory_dim:
                # Pool
                indices = np.linspace(0, len(sensory_features) - 1, self.sensory_dim).astype(int)
                sensory_features = sensory_features[indices]
            else:
                # Pad
                padded = np.zeros(self.sensory_dim)
                padded[:len(sensory_features)] = sensory_features
                sensory_features = padded

        # Normalize
        sensory_norm = sensory_features / (np.linalg.norm(sensory_features) + 1e-8)

        if word.sensory_grounding is None:
            word.sensory_grounding = sensory_norm.copy()
        else:
            # Exponential moving average
            word.sensory_grounding = (
                0.9 * word.sensory_grounding + 0.1 * sensory_norm
            )
            word.sensory_grounding /= np.linalg.norm(word.sensory_grounding) + 1e-8

        # Increase confidence
        word.confidence = min(1.0, word.confidence + 0.01)

    def _update_embeddings_from_cooccurrence(self, tokens: List[str]) -> None:
        """Update word embeddings based on co-occurrence patterns."""
        # Simple skip-gram style update
        for i, token in enumerate(tokens):
            word = self.vocabulary.get(token)
            if word is None:
                continue

            start = max(0, i - self.context_window)
            end = min(len(tokens), i + self.context_window + 1)

            for j in range(start, end):
                if i != j:
                    context_token = tokens[j]
                    context_word = self.vocabulary.get(context_token)
                    if context_word is None:
                        continue

                    # Move embeddings closer together
                    similarity = np.dot(word.embedding, context_word.embedding)
                    target = 1.0  # Positive co-occurrence
                    error = target - similarity

                    # Gradient update
                    word.embedding += self.learning_rate * error * context_word.embedding
                    context_word.embedding += self.learning_rate * error * word.embedding

                    # Normalize
                    word.embedding /= np.linalg.norm(word.embedding) + 1e-8
                    context_word.embedding /= np.linalg.norm(context_word.embedding) + 1e-8

    def link_word_to_concept(
        self,
        word: str,
        concept_id: str,
        concept_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Link a word to a semantic concept.

        Args:
            word: Word to link
            concept_id: ID of the semantic concept
            concept_embedding: Optional embedding from semantic memory
        """
        grounded_word = self._get_or_create_word(word)

        if concept_id not in grounded_word.concept_ids:
            grounded_word.concept_ids.append(concept_id)

        self.concept_to_words[concept_id].add(word.lower())

        # Align word embedding with concept embedding
        if concept_embedding is not None:
            # Resize if needed
            if len(concept_embedding) != self.embedding_dim:
                if len(concept_embedding) > self.embedding_dim:
                    concept_embedding = concept_embedding[:self.embedding_dim]
                else:
                    padded = np.zeros(self.embedding_dim)
                    padded[:len(concept_embedding)] = concept_embedding
                    concept_embedding = padded

            # Move word embedding toward concept embedding
            concept_norm = concept_embedding / (np.linalg.norm(concept_embedding) + 1e-8)
            grounded_word.embedding = (
                0.7 * grounded_word.embedding + 0.3 * concept_norm
            )
            grounded_word.embedding /= np.linalg.norm(grounded_word.embedding) + 1e-8

        grounded_word.confidence = min(1.0, grounded_word.confidence + 0.05)

        logger.debug(f"Linked word '{word}' to concept '{concept_id}'")

    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding for a word."""
        word_lower = word.lower()
        if word_lower in self.vocabulary:
            return self.vocabulary[word_lower].embedding.copy()
        return None

    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Get a sentence embedding by composing word embeddings.

        Uses weighted average with position encoding.
        """
        tokens = self.tokenize(text)

        if len(tokens) == 0:
            return np.zeros(self.embedding_dim)

        embeddings = []
        weights = []

        for i, token in enumerate(tokens):
            word = self.vocabulary.get(token)
            if word is not None:
                embeddings.append(word.embedding)
                # Weight by confidence and position (content words weighted higher)
                weight = word.confidence
                if word.word_type in [WordType.NOUN, WordType.VERB, WordType.ADJECTIVE]:
                    weight *= 1.5
                weights.append(weight)

        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)

        # Weighted average
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)

        sentence_embedding = np.zeros(self.embedding_dim)
        for emb, w in zip(embeddings, weights):
            sentence_embedding += w * emb

        # Normalize
        sentence_embedding /= np.linalg.norm(sentence_embedding) + 1e-8

        return sentence_embedding

    def parse_sentence(self, text: str) -> ParsedSentence:
        """
        Parse a sentence into structured representation.

        Extracts subject, verb, object, and modifiers using simple heuristics.
        """
        tokens = self.tokenize(text)

        if len(tokens) == 0:
            return ParsedSentence(
                raw_text=text,
                tokens=[],
                word_types=[],
                embeddings=[],
                sentence_embedding=np.zeros(self.embedding_dim),
            )

        word_types = []
        embeddings = []

        for token in tokens:
            word = self._get_or_create_word(token)
            word_types.append(word.word_type)
            embeddings.append(word.embedding.copy())

        sentence_embedding = self.get_sentence_embedding(text)

        # Simple SVO extraction
        subject = None
        verb = None
        obj = None
        modifiers = []

        # Find first noun (likely subject)
        for i, (token, wtype) in enumerate(zip(tokens, word_types)):
            if wtype == WordType.NOUN:
                subject = token
                break
            elif wtype == WordType.PRONOUN:
                subject = token
                break

        # Find first verb
        for i, (token, wtype) in enumerate(zip(tokens, word_types)):
            if wtype == WordType.VERB:
                verb = token
                break

        # Find object (noun after verb)
        found_verb = False
        for i, (token, wtype) in enumerate(zip(tokens, word_types)):
            if wtype == WordType.VERB:
                found_verb = True
            elif found_verb and wtype == WordType.NOUN:
                obj = token
                break

        # Collect modifiers
        for token, wtype in zip(tokens, word_types):
            if wtype in [WordType.ADJECTIVE, WordType.ADVERB]:
                modifiers.append(token)

        return ParsedSentence(
            raw_text=text,
            tokens=tokens,
            word_types=word_types,
            embeddings=embeddings,
            sentence_embedding=sentence_embedding,
            subject=subject,
            verb=verb,
            object=obj,
            modifiers=modifiers,
        )

    def find_similar_words(
        self,
        word: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find words with similar embeddings."""
        word_lower = word.lower()
        if word_lower not in self.vocabulary:
            return []

        target_embedding = self.vocabulary[word_lower].embedding

        similarities = []
        for other_word, grounded in self.vocabulary.items():
            if other_word != word_lower:
                sim = np.dot(target_embedding, grounded.embedding)
                similarities.append((other_word, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_words_for_concept(self, concept_id: str) -> List[str]:
        """Find all words linked to a concept."""
        return list(self.concept_to_words.get(concept_id, set()))

    def generate_description(
        self,
        concept_embedding: np.ndarray,
        max_words: int = 10,
    ) -> str:
        """
        Generate a text description from a conceptual representation.

        This is a simple word retrieval approach - more sophisticated
        generation would require additional mechanisms.

        Args:
            concept_embedding: Embedding to describe
            max_words: Maximum words in description

        Returns:
            Generated description
        """
        if len(self.vocabulary) == 0:
            return ""

        # Resize concept embedding if needed
        if len(concept_embedding) != self.embedding_dim:
            if len(concept_embedding) > self.embedding_dim:
                concept_embedding = concept_embedding[:self.embedding_dim]
            else:
                padded = np.zeros(self.embedding_dim)
                padded[:len(concept_embedding)] = concept_embedding
                concept_embedding = padded

        concept_norm = concept_embedding / (np.linalg.norm(concept_embedding) + 1e-8)

        # Find most similar words
        word_scores = []
        for word, grounded in self.vocabulary.items():
            # Prefer content words
            type_bonus = 1.0
            if grounded.word_type in [WordType.NOUN, WordType.VERB, WordType.ADJECTIVE]:
                type_bonus = 1.5

            similarity = np.dot(concept_norm, grounded.embedding) * type_bonus * grounded.confidence
            word_scores.append((word, similarity, grounded.word_type))

        word_scores.sort(key=lambda x: x[1], reverse=True)

        # Select diverse words (different types)
        selected = []
        types_used = set()

        for word, score, wtype in word_scores:
            if len(selected) >= max_words:
                break

            # Prefer diverse types
            if wtype not in types_used or len(selected) < 3:
                selected.append(word)
                types_used.add(wtype)

        # Simple ordering: adjectives before nouns, adverbs before verbs
        nouns = [w for w in selected if self.vocabulary[w].word_type == WordType.NOUN]
        verbs = [w for w in selected if self.vocabulary[w].word_type == WordType.VERB]
        adjs = [w for w in selected if self.vocabulary[w].word_type == WordType.ADJECTIVE]
        advs = [w for w in selected if self.vocabulary[w].word_type == WordType.ADVERB]
        others = [w for w in selected if w not in nouns + verbs + adjs + advs]

        # Compose: adj noun verb adv
        ordered = adjs[:2] + nouns[:2] + verbs[:2] + advs[:1] + others[:2]

        return " ".join(ordered[:max_words])

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        emb1 = self.get_sentence_embedding(text1)
        emb2 = self.get_sentence_embedding(text2)

        return float(np.dot(emb1, emb2))

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Simple question answering based on learned associations.

        Args:
            question: Question to answer
            context: Optional context text

        Returns:
            (answer, confidence)
        """
        # Parse question
        parsed = self.parse_sentence(question)

        # Identify question type and target
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        question_type = None
        for token in parsed.tokens:
            if token in question_words:
                question_type = token
                break

        # Find key concepts in question
        key_words = []
        for token, wtype in zip(parsed.tokens, parsed.word_types):
            if wtype in [WordType.NOUN, WordType.VERB] and token not in question_words:
                key_words.append(token)

        if len(key_words) == 0:
            return "Unknown", 0.0

        # Find associated words
        all_associations = defaultdict(float)
        for key_word in key_words:
            if key_word in self.vocabulary:
                word = self.vocabulary[key_word]
                for assoc_word, strength in word.associations.items():
                    if assoc_word not in key_words and assoc_word not in question_words:
                        all_associations[assoc_word] += strength

        if len(all_associations) == 0:
            return "Unknown", 0.0

        # Find best answer
        best_word = max(all_associations, key=all_associations.get)
        confidence = min(1.0, all_associations[best_word] / 10.0)

        return best_word, confidence

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the vocabulary."""
        type_counts = defaultdict(int)
        grounded_count = 0
        concept_linked_count = 0

        for word in self.vocabulary.values():
            type_counts[word.word_type.value] += 1
            if word.sensory_grounding is not None:
                grounded_count += 1
            if len(word.concept_ids) > 0:
                concept_linked_count += 1

        return {
            'vocabulary_size': len(self.vocabulary),
            'words_by_type': dict(type_counts),
            'sensory_grounded_words': grounded_count,
            'concept_linked_words': concept_linked_count,
            'total_words_processed': self.total_words_processed,
            'total_sentences_processed': self.total_sentences_processed,
            'unique_cooccurrences': len(self.cooccurrence),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the language system state."""
        vocab_data = {}
        for word, grounded in self.vocabulary.items():
            vocab_data[word] = {
                'word_type': grounded.word_type.value,
                'embedding': grounded.embedding.tolist(),
                'sensory_grounding': grounded.sensory_grounding.tolist() if grounded.sensory_grounding is not None else None,
                'concept_ids': grounded.concept_ids,
                'usage_count': grounded.usage_count,
                'confidence': grounded.confidence,
                'associations': grounded.associations,
            }

        return {
            'embedding_dim': self.embedding_dim,
            'sensory_dim': self.sensory_dim,
            'vocabulary': vocab_data,
            'concept_to_words': {k: list(v) for k, v in self.concept_to_words.items()},
            'stats': self.get_vocabulary_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'LanguageGrounding':
        """Deserialize a language system from saved state."""
        instance = cls(
            embedding_dim=data['embedding_dim'],
            sensory_dim=data.get('sensory_dim', 64),
        )

        for word, word_data in data.get('vocabulary', {}).items():
            grounded = GroundedWord(
                word=word,
                word_type=WordType(word_data['word_type']),
                embedding=np.array(word_data['embedding']),
                sensory_grounding=np.array(word_data['sensory_grounding']) if word_data['sensory_grounding'] else None,
                concept_ids=word_data['concept_ids'],
                usage_count=word_data['usage_count'],
                confidence=word_data['confidence'],
                associations=word_data.get('associations', {}),
            )
            instance.vocabulary[word] = grounded

        for concept_id, words in data.get('concept_to_words', {}).items():
            instance.concept_to_words[concept_id] = set(words)

        return instance


class TextCorpusLearner:
    """
    Learns language from a text corpus with optional sensory grounding.

    This enables ATLAS to acquire vocabulary and language understanding
    from large amounts of text.
    """

    def __init__(
        self,
        language_system: LanguageGrounding,
        batch_size: int = 100,
    ):
        """
        Initialize the corpus learner.

        Args:
            language_system: Language grounding system to train
            batch_size: Number of sentences per batch
        """
        self.language = language_system
        self.batch_size = batch_size
        self.sentences_processed = 0

    def learn_from_corpus(
        self,
        corpus: List[str],
        epochs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn from a corpus of text.

        Args:
            corpus: List of sentences/texts
            epochs: Number of passes through corpus
            verbose: Whether to print progress

        Returns:
            Training statistics
        """
        start_time = time.time()

        for epoch in range(epochs):
            # Shuffle corpus
            indices = np.random.permutation(len(corpus))

            for i in range(0, len(corpus), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                for idx in batch_indices:
                    self.language.learn_from_text(corpus[idx])
                    self.sentences_processed += 1

            if verbose:
                stats = self.language.get_vocabulary_stats()
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"vocab={stats['vocabulary_size']}, "
                      f"sentences={self.sentences_processed}")

        duration = time.time() - start_time

        return {
            'epochs': epochs,
            'sentences_processed': self.sentences_processed,
            'duration': duration,
            'final_vocab_size': len(self.language.vocabulary),
        }

    def learn_from_file(
        self,
        filepath: str,
        epochs: int = 1,
        max_sentences: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn from a text file.

        Args:
            filepath: Path to text file
            epochs: Number of passes
            max_sentences: Maximum sentences to read
            verbose: Whether to print progress

        Returns:
            Training statistics
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if max_sentences is not None:
            sentences = sentences[:max_sentences]

        return self.learn_from_corpus(sentences, epochs=epochs, verbose=verbose)
