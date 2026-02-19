#!/usr/bin/env python3
"""
Atlas Coherence Evaluator

Fixes the evaluation system to detect gibberish/code-noise responses.
Adds coherence checking to ensure Atlas is actually understanding, not just keyword matching.
"""

import re
import string
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class CoherenceIssue(Enum):
    """Types of coherence issues found in responses."""
    CODE_NOISE = "code_noise"
    NO_VERBS = "no_verbs"
    NO_SUBJECTS = "no_subjects"
    TOO_SHORT = "too_short"
    WORD_SALAD = "word_salad"
    OFF_TOPIC = "off_topic"


@dataclass
class CoherenceResult:
    """Result of coherence analysis."""
    score: float  # 0.0 to 1.0
    is_coherent: bool
    issues: List[CoherenceIssue]
    details: Dict[str, any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'score': self.score,
            'is_coherent': self.is_coherent,
            'issues': [issue.value for issue in self.issues],
            'details': self.details
        }


class CoherenceEvaluator:
    """
    Evaluates the coherence of Atlas responses.
    Detects gibberish, code noise, and word salad.
    """
    
    # Common code/programming terms that indicate gibberish
    CODE_NOISE_TERMS = {
        'engine', 'state_norm', 'debug', 'subsystem', 'stats', 'tokens',
        'av_system', 'share', 'processor', 'handler', 'callback', 'buffer',
        'config', 'params', 'args', 'kwargs', 'init', 'setup', 'teardown',
        'import', 'from', 'class', 'def', 'return', 'self', 'cls',
        'null', 'none', 'true', 'false', 'undefined', 'nan', 'inf',
        'array', 'vector', 'matrix', 'tensor', 'layer', 'node', 'edge',
        'forward', 'backward', 'gradient', 'loss', 'optimizer', 'epoch',
        'batch', 'train', 'eval', 'inference', 'model', 'checkpoint',
        'api', 'endpoint', 'request', 'response', 'header', 'payload',
        'json', 'xml', 'yaml', 'csv', 'db', 'query', 'schema',
        'thread', 'process', 'lock', 'mutex', 'semaphore', 'queue',
        'async', 'await', 'promise', 'future', 'callback', 'event',
        'widget', 'component', 'props', 'state', 'render', 'mount',
        'div', 'span', 'class_name', 'id_name', 'style', 'css',
        'printf', 'scanf', 'cout', 'cin', 'println', 'console',
        'log', 'debug', 'info', 'warn', 'error', 'trace', 'fatal',
        'bracket', 'paren', 'brace', 'angle', 'quote', 'backtick',
        'underscore', 'camelcase', 'snake_case', 'kebab-case',
    }
    
    # Common English verbs (for sentence structure checking)
    COMMON_VERBS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'done',
        'can', 'could', 'will', 'would', 'shall', 'should',
        'may', 'might', 'must', 'shall', 'should',
        'get', 'got', 'gotten', 'make', 'made', 'take', 'took', 'taken',
        'go', 'went', 'gone', 'come', 'came', 'see', 'saw', 'seen',
        'know', 'knew', 'known', 'think', 'thought', 'say', 'said',
        'tell', 'told', 'ask', 'asked', 'work', 'worked', 'feel', 'felt',
        'try', 'tried', 'leave', 'left', 'call', 'called', 'need', 'needed',
        'become', 'became', 'mean', 'meant', 'keep', 'kept', 'let', 'put',
        'bring', 'brought', 'begin', 'began', 'begun', 'seem', 'seemed',
        'help', 'helped', 'show', 'showed', 'shown', 'hear', 'heard',
        'play', 'played', 'run', 'ran', 'move', 'moved', 'live', 'lived',
        'believe', 'believed', 'bring', 'brought', 'happen', 'happened',
        'write', 'wrote', 'written', 'provide', 'provided', 'sit', 'sat',
        'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met',
        'include', 'included', 'continue', 'continued', 'set', 'learn',
        'learned', 'learnt', 'change', 'changed', 'lead', 'led', 'understand',
        'understood', 'watch', 'watched', 'follow', 'followed', 'stop',
        'stopped', 'create', 'created', 'speak', 'spoke', 'spoken', 'read',
        'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew',
        'grown', 'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer',
        'offered', 'remember', 'remembered', 'love', 'loved', 'consider',
        'considered', 'appear', 'appeared', 'buy', 'bought', 'wait', 'waited',
        'serve', 'served', 'die', 'died', 'send', 'sent', 'expect', 'expected',
        'build', 'built', 'stay', 'stayed', 'fall', 'fell', 'fallen', 'cut',
        'reach', 'reached', 'kill', 'killed', 'remain', 'remained', 'suggest',
        'suggested', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold',
        'require', 'required', 'report', 'reported', 'decide', 'decided',
        'pull', 'pulled', 'return', 'returned', 'explain', 'explained',
        'carry', 'carried', 'develop', 'developed', 'hope', 'hoped', 'drive',
        'drove', 'driven', 'break', 'broke', 'broken', 'receive', 'received',
        'agree', 'agreed', 'support', 'supported', 'remove', 'removed',
        'return', 'returned', 'describe', 'described', 'create', 'created',
        'add', 'added', 'take', 'took', 'taken', 'provide', 'provided',
        'apply', 'applied', 'use', 'used', 'find', 'found', 'give', 'gave',
        'given', 'tell', 'told', 'become', 'became', 'leave', 'left',
        'feel', 'felt', 'put', 'mean', 'meant', 'keep', 'kept', 'let',
        'begin', 'began', 'begun', 'seem', 'seemed', 'help', 'helped',
        'show', 'showed', 'shown', 'hear', 'heard', 'play', 'played',
        'run', 'ran', 'move', 'moved', 'live', 'lived', 'believe',
        'believed', 'bring', 'brought', 'happen', 'happened', 'write',
        'wrote', 'written', 'provide', 'provided', 'sit', 'sat', 'stand',
        'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include',
        'included', 'continue', 'continued', 'set', 'learn', 'learned',
        'learnt', 'change', 'changed', 'lead', 'led', 'understand',
        'understood', 'watch', 'watched', 'follow', 'followed', 'stop',
        'stopped', 'create', 'created', 'speak', 'spoke', 'spoken',
        'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent',
        'grow', 'grew', 'grown', 'open', 'opened', 'walk', 'walked',
        'win', 'won', 'offer', 'offered', 'remember', 'remembered',
        'love', 'loved', 'consider', 'considered', 'appear', 'appeared',
        'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die',
        'died', 'send', 'sent', 'expect', 'expected', 'build', 'built',
        'stay', 'stayed', 'fall', 'fell', 'fallen', 'cut', 'reach',
        'reached', 'kill', 'killed', 'remain', 'remained', 'suggest',
        'suggested', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold',
        'require', 'required', 'report', 'reported', 'decide', 'decided',
        'pull', 'pulled', 'explain', 'explained', 'carry', 'carried',
        'develop', 'developed', 'hope', 'hoped', 'drive', 'drove',
        'driven', 'break', 'broke', 'broken', 'receive', 'received',
        'agree', 'agreed', 'support', 'supported', 'remove', 'removed',
        'return', 'returned', 'describe', 'described', 'work', 'worked',
        'call', 'called', 'try', 'tried', 'ask', 'asked', 'need',
        'needed', 'feel', 'felt', 'become', 'became', 'leave', 'left',
        'put', 'mean', 'meant', 'keep', 'kept', 'let', 'begin', 'began',
        'begun', 'help', 'helped', 'show', 'showed', 'hear', 'heard',
        'play', 'played', 'run', 'ran', 'move', 'moved', 'live', 'lived',
        'believe', 'believed', 'bring', 'brought', 'happen', 'happened',
        'write', 'wrote', 'provide', 'provided', 'sit', 'sat', 'stand',
        'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include',
        'included', 'continue', 'continued', 'set', 'learn', 'learned',
        'change', 'changed', 'lead', 'led', 'understand', 'understood',
        'watch', 'watched', 'follow', 'followed', 'stop', 'stopped',
        'create', 'created', 'speak', 'spoke', 'read', 'allow', 'allowed',
        'add', 'added', 'spend', 'spent', 'grow', 'grew', 'open', 'opened',
        'walk', 'walked', 'win', 'won', 'offer', 'offered', 'remember',
        'remembered', 'love', 'loved', 'consider', 'considered', 'appear',
        'appeared', 'buy', 'bought', 'wait', 'waited', 'serve', 'served',
        'die', 'died', 'send', 'sent', 'expect', 'expected', 'build',
        'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'reach',
        'reached', 'kill', 'killed', 'remain', 'remained', 'suggest',
        'suggested', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold',
        'require', 'required', 'report', 'reported', 'decide', 'decided',
        'pull', 'pulled', 'explain', 'explained', 'carry', 'carried',
        'develop', 'developed', 'hope', 'hoped', 'drive', 'drove',
        'break', 'broke', 'receive', 'received', 'agree', 'agreed',
        'support', 'supported', 'remove', 'removed', 'return', 'returned',
        'describe', 'described', 'equals', 'equal', 'sum', 'add', 'subtract',
        'multiply', 'divide', 'calculate', 'compute', 'solve', 'find',
        'determine', 'result', 'produce', 'generate', 'form', 'create',
        'make', 'construct', 'derive', 'obtain', 'get', 'gives', 'yields',
        'produces', 'results', 'forms', 'creates', 'makes', 'constructs',
        'derives', 'obtains', 'gets', 'represents', 'indicates', 'shows',
        'demonstrates', 'proves', 'follows', 'comes', 'arises', 'emerges',
        'appears', 'seems', 'looks', 'sounds', 'acts', 'behaves', 'functions',
        'operates', 'works', 'runs', 'executes', 'performs', 'completes',
        'finishes', 'ends', 'terminates', 'starts', 'begins', 'initiates',
        'launches', 'triggers', 'causes', 'induces', 'produces', 'generates',
        'creates', 'forms', 'makes', 'builds', 'constructs', 'assembles',
        'composes', 'comprises', 'consists', 'contains', 'includes',
        'incorporates', 'involves', 'entails', 'requires', 'needs',
        'demands', 'necessitates', 'depends', 'relies', 'counts',
        'hinges', 'rests', 'bases', 'founds', 'establishes', 'sets',
        'places', 'puts', 'lays', 'positions', 'locates', 'situates',
        'stands', 'exists', 'resides', 'dwells', 'lives', 'stays',
        'remains', 'continues', 'persists', 'endures', 'lasts',
        'survives', 'outlasts', 'outlives', 'precedes', 'follows',
        'succeeds', 'replaces', 'substitutes', 'alternates', 'rotates',
        'cycles', 'repeats', 'recurs', 'returns', 'reverts', 'regresses',
        'progresses', 'advances', 'proceeds', 'moves', 'goes', 'travels',
        'journeys', 'wanders', 'roams', 'ranges', 'extends', 'stretches',
        'spreads', 'expands', 'grows', 'increases', 'decreases', 'reduces',
        'diminishes', 'shrinks', 'contracts', 'compresses', 'condenses',
        'concentrates', 'focuses', 'centers', 'converges', 'diverges',
        'deviates', 'varies', 'changes', 'differs', 'contrasts', 'compares',
        'matches', 'parallels', 'mirrors', 'reflects', 'echoes', 'repeats',
        'copies', 'duplicates', 'replicates', 'reproduces', 'clones',
        'imitates', 'mimics', 'emulates', 'simulates', 'models', 'patterns',
    }
    
    # Common pronouns and subjects
    SUBJECT_INDICATORS = {
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'this', 'that', 'these', 'those', 'the', 'a', 'an',
        'fibonacci', 'sequence', 'number', 'pattern', 'each', 'every',
        'all', 'some', 'many', 'most', 'more', 'one', 'two', 'three',
        'first', 'second', 'third', 'next', 'previous', 'following',
        'preceding', 'prior', 'former', 'latter', 'last', 'final',
        'mathematics', 'math', 'science', 'programming', 'code',
        'algorithm', 'function', 'variable', 'equation', 'formula',
        'principle', 'concept', 'idea', 'theory', 'law', 'rule',
        'method', 'technique', 'approach', 'way', 'manner', 'style',
        'form', 'shape', 'structure', 'organization', 'arrangement',
        'order', 'sequence', 'series', 'chain', 'string', 'line',
        'row', 'column', 'array', 'list', 'set', 'group', 'collection',
        'system', 'network', 'web', 'pattern', 'design', 'plan',
        'scheme', 'strategy', 'tactic', 'procedure', 'process',
        'operation', 'action', 'activity', 'task', 'job', 'work',
        'role', 'function', 'purpose', 'goal', 'aim', 'objective',
        'target', 'end', 'result', 'outcome', 'effect', 'consequence',
        'product', 'output', 'yield', 'return', 'value', 'worth',
        'importance', 'significance', 'meaning', 'sense', 'reason',
        'cause', 'source', 'origin', 'root', 'basis', 'foundation',
        'ground', 'reason', 'motive', 'purpose', 'intention', 'aim',
        'goal', 'objective', 'target', 'end', 'point', 'place',
        'position', 'location', 'site', 'spot', 'area', 'region',
        'zone', 'sector', 'section', 'part', 'portion', 'piece',
        'segment', 'component', 'element', 'factor', 'aspect',
        'feature', 'characteristic', 'quality', 'property', 'attribute',
        'trait', 'nature', 'essence', 'core', 'heart', 'soul',
        'spirit', 'mind', 'brain', 'thought', 'idea', 'notion',
        'concept', 'conception', 'perception', 'view', 'opinion',
        'belief', 'conviction', 'faith', 'trust', 'confidence',
        'certainty', 'sureness', 'assurance', 'guarantee', 'warranty',
        'promise', 'pledge', 'commitment', 'dedication', 'devotion',
        'loyalty', 'allegiance', 'fidelity', 'faithfulness', 'honesty',
        'integrity', 'honor', 'dignity', 'respect', 'esteem', 'regard',
        'admiration', 'appreciation', 'gratitude', 'thanks', 'praise',
        'compliment', 'flattery', 'admiration', 'wonder', 'awe',
        'amazement', 'astonishment', 'surprise', 'shock', 'startle',
    }
    
    # Topic-specific keywords for relevance checking
    TOPIC_KEYWORDS = {
        'Mathematics': {'number', 'sum', 'add', 'subtract', 'multiply', 'divide',
                       'equation', 'formula', 'calculate', 'compute', 'math',
                       'fibonacci', 'sequence', 'pattern', 'prime', 'geometry',
                       'algebra', 'calculus', 'derivative', 'integral', 'function',
                       'variable', 'constant', 'value', 'result', 'solution',
                       'problem', 'answer', 'proof', 'theorem', 'lemma',
                       'axiom', 'postulate', 'definition', 'property', 'rule'},
        'Science': {'experiment', 'observe', 'measure', 'test', 'hypothesis',
                   'theory', 'law', 'evidence', 'data', 'result', 'conclusion',
                   'photosynthesis', 'cell', 'organism', 'species', 'evolution',
                   'natural', 'selection', 'dna', 'gene', 'genetic', 'heredity',
                   'atom', 'molecule', 'compound', 'element', 'chemical',
                   'reaction', 'energy', 'force', 'motion', 'gravity',
                   'planet', 'star', 'galaxy', 'universe', 'earth'},
        'Programming': {'code', 'program', 'software', 'computer', 'algorithm',
                       'function', 'variable', 'class', 'object', 'method',
                       'loop', 'condition', 'if', 'else', 'while', 'for',
                       'return', 'call', 'invoke', 'execute', 'run',
                       'compile', 'debug', 'error', 'bug', 'fix',
                       'syntax', 'semantic', 'type', 'data', 'structure',
                       'array', 'list', 'string', 'integer', 'boolean'},
        'Language': {'word', 'sentence', 'grammar', 'syntax', 'semantic',
                    'meaning', 'definition', 'vocabulary', 'spelling',
                    'pronunciation', 'speak', 'write', 'read', 'communicate',
                    'language', 'linguistic', 'etymology', 'origin', 'root',
                    'prefix', 'suffix', 'phrase', 'clause', 'paragraph',
                    'noun', 'verb', 'adjective', 'adverb', 'pronoun'},
        'Logic': {'reason', 'argument', 'premise', 'conclusion', 'valid',
                 'invalid', 'true', 'false', 'deduction', 'induction',
                 'syllogism', 'fallacy', 'bias', 'evidence', 'proof',
                 'logic', 'logical', 'rational', 'irrational', 'critical',
                 'think', 'analysis', 'evaluate', 'assess', 'judge'},
    }
    
    def __init__(self, coherence_threshold: float = 0.5):
        self.coherence_threshold = coherence_threshold
    
    def analyze(self, response: str, topic: str = None) -> CoherenceResult:
        """
        Analyze the coherence of a response.
        
        Returns a CoherenceResult with score and identified issues.
        """
        issues = []
        details = {}
        
        # Clean and tokenize
        words = self._tokenize(response)
        details['word_count'] = len(words)
        
        if len(words) < 3:
            issues.append(CoherenceIssue.TOO_SHORT)
            return CoherenceResult(
                score=0.0,
                is_coherent=False,
                issues=issues,
                details=details
            )
        
        # Check for code noise
        code_noise_ratio = self._check_code_noise(words)
        details['code_noise_ratio'] = code_noise_ratio
        if code_noise_ratio > 0.3:  # More than 30% code terms
            issues.append(CoherenceIssue.CODE_NOISE)
        
        # Check for sentence structure (verbs and subjects)
        has_verbs = self._has_verbs(words)
        details['has_verbs'] = has_verbs
        if not has_verbs:
            issues.append(CoherenceIssue.NO_VERBS)
        
        has_subjects = self._has_subjects(words)
        details['has_subjects'] = has_subjects
        if not has_subjects:
            issues.append(CoherenceIssue.NO_SUBJECTS)
        
        # Check for word salad (random word combinations)
        word_salad_score = self._check_word_salad(words)
        details['word_salad_score'] = word_salad_score
        if word_salad_score > 0.7:
            issues.append(CoherenceIssue.WORD_SALAD)
        
        # Check topic relevance
        if topic:
            relevance_score = self._check_topic_relevance(words, topic)
            details['relevance_score'] = relevance_score
            if relevance_score < 0.2:
                issues.append(CoherenceIssue.OFF_TOPIC)
        else:
            relevance_score = 0.5  # Neutral if no topic specified
            details['relevance_score'] = relevance_score
        
        # Calculate overall coherence score
        score = self._calculate_coherence_score(
            code_noise_ratio, has_verbs, has_subjects,
            word_salad_score, relevance_score, len(words)
        )
        details['final_score'] = score
        
        is_coherent = score >= self.coherence_threshold and len(issues) == 0
        
        return CoherenceResult(
            score=score,
            is_coherent=is_coherent,
            issues=issues,
            details=details
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation except brackets which might indicate code
        text = text.lower()
        # Keep brackets as separate tokens to detect code patterns
        text = re.sub(r'([\[\]{}()<>])', r' \1 ', text)
        words = text.split()
        # Clean each word
        cleaned = []
        for word in words:
            # Remove punctuation but keep alphanumeric and brackets
            clean = ''.join(c for c in word if c.isalnum() or c in '[]{}()<>_')
            if clean:
                cleaned.append(clean)
        return cleaned
    
    def _check_code_noise(self, words: List[str]) -> float:
        """Check what percentage of words are code/programming terms."""
        if not words:
            return 0.0
        code_words = sum(1 for w in words if w in self.CODE_NOISE_TERMS)
        return code_words / len(words)
    
    def _has_verbs(self, words: List[str]) -> bool:
        """Check if response contains verbs."""
        return any(w in self.COMMON_VERBS for w in words)
    
    def _has_subjects(self, words: List[str]) -> bool:
        """Check if response contains subjects or subject indicators."""
        return any(w in self.SUBJECT_INDICATORS for w in words)
    
    def _check_word_salad(self, words: List[str]) -> float:
        """
        Check for word salad - random combinations without meaning.
        Returns a score where higher = more like word salad.
        """
        if len(words) < 2:
            return 0.0
        
        # Check for unusual character patterns (like code brackets)
        unusual_patterns = 0
        for word in words:
            # Count words with brackets, underscores, or mixed case
            if '[' in word or ']' in word or '_' in word:
                unusual_patterns += 1
            # Count words that are all uppercase (likely constants/variables)
            if word.isupper() and len(word) > 1:
                unusual_patterns += 1
        
        unusual_ratio = unusual_patterns / len(words) if words else 0
        return unusual_ratio
    
    def _check_topic_relevance(self, words: List[str], topic: str) -> float:
        """Check what percentage of words are relevant to the topic."""
        if not words or topic not in self.TOPIC_KEYWORDS:
            return 0.5
        
        topic_words = self.TOPIC_KEYWORDS[topic]
        # Also include general academic words as relevant
        general_words = {'the', 'is', 'are', 'and', 'or', 'but', 'because',
                        'therefore', 'thus', 'however', 'moreover', 'furthermore',
                        'for', 'example', 'instance', 'such', 'as', 'like',
                        'when', 'where', 'why', 'how', 'what', 'who', 'which'}
        
        relevant = sum(1 for w in words if w in topic_words or w in general_words)
        return relevant / len(words)
    
    def _calculate_coherence_score(self, code_noise_ratio: float,
                                    has_verbs: bool, has_subjects: bool,
                                    word_salad_score: float,
                                    relevance_score: float,
                                    word_count: int) -> float:
        """
        Calculate overall coherence score based on multiple factors.
        """
        score = 1.0
        
        # Penalize code noise heavily
        score -= code_noise_ratio * 1.5
        
        # Penalize missing verbs
        if not has_verbs:
            score -= 0.3
        
        # Penalize missing subjects
        if not has_subjects:
            score -= 0.2
        
        # Penalize word salad
        score -= word_salad_score * 0.8
        
        # Boost for relevance
        score += relevance_score * 0.3
        
        # Slight boost for appropriate length
        if 10 <= word_count <= 100:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_feedback(self, result: CoherenceResult) -> str:
        """Generate human-readable feedback from coherence result."""
        if result.is_coherent:
            return f"Response is coherent (score: {result.score:.2f})."
        
        feedback_parts = []
        for issue in result.issues:
            if issue == CoherenceIssue.CODE_NOISE:
                feedback_parts.append("contains programming/code terms")
            elif issue == CoherenceIssue.NO_VERBS:
                feedback_parts.append("lacks verbs (not a complete sentence)")
            elif issue == CoherenceIssue.NO_SUBJECTS:
                feedback_parts.append("lacks clear subject")
            elif issue == CoherenceIssue.TOO_SHORT:
                feedback_parts.append("too short to evaluate")
            elif issue == CoherenceIssue.WORD_SALAD:
                feedback_parts.append("appears to be random word combinations")
            elif issue == CoherenceIssue.OFF_TOPIC:
                feedback_parts.append("not relevant to the topic")
        
        if feedback_parts:
            return f"Response is incoherent (score: {result.score:.2f}): {', '.join(feedback_parts)}."
        else:
            return f"Response is incoherent (score: {result.score:.2f})."


# Global evaluator instance
_coherence_evaluator = None


def get_coherence_evaluator(threshold: float = 0.5) -> CoherenceEvaluator:
    """Get or create the global coherence evaluator."""
    global _coherence_evaluator
    if _coherence_evaluator is None:
        _coherence_evaluator = CoherenceEvaluator(threshold)
    return _coherence_evaluator


def evaluate_coherence(response: str, topic: str = None, threshold: float = 0.5) -> CoherenceResult:
    """
    Convenience function to evaluate coherence of a response.
    
    Args:
        response: The response text to evaluate
        topic: Optional topic for relevance checking
        threshold: Minimum coherence score to be considered coherent
        
    Returns:
        CoherenceResult with score and issues
    """
    evaluator = get_coherence_evaluator(threshold)
    return evaluator.analyze(response, topic)


if __name__ == "__main__":
    # Test the coherence evaluator
    test_cases = [
        ("The Fibonacci sequence is where each number is the sum of the two preceding numbers.", "Mathematics"),
        ("engine [ state_norm share debug_av_system subsystem_stats ] tokens", "Mathematics"),
        ("sum preceding previous add sequence", "Mathematics"),
        ("The pattern works by adding the previous two numbers together, so each number equals the sum of what came before.", "Mathematics"),
        ("state_norm debug_av_system processor handler callback", "Mathematics"),
        ("Photosynthesis requires sunlight, water, and carbon dioxide to produce glucose and oxygen.", "Science"),
        ("config params args kwargs init setup teardown", "Science"),
    ]
    
    evaluator = CoherenceEvaluator()
    print("=" * 70)
    print("Coherence Evaluator Tests")
    print("=" * 70)
    
    for response, topic in test_cases:
        result = evaluator.analyze(response, topic)
        print(f"\nTopic: {topic}")
        print(f"Response: {response[:60]}...")
        print(f"Coherent: {result.is_coherent} (score: {result.score:.2f})")
        print(f"Issues: {[i.value for i in result.issues]}")
        print(f"Feedback: {evaluator.get_feedback(result)}")
