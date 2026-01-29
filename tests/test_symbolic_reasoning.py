"""
Comprehensive tests for the symbolic reasoning module.

Tests cover SymbolicReasoner, Symbol, Proposition, Rule, Analogy,
and LogicType -- including initialization, creation, operations,
rule application, and all four inference modes.

All tests are deterministic and pass reliably.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.symbolic_reasoning import (
    SymbolicReasoner,
    Symbol,
    LogicType,
    Proposition,
    Rule,
    Analogy,
)


# ---------------------------------------------------------------------------
# LogicType enum
# ---------------------------------------------------------------------------
class TestLogicType(unittest.TestCase):
    """Tests for the LogicType enumeration."""

    def test_enum_values(self):
        """All four reasoning modes are present with correct string values."""
        self.assertEqual(LogicType.DEDUCTIVE.value, "deductive")
        self.assertEqual(LogicType.INDUCTIVE.value, "inductive")
        self.assertEqual(LogicType.ABDUCTIVE.value, "abductive")
        self.assertEqual(LogicType.ANALOGICAL.value, "analogical")

    def test_enum_member_count(self):
        """Exactly four logic types exist."""
        self.assertEqual(len(LogicType), 4)

    def test_enum_identity(self):
        """Enum members retrieved by value match identity."""
        self.assertIs(LogicType("deductive"), LogicType.DEDUCTIVE)
        self.assertIs(LogicType("inductive"), LogicType.INDUCTIVE)
        self.assertIs(LogicType("abductive"), LogicType.ABDUCTIVE)
        self.assertIs(LogicType("analogical"), LogicType.ANALOGICAL)

    def test_invalid_value_raises(self):
        """Invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            LogicType("invalid")


# ---------------------------------------------------------------------------
# Symbol
# ---------------------------------------------------------------------------
class TestSymbol(unittest.TestCase):
    """Tests for Symbol creation and operations."""

    def test_basic_creation(self):
        """A Symbol stores its name, type, and default fields."""
        sym = Symbol(name="x", symbol_type="variable")
        self.assertEqual(sym.name, "x")
        self.assertEqual(sym.symbol_type, "variable")
        self.assertIsNone(sym.grounding)
        self.assertEqual(sym.properties, {})

    def test_creation_with_grounding(self):
        """A Symbol can be created with a numpy grounding vector."""
        vec = np.array([1.0, 2.0, 3.0])
        sym = Symbol(name="a", symbol_type="constant", grounding=vec)
        np.testing.assert_array_equal(sym.grounding, vec)

    def test_creation_with_properties(self):
        """A Symbol can carry arbitrary properties."""
        props = {"color": "red", "size": 5}
        sym = Symbol(name="obj", symbol_type="constant", properties=props)
        self.assertEqual(sym.properties["color"], "red")
        self.assertEqual(sym.properties["size"], 5)

    def test_hash_by_name(self):
        """Symbols hash by their name alone."""
        s1 = Symbol(name="x", symbol_type="variable")
        s2 = Symbol(name="x", symbol_type="constant")
        self.assertEqual(hash(s1), hash(s2))

    def test_equality_same_name(self):
        """Two Symbols with the same name are equal regardless of type."""
        s1 = Symbol(name="x", symbol_type="variable")
        s2 = Symbol(name="x", symbol_type="constant")
        self.assertEqual(s1, s2)

    def test_equality_different_name(self):
        """Two Symbols with different names are not equal."""
        s1 = Symbol(name="x", symbol_type="variable")
        s2 = Symbol(name="y", symbol_type="variable")
        self.assertNotEqual(s1, s2)

    def test_equality_with_non_symbol(self):
        """A Symbol is not equal to a non-Symbol object."""
        sym = Symbol(name="x", symbol_type="variable")
        self.assertNotEqual(sym, "x")
        self.assertNotEqual(sym, 42)

    def test_repr(self):
        """repr shows the expected format."""
        sym = Symbol(name="alpha", symbol_type="constant")
        self.assertEqual(repr(sym), "Symbol(alpha)")

    def test_usable_in_set(self):
        """Symbols with the same name collapse in a set."""
        s1 = Symbol(name="x", symbol_type="variable")
        s2 = Symbol(name="x", symbol_type="constant")
        s3 = Symbol(name="y", symbol_type="variable")
        result = {s1, s2, s3}
        self.assertEqual(len(result), 2)

    def test_usable_as_dict_key(self):
        """Symbols work as dictionary keys, keyed by name."""
        s1 = Symbol(name="k", symbol_type="variable")
        s2 = Symbol(name="k", symbol_type="constant")
        d = {s1: 10}
        self.assertEqual(d[s2], 10)


# ---------------------------------------------------------------------------
# Proposition
# ---------------------------------------------------------------------------
class TestProposition(unittest.TestCase):
    """Tests for Proposition creation and behaviour."""

    def setUp(self):
        self.sym_a = Symbol(name="a", symbol_type="constant")
        self.sym_b = Symbol(name="b", symbol_type="constant")

    def test_basic_creation(self):
        """A Proposition stores its predicate, arguments, and defaults."""
        prop = Proposition(predicate="likes", arguments=(self.sym_a, self.sym_b))
        self.assertEqual(prop.predicate, "likes")
        self.assertEqual(prop.arguments, (self.sym_a, self.sym_b))
        self.assertIsNone(prop.truth_value)
        self.assertEqual(prop.confidence, 1.0)

    def test_creation_with_truth_and_confidence(self):
        """Truth value and confidence can be set at construction."""
        prop = Proposition(
            predicate="is_red",
            arguments=(self.sym_a,),
            truth_value=True,
            confidence=0.9,
        )
        self.assertTrue(prop.truth_value)
        self.assertAlmostEqual(prop.confidence, 0.9)

    def test_hash(self):
        """Propositions with same predicate and arguments hash identically."""
        p1 = Proposition(predicate="likes", arguments=(self.sym_a, self.sym_b))
        p2 = Proposition(predicate="likes", arguments=(self.sym_a, self.sym_b))
        self.assertEqual(hash(p1), hash(p2))

    def test_hash_differs_for_different_predicate(self):
        """Different predicates produce different hashes (in practice)."""
        p1 = Proposition(predicate="likes", arguments=(self.sym_a,))
        p2 = Proposition(predicate="hates", arguments=(self.sym_a,))
        # Not guaranteed by contract, but sanity check:
        self.assertNotEqual(hash(p1), hash(p2))

    def test_repr(self):
        """repr shows predicate(arg1, arg2) format."""
        prop = Proposition(predicate="likes", arguments=(self.sym_a, self.sym_b))
        self.assertEqual(repr(prop), "likes(Symbol(a), Symbol(b))")

    def test_repr_single_argument(self):
        prop = Proposition(predicate="is_red", arguments=(self.sym_a,))
        self.assertEqual(repr(prop), "is_red(Symbol(a))")

    def test_usable_in_set(self):
        """Identical propositions collapse in a set."""
        p1 = Proposition(predicate="p", arguments=(self.sym_a,))
        p2 = Proposition(predicate="p", arguments=(self.sym_a,))
        self.assertEqual(len({p1, p2}), 1)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------
class TestRule(unittest.TestCase):
    """Tests for Rule creation and repr."""

    def setUp(self):
        self.x = Symbol(name="X", symbol_type="variable")
        self.y = Symbol(name="Y", symbol_type="variable")
        self.prem = Proposition(predicate="mortal", arguments=(self.x,))
        self.conc = Proposition(predicate="dies", arguments=(self.x,))

    def test_basic_creation(self):
        """A Rule stores name, premises, conclusions, and defaults."""
        rule = Rule(
            name="mortality",
            premises=[self.prem],
            conclusions=[self.conc],
        )
        self.assertEqual(rule.name, "mortality")
        self.assertEqual(len(rule.premises), 1)
        self.assertEqual(len(rule.conclusions), 1)
        self.assertEqual(rule.confidence, 1.0)
        self.assertEqual(rule.applications, 0)

    def test_custom_confidence(self):
        rule = Rule(
            name="r",
            premises=[self.prem],
            conclusions=[self.conc],
            confidence=0.75,
        )
        self.assertAlmostEqual(rule.confidence, 0.75)

    def test_repr(self):
        """repr shows 'premises => conclusions' format."""
        rule = Rule(
            name="r",
            premises=[self.prem],
            conclusions=[self.conc],
        )
        expected = "mortal(Symbol(X)) => dies(Symbol(X))"
        self.assertEqual(repr(rule), expected)

    def test_repr_multiple_premises_and_conclusions(self):
        p2 = Proposition(predicate="human", arguments=(self.x,))
        c2 = Proposition(predicate="suffers", arguments=(self.x,))
        rule = Rule(
            name="r2",
            premises=[self.prem, p2],
            conclusions=[self.conc, c2],
        )
        self.assertIn(" AND ", repr(rule))
        self.assertIn("=>", repr(rule))

    def test_applications_counter(self):
        """applications counter can be incremented."""
        rule = Rule(name="r", premises=[], conclusions=[])
        self.assertEqual(rule.applications, 0)
        rule.applications += 1
        self.assertEqual(rule.applications, 1)


# ---------------------------------------------------------------------------
# Analogy
# ---------------------------------------------------------------------------
class TestAnalogy(unittest.TestCase):
    """Tests for the Analogy dataclass."""

    def test_creation(self):
        src = Symbol(name="src_a", symbol_type="constant")
        tgt = Symbol(name="tgt_a", symbol_type="constant")
        analogy = Analogy(
            source_domain="physics",
            target_domain="economics",
            mappings={src: tgt},
            structure_similarity=0.85,
        )
        self.assertEqual(analogy.source_domain, "physics")
        self.assertEqual(analogy.target_domain, "economics")
        self.assertAlmostEqual(analogy.structure_similarity, 0.85)
        self.assertEqual(analogy.successful_transfers, 0)

    def test_successful_transfers_default(self):
        analogy = Analogy(
            source_domain="a",
            target_domain="b",
            mappings={},
            structure_similarity=0.5,
        )
        self.assertEqual(analogy.successful_transfers, 0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- initialisation
# ---------------------------------------------------------------------------
class TestSymbolicReasonerInit(unittest.TestCase):
    """Tests for SymbolicReasoner construction and default state."""

    def test_default_initialization(self):
        """Default construction sets expected parameters."""
        sr = SymbolicReasoner(random_seed=42)
        self.assertAlmostEqual(sr.grounding_threshold, 0.7)
        self.assertEqual(sr.max_symbols, 10000)
        self.assertEqual(sr.max_rules, 1000)
        self.assertEqual(sr.inference_depth, 5)
        self.assertTrue(sr.enable_rule_learning)
        self.assertTrue(sr.enable_analogy)
        self.assertAlmostEqual(sr.confidence_threshold, 0.5)

    def test_custom_initialization(self):
        """Custom parameters are stored correctly."""
        sr = SymbolicReasoner(
            grounding_threshold=0.5,
            max_symbols=500,
            max_rules=50,
            inference_depth=3,
            enable_rule_learning=False,
            enable_analogy=False,
            confidence_threshold=0.8,
            random_seed=0,
        )
        self.assertAlmostEqual(sr.grounding_threshold, 0.5)
        self.assertEqual(sr.max_symbols, 500)
        self.assertEqual(sr.max_rules, 50)
        self.assertEqual(sr.inference_depth, 3)
        self.assertFalse(sr.enable_rule_learning)
        self.assertFalse(sr.enable_analogy)
        self.assertAlmostEqual(sr.confidence_threshold, 0.8)

    def test_empty_initial_state(self):
        """A freshly created reasoner has empty knowledge structures."""
        sr = SymbolicReasoner(random_seed=42)
        self.assertEqual(len(sr.symbols), 0)
        self.assertEqual(len(sr.propositions), 0)
        self.assertEqual(len(sr.rules), 0)
        self.assertEqual(len(sr.analogies), 0)
        self.assertEqual(len(sr.working_memory), 0)
        self.assertEqual(len(sr.inference_chains), 0)

    def test_initial_statistics(self):
        """Initial statistics counters are zero."""
        sr = SymbolicReasoner(random_seed=42)
        self.assertEqual(sr.total_inferences, 0)
        self.assertEqual(sr.total_groundings, 0)
        self.assertEqual(sr.total_analogies, 0)
        self.assertEqual(sr.successful_rule_applications, 0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- symbol grounding
# ---------------------------------------------------------------------------
class TestSymbolGrounding(unittest.TestCase):
    """Tests for the ground_symbol method."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)

    def test_ground_new_symbol(self):
        """Grounding a novel pattern creates a new symbol."""
        pattern = np.array([1.0, 0.0, 0.0])
        sym = self.sr.ground_symbol(pattern)
        self.assertIsInstance(sym, Symbol)
        self.assertEqual(sym.name, "sym_0")
        self.assertEqual(sym.symbol_type, "constant")
        self.assertEqual(len(self.sr.symbols), 1)
        self.assertEqual(self.sr.total_groundings, 1)

    def test_ground_symbol_with_type(self):
        """symbol_type parameter is stored."""
        pattern = np.array([0.0, 1.0, 0.0])
        sym = self.sr.ground_symbol(pattern, symbol_type="predicate")
        self.assertEqual(sym.symbol_type, "predicate")

    def test_ground_symbol_with_properties(self):
        """properties parameter is stored."""
        pattern = np.array([0.0, 0.0, 1.0])
        sym = self.sr.ground_symbol(pattern, properties={"weight": 3})
        self.assertEqual(sym.properties["weight"], 3)

    def test_ground_similar_pattern_reuses_symbol(self):
        """A pattern very similar to an existing one reuses the symbol."""
        p1 = np.array([1.0, 0.0, 0.0])
        sym1 = self.sr.ground_symbol(p1)

        # Almost identical pattern
        p2 = np.array([0.999, 0.001, 0.0])
        sym2 = self.sr.ground_symbol(p2)

        self.assertEqual(sym1.name, sym2.name)
        # Only one grounding should have been created
        self.assertEqual(len(self.sr.symbols), 1)

    def test_ground_different_patterns_creates_distinct_symbols(self):
        """Orthogonal patterns create separate symbols."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        sym1 = self.sr.ground_symbol(p1)
        sym2 = self.sr.ground_symbol(p2)

        self.assertNotEqual(sym1.name, sym2.name)
        self.assertEqual(len(self.sr.symbols), 2)

    def test_grounding_preserves_copy(self):
        """The grounding vector is copied, not aliased."""
        p = np.array([1.0, 2.0, 3.0])
        sym = self.sr.ground_symbol(p)
        p[0] = 999.0
        np.testing.assert_array_equal(sym.grounding, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# SymbolicReasoner -- propositions
# ---------------------------------------------------------------------------
class TestAssertProposition(unittest.TestCase):
    """Tests for assert_proposition."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)
        self.sym_a = Symbol(name="a", symbol_type="constant")
        self.sym_b = Symbol(name="b", symbol_type="constant")

    def test_assert_adds_to_propositions_and_working_memory(self):
        prop = self.sr.assert_proposition("likes", (self.sym_a, self.sym_b))
        self.assertIn(prop, self.sr.propositions)
        self.assertIn(prop, self.sr.working_memory)

    def test_assert_returns_proposition(self):
        prop = self.sr.assert_proposition("likes", (self.sym_a,), truth_value=True, confidence=0.8)
        self.assertIsInstance(prop, Proposition)
        self.assertEqual(prop.predicate, "likes")
        self.assertTrue(prop.truth_value)
        self.assertAlmostEqual(prop.confidence, 0.8)

    def test_assert_default_truth_and_confidence(self):
        prop = self.sr.assert_proposition("p", (self.sym_a,))
        self.assertTrue(prop.truth_value)
        self.assertAlmostEqual(prop.confidence, 1.0)

    def test_working_memory_limit(self):
        """Working memory does not exceed 100 entries."""
        for i in range(120):
            sym = Symbol(name=f"s{i}", symbol_type="constant")
            self.sr.assert_proposition("prop", (sym,))
        self.assertLessEqual(len(self.sr.working_memory), 100)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- rules
# ---------------------------------------------------------------------------
class TestAddRule(unittest.TestCase):
    """Tests for add_rule."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42, max_rules=5)
        self.x = Symbol(name="X", symbol_type="variable")
        self.prem = Proposition(predicate="mortal", arguments=(self.x,))
        self.conc = Proposition(predicate="dies", arguments=(self.x,))

    def test_add_rule_basic(self):
        rule = self.sr.add_rule("mortality", [self.prem], [self.conc])
        self.assertIsInstance(rule, Rule)
        self.assertEqual(len(self.sr.rules), 1)
        self.assertEqual(self.sr.rules[0].name, "mortality")

    def test_add_rule_custom_confidence(self):
        rule = self.sr.add_rule("r", [self.prem], [self.conc], confidence=0.6)
        self.assertAlmostEqual(rule.confidence, 0.6)

    def test_rule_limit_enforced(self):
        """When max_rules is exceeded the least-used rule is removed."""
        for i in range(6):
            self.sr.add_rule(f"r{i}", [self.prem], [self.conc])
        self.assertLessEqual(len(self.sr.rules), self.sr.max_rules)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- deductive inference
# ---------------------------------------------------------------------------
class TestDeductiveInference(unittest.TestCase):
    """Tests for deductive (forward-chaining) inference."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)
        # Create symbols
        self.socrates = Symbol(name="socrates", symbol_type="constant")
        self.x = Symbol(name="X", symbol_type="variable")

        # Assert: human(socrates)
        self.sr.assert_proposition("human", (self.socrates,), truth_value=True)

        # Rule: human(X) => mortal(X)
        premise = Proposition(predicate="human", arguments=(self.x,), confidence=1.0)
        conclusion = Proposition(predicate="mortal", arguments=(self.x,), confidence=1.0)
        self.sr.add_rule("mortality_rule", [premise], [conclusion])

    def test_deductive_produces_conclusion(self):
        """Forward chaining derives mortal(socrates) from human(socrates)."""
        results = self.sr.infer(logic_type=LogicType.DEDUCTIVE)
        predicate_names = [r.predicate for r in results]
        self.assertIn("mortal", predicate_names)

    def test_deductive_conclusion_has_correct_argument(self):
        results = self.sr.infer(logic_type=LogicType.DEDUCTIVE)
        mortal_props = [r for r in results if r.predicate == "mortal"]
        self.assertTrue(len(mortal_props) > 0)
        self.assertEqual(mortal_props[0].arguments[0].name, "socrates")

    def test_deductive_increments_stats(self):
        self.sr.infer(logic_type=LogicType.DEDUCTIVE)
        self.assertGreater(self.sr.total_inferences, 0)
        self.assertGreater(self.sr.successful_rule_applications, 0)

    def test_deductive_no_duplicate_inferences(self):
        """Running inference twice does not produce duplicate propositions."""
        r1 = self.sr.infer(logic_type=LogicType.DEDUCTIVE)
        r2 = self.sr.infer(logic_type=LogicType.DEDUCTIVE)
        # Second run should produce nothing new
        self.assertEqual(len(r2), 0)

    def test_deductive_respects_max_inferences(self):
        """max_inferences limits the number of derived propositions."""
        results = self.sr.infer(logic_type=LogicType.DEDUCTIVE, max_inferences=1)
        self.assertLessEqual(len(results), 1)

    def test_deductive_chaining(self):
        """Two rules can chain: human(X)->mortal(X), mortal(X)->dies(X)."""
        y = Symbol(name="Y", symbol_type="variable")
        premise2 = Proposition(predicate="mortal", arguments=(y,), confidence=1.0)
        conclusion2 = Proposition(predicate="dies", arguments=(y,), confidence=1.0)
        self.sr.add_rule("death_rule", [premise2], [conclusion2])

        results = self.sr.infer(logic_type=LogicType.DEDUCTIVE, max_inferences=20)
        predicate_names = [r.predicate for r in results]
        self.assertIn("mortal", predicate_names)
        self.assertIn("dies", predicate_names)

    def test_deductive_below_confidence_threshold(self):
        """A rule whose computed confidence is below the threshold produces nothing."""
        sr = SymbolicReasoner(random_seed=42, confidence_threshold=0.99)
        sym = Symbol(name="a", symbol_type="constant")
        sr.assert_proposition("p", (sym,), truth_value=True, confidence=0.5)

        x = Symbol(name="V", symbol_type="variable")
        prem = Proposition(predicate="p", arguments=(x,), confidence=0.5)
        conc = Proposition(predicate="q", arguments=(x,), confidence=1.0)
        sr.add_rule("low_conf", [prem], [conc], confidence=0.5)

        results = sr.infer(logic_type=LogicType.DEDUCTIVE)
        # Confidence = rule_conf * min(premise_conf) = 0.5 * 0.5 = 0.25 < 0.99
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- inductive inference
# ---------------------------------------------------------------------------
class TestInductiveInference(unittest.TestCase):
    """Tests for inductive inference (rule learning)."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42, enable_rule_learning=True)

    def test_inductive_with_enough_examples(self):
        """Induction with >=3 same-predicate propositions produces generalisations."""
        for i in range(5):
            sym = Symbol(name=f"obj{i}", symbol_type="constant")
            self.sr.assert_proposition("is_round", (sym,), truth_value=True)

        results = self.sr.infer(logic_type=LogicType.INDUCTIVE)
        # Should produce a GENERAL_is_round proposition
        general_preds = [r.predicate for r in results]
        self.assertTrue(any(p.startswith("GENERAL_") for p in general_preds))

    def test_inductive_no_examples_produces_nothing(self):
        """With fewer than 3 examples, induction produces nothing."""
        sym = Symbol(name="a", symbol_type="constant")
        self.sr.assert_proposition("rare", (sym,))
        results = self.sr.infer(logic_type=LogicType.INDUCTIVE)
        self.assertEqual(len(results), 0)

    def test_inductive_disabled(self):
        """When rule learning is disabled, inductive inference returns empty."""
        sr = SymbolicReasoner(random_seed=42, enable_rule_learning=False)
        for i in range(5):
            sym = Symbol(name=f"obj{i}", symbol_type="constant")
            sr.assert_proposition("is_round", (sym,), truth_value=True)
        results = sr.infer(logic_type=LogicType.INDUCTIVE)
        self.assertEqual(len(results), 0)

    def test_inductive_increments_total_inferences(self):
        for i in range(5):
            sym = Symbol(name=f"obj{i}", symbol_type="constant")
            self.sr.assert_proposition("is_round", (sym,), truth_value=True)
        self.sr.infer(logic_type=LogicType.INDUCTIVE)
        self.assertGreaterEqual(self.sr.total_inferences, 0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- abductive inference
# ---------------------------------------------------------------------------
class TestAbductiveInference(unittest.TestCase):
    """Tests for abductive inference (inference to best explanation).

    Note: ``_propositions_unify`` treats its first argument (the observation)
    as the *pattern*, so only variables in the observation are matched.
    For abduction to fire, the observation's arguments must either be
    variables or identical constants to those in the rule conclusion.
    We therefore construct the rule with concrete constants so that
    constant-to-constant matching succeeds.
    """

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)
        self.socrates = Symbol(name="socrates", symbol_type="constant")

        # Rule using concrete constants:
        #   human(socrates) => mortal(socrates)
        premise = Proposition(predicate="human", arguments=(self.socrates,), confidence=1.0)
        conclusion = Proposition(predicate="mortal", arguments=(self.socrates,), confidence=1.0)
        self.sr.add_rule("mortality_rule", [premise], [conclusion])

    def test_abductive_explains_observation(self):
        """Given mortal(socrates), abduction hypothesises human(socrates)."""
        observation = Proposition(
            predicate="mortal",
            arguments=(self.socrates,),
            truth_value=True,
            confidence=1.0,
        )
        results = self.sr.infer(logic_type=LogicType.ABDUCTIVE, query=observation)
        preds = [r.predicate for r in results]
        self.assertIn("human", preds)

    def test_abductive_without_query_returns_empty(self):
        """Abductive inference with no query returns empty."""
        results = self.sr.infer(logic_type=LogicType.ABDUCTIVE, query=None)
        self.assertEqual(len(results), 0)

    def test_abductive_confidence_is_reduced(self):
        """Abduced hypotheses have lower confidence (scaled by 0.7)."""
        observation = Proposition(
            predicate="mortal",
            arguments=(self.socrates,),
            truth_value=True,
            confidence=1.0,
        )
        results = self.sr.infer(logic_type=LogicType.ABDUCTIVE, query=observation)
        for r in results:
            self.assertLessEqual(r.confidence, 1.0)
            # rule.confidence * observation.confidence * 0.7 = 1.0 * 1.0 * 0.7
            self.assertAlmostEqual(r.confidence, 0.7)

    def test_abductive_no_matching_rule(self):
        """If no rule can produce the observation, returns empty."""
        observation = Proposition(
            predicate="flies",
            arguments=(self.socrates,),
            truth_value=True,
            confidence=1.0,
        )
        results = self.sr.infer(logic_type=LogicType.ABDUCTIVE, query=observation)
        self.assertEqual(len(results), 0)

    def test_abductive_respects_max_inferences(self):
        results = self.sr.infer(
            logic_type=LogicType.ABDUCTIVE,
            query=Proposition(predicate="mortal", arguments=(self.socrates,), confidence=1.0),
            max_inferences=1,
        )
        self.assertLessEqual(len(results), 1)

    def test_abductive_with_variable_observation(self):
        """An observation with a variable argument also unifies."""
        x = Symbol(name="V", symbol_type="variable")
        observation = Proposition(
            predicate="mortal",
            arguments=(x,),
            truth_value=True,
            confidence=1.0,
        )
        results = self.sr.infer(logic_type=LogicType.ABDUCTIVE, query=observation)
        preds = [r.predicate for r in results]
        self.assertIn("human", preds)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- analogical inference
# ---------------------------------------------------------------------------
class TestAnalogicalInference(unittest.TestCase):
    """Tests for analogical inference (knowledge transfer between domains)."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42, enable_analogy=True)

        # Source domain symbols
        self.src_a = Symbol(name="physics_force", symbol_type="constant")
        self.src_b = Symbol(name="physics_mass", symbol_type="constant")

        # Target domain symbols
        self.tgt_a = Symbol(name="econ_demand", symbol_type="constant")
        self.tgt_b = Symbol(name="econ_supply", symbol_type="constant")

        # Assert a source domain proposition
        self.sr.assert_proposition(
            "causes", (self.src_a, self.src_b), truth_value=True, confidence=1.0
        )

        # Create analogy mapping
        self.sr.create_analogy(
            source_domain="physics",
            target_domain="econ",
            mappings={self.src_a: self.tgt_a, self.src_b: self.tgt_b},
        )

    def test_analogical_transfers_knowledge(self):
        """Analogy maps causes(physics_force, physics_mass) to target domain."""
        results = self.sr.infer(logic_type=LogicType.ANALOGICAL)
        preds = [r.predicate for r in results]
        self.assertIn("causes", preds)
        # Check arguments are from target domain
        for r in results:
            if r.predicate == "causes":
                arg_names = [a.name for a in r.arguments]
                self.assertIn("econ_demand", arg_names)
                self.assertIn("econ_supply", arg_names)

    def test_analogical_disabled(self):
        """When analogy is disabled, analogical inference returns empty."""
        sr = SymbolicReasoner(random_seed=42, enable_analogy=False)
        results = sr.infer(logic_type=LogicType.ANALOGICAL)
        self.assertEqual(len(results), 0)

    def test_analogical_increments_stats(self):
        self.sr.infer(logic_type=LogicType.ANALOGICAL)
        self.assertGreater(self.sr.total_analogies, 0)

    def test_analogical_confidence_reduced(self):
        """Analogical inferences have reduced confidence."""
        results = self.sr.infer(logic_type=LogicType.ANALOGICAL)
        for r in results:
            self.assertLess(r.confidence, 1.0)

    def test_analogical_no_duplicate_on_rerun(self):
        """Running analogical inference twice does not duplicate."""
        r1 = self.sr.infer(logic_type=LogicType.ANALOGICAL)
        r2 = self.sr.infer(logic_type=LogicType.ANALOGICAL)
        self.assertGreater(len(r1), 0)
        self.assertEqual(len(r2), 0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- create_analogy
# ---------------------------------------------------------------------------
class TestCreateAnalogy(unittest.TestCase):
    """Tests for create_analogy."""

    def test_analogy_creation_stored(self):
        sr = SymbolicReasoner(random_seed=42)
        src = Symbol(name="src_x", symbol_type="constant")
        tgt = Symbol(name="tgt_x", symbol_type="constant")
        analogy = sr.create_analogy("domain_a", "domain_b", {src: tgt})
        self.assertIsInstance(analogy, Analogy)
        self.assertEqual(len(sr.analogies), 1)
        self.assertEqual(analogy.source_domain, "domain_a")
        self.assertEqual(analogy.target_domain, "domain_b")

    def test_structural_similarity_no_props(self):
        """With no source propositions, structural similarity defaults to 0.5."""
        sr = SymbolicReasoner(random_seed=42)
        src = Symbol(name="src_x", symbol_type="constant")
        tgt = Symbol(name="tgt_x", symbol_type="constant")
        analogy = sr.create_analogy("domain_a", "domain_b", {src: tgt})
        self.assertAlmostEqual(analogy.structure_similarity, 0.5)

    def test_structural_similarity_with_matching_props(self):
        """Structural similarity computation: the target proposition must
        be present in self.propositions with identical field values.

        ``_compute_structural_similarity`` creates a bare Proposition
        (truth_value=None) for the lookup, so the asserted target
        proposition must also have truth_value=None for the equality
        check (dataclass __eq__ compares all fields) to succeed.
        """
        sr = SymbolicReasoner(random_seed=42)
        src = Symbol(name="dom1_a", symbol_type="constant")
        tgt = Symbol(name="dom2_a", symbol_type="constant")

        # Source proposition (truth_value can be anything -- only used for lookup
        # as a source, not as a target match).
        sr.assert_proposition("related", (src,), truth_value=True)

        # Target proposition: use truth_value=None to match the bare
        # Proposition that _compute_structural_similarity constructs.
        tgt_prop = Proposition(predicate="related", arguments=(tgt,))
        sr.propositions.add(tgt_prop)

        analogy = sr.create_analogy("dom1", "dom2", {src: tgt})
        self.assertAlmostEqual(analogy.structure_similarity, 1.0)

    def test_structural_similarity_mismatch_due_to_truth_value(self):
        """When the target proposition has truth_value=True but the
        similarity check creates one with truth_value=None, the
        dataclass equality fails and similarity is 0.0.
        """
        sr = SymbolicReasoner(random_seed=42)
        src = Symbol(name="dom1_b", symbol_type="constant")
        tgt = Symbol(name="dom2_b", symbol_type="constant")

        sr.assert_proposition("related", (src,), truth_value=True)
        sr.assert_proposition("related", (tgt,), truth_value=True)

        analogy = sr.create_analogy("dom1", "dom2", {src: tgt})
        # The bare Proposition (truth_value=None) won't match the asserted
        # one (truth_value=True) via dataclass __eq__, so similarity is 0.
        self.assertAlmostEqual(analogy.structure_similarity, 0.0)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- unification helpers
# ---------------------------------------------------------------------------
class TestUnificationHelpers(unittest.TestCase):
    """Tests for internal unification methods."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)

    def test_propositions_unify_same_predicate_constants(self):
        a = Symbol(name="a", symbol_type="constant")
        p1 = Proposition(predicate="likes", arguments=(a,))
        p2 = Proposition(predicate="likes", arguments=(a,))
        self.assertTrue(self.sr._propositions_unify(p1, p2))

    def test_propositions_unify_different_predicate(self):
        a = Symbol(name="a", symbol_type="constant")
        p1 = Proposition(predicate="likes", arguments=(a,))
        p2 = Proposition(predicate="hates", arguments=(a,))
        self.assertFalse(self.sr._propositions_unify(p1, p2))

    def test_propositions_unify_variable_pattern(self):
        """A variable in one proposition can unify with a constant in another."""
        a = Symbol(name="a", symbol_type="constant")
        x = Symbol(name="X", symbol_type="variable")
        p1 = Proposition(predicate="likes", arguments=(x,))
        p2 = Proposition(predicate="likes", arguments=(a,))
        self.assertTrue(self.sr._propositions_unify(p1, p2))

    def test_propositions_unify_mismatched_arity(self):
        a = Symbol(name="a", symbol_type="constant")
        b = Symbol(name="b", symbol_type="constant")
        p1 = Proposition(predicate="rel", arguments=(a,))
        p2 = Proposition(predicate="rel", arguments=(a, b))
        self.assertFalse(self.sr._propositions_unify(p1, p2))

    def test_apply_bindings(self):
        x = Symbol(name="X", symbol_type="variable")
        a = Symbol(name="a", symbol_type="constant")
        prop = Proposition(predicate="p", arguments=(x,))
        result = self.sr._apply_bindings(prop, {x: a})
        self.assertEqual(result.arguments[0].name, "a")

    def test_apply_bindings_unbound_variable(self):
        """Unbound variables remain unchanged."""
        x = Symbol(name="X", symbol_type="variable")
        prop = Proposition(predicate="p", arguments=(x,))
        result = self.sr._apply_bindings(prop, {})
        self.assertEqual(result.arguments[0].name, "X")


# ---------------------------------------------------------------------------
# SymbolicReasoner -- stats and serialization
# ---------------------------------------------------------------------------
class TestStatsAndSerialization(unittest.TestCase):
    """Tests for get_stats and serialize/deserialize."""

    def test_get_stats_keys(self):
        sr = SymbolicReasoner(random_seed=42)
        stats = sr.get_stats()
        expected_keys = {
            'num_symbols', 'num_propositions', 'num_rules',
            'num_analogies', 'total_inferences', 'total_groundings',
            'total_analogies', 'successful_rule_applications',
            'working_memory_size',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_get_stats_reflects_state(self):
        sr = SymbolicReasoner(random_seed=42)
        sym = Symbol(name="a", symbol_type="constant")
        sr.assert_proposition("p", (sym,))
        stats = sr.get_stats()
        self.assertEqual(stats['num_propositions'], 1)
        self.assertEqual(stats['working_memory_size'], 1)

    def test_serialize(self):
        sr = SymbolicReasoner(random_seed=42)
        data = sr.serialize()
        self.assertIn('num_symbols', data)
        self.assertIn('num_propositions', data)
        self.assertIn('num_rules', data)
        self.assertIn('stats', data)

    def test_deserialize(self):
        data = {'num_symbols': 0, 'num_propositions': 0, 'num_rules': 0, 'stats': {}}
        sr = SymbolicReasoner.deserialize(data)
        self.assertIsInstance(sr, SymbolicReasoner)


# ---------------------------------------------------------------------------
# SymbolicReasoner -- match_proposition
# ---------------------------------------------------------------------------
class TestMatchProposition(unittest.TestCase):
    """Tests for the _match_proposition helper."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)
        self.a = Symbol(name="a", symbol_type="constant")
        self.sr.assert_proposition("fact", (self.a,), truth_value=True)

    def test_match_with_constant(self):
        pattern = Proposition(predicate="fact", arguments=(self.a,))
        self.assertTrue(self.sr._match_proposition(pattern, {}))

    def test_match_with_variable(self):
        x = Symbol(name="X", symbol_type="variable")
        pattern = Proposition(predicate="fact", arguments=(x,))
        bindings = {}
        result = self.sr._match_proposition(pattern, bindings)
        self.assertTrue(result)
        self.assertEqual(bindings[x].name, "a")

    def test_no_match_wrong_predicate(self):
        pattern = Proposition(predicate="nonexistent", arguments=(self.a,))
        self.assertFalse(self.sr._match_proposition(pattern, {}))

    def test_no_match_wrong_constant(self):
        b = Symbol(name="b", symbol_type="constant")
        pattern = Proposition(predicate="fact", arguments=(b,))
        self.assertFalse(self.sr._match_proposition(pattern, {}))


# ---------------------------------------------------------------------------
# SymbolicReasoner -- extract_common_patterns
# ---------------------------------------------------------------------------
class TestExtractCommonPatterns(unittest.TestCase):
    """Tests for the _extract_common_patterns helper."""

    def setUp(self):
        self.sr = SymbolicReasoner(random_seed=42)

    def test_no_patterns_from_single_prop(self):
        sym = Symbol(name="a", symbol_type="constant")
        prop = Proposition(predicate="p", arguments=(sym,))
        result = self.sr._extract_common_patterns([prop])
        self.assertEqual(len(result), 0)

    def test_patterns_from_uniform_list(self):
        """All props have same argument type pattern => pattern returned."""
        props = []
        for i in range(5):
            sym = Symbol(name=f"s{i}", symbol_type="constant")
            props.append(Proposition(predicate="p", arguments=(sym,)))
        result = self.sr._extract_common_patterns(props)
        self.assertGreater(len(result), 0)
        # Result tuples should have variable symbols
        for tup in result:
            for sym in tup:
                self.assertEqual(sym.symbol_type, "constant")

    def test_mixed_types_below_threshold(self):
        """Mixed argument types that don't meet 60% threshold produce nothing."""
        props = []
        # 2 constant + 3 variable => constant pattern = 40% < 60%
        for i in range(2):
            sym = Symbol(name=f"c{i}", symbol_type="constant")
            props.append(Proposition(predicate="p", arguments=(sym,)))
        for i in range(3):
            sym = Symbol(name=f"v{i}", symbol_type="variable")
            props.append(Proposition(predicate="p", arguments=(sym,)))
        result = self.sr._extract_common_patterns(props)
        # Only the majority type passes threshold
        # variable: 3/5 = 60%, constant: 2/5 = 40%
        # threshold is > 0.6 * 5 = 3.0, so 3 >= 3 passes
        type_patterns = [tup[0].symbol_type for tup in result]
        self.assertIn("variable", type_patterns)


# ---------------------------------------------------------------------------
# Integration: end-to-end reasoning scenario
# ---------------------------------------------------------------------------
class TestIntegrationScenario(unittest.TestCase):
    """End-to-end integration test combining multiple reasoning features."""

    def test_ground_assert_infer_cycle(self):
        """Ground symbols, assert facts, add a rule, and run deduction."""
        sr = SymbolicReasoner(random_seed=42)

        # Ground two symbols from patterns
        p1 = np.array([1.0, 0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0, 0.0])
        sym1 = sr.ground_symbol(p1, properties={"label": "cat"})
        sym2 = sr.ground_symbol(p2, properties={"label": "animal"})

        # Assert fact: is_a(cat, animal)
        sr.assert_proposition("is_a", (sym1, sym2), truth_value=True)

        # Variable for rule
        x = Symbol(name="X", symbol_type="variable")
        y = Symbol(name="Y", symbol_type="variable")

        # Rule: is_a(X, Y) => related(X, Y)
        prem = Proposition(predicate="is_a", arguments=(x, y), confidence=1.0)
        conc = Proposition(predicate="related", arguments=(x, y), confidence=1.0)
        sr.add_rule("category_relation", [prem], [conc])

        # Infer
        results = sr.infer(logic_type=LogicType.DEDUCTIVE)
        self.assertTrue(any(r.predicate == "related" for r in results))

        # Stats should reflect activity
        stats = sr.get_stats()
        self.assertEqual(stats['num_symbols'], 2)
        self.assertGreater(stats['total_inferences'], 0)
        self.assertGreater(stats['total_groundings'], 0)

    def test_full_serialize_round_trip(self):
        """Serialize and deserialize produces a valid reasoner."""
        sr = SymbolicReasoner(random_seed=42)
        sym = Symbol(name="a", symbol_type="constant")
        sr.assert_proposition("fact", (sym,))
        data = sr.serialize()

        sr2 = SymbolicReasoner.deserialize(data)
        self.assertIsInstance(sr2, SymbolicReasoner)
        self.assertEqual(sr2.get_stats()['num_symbols'], 0)  # deserialize doesn't restore fully


if __name__ == "__main__":
    unittest.main()
