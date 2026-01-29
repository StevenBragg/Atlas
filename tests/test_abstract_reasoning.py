"""
Comprehensive tests for the Abstract Reasoning module.

Tests cover:
- KnowledgeBase: adding rules, facts, querying, relations, type hierarchy
- LogicEngine: forward chaining, backward chaining, expression evaluation
- AnalogyEngine: structural mapping, similarity assessment, knowledge transfer
- PatternDetector: arithmetic, geometric, repetition, alternation, fibonacci
- RuleInducer: rule learning from examples
- AbstractReasoner: unified reasoning across all sub-engines
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.abstract_reasoning import (
    AbstractReasoner,
    KnowledgeBase,
    LogicEngine,
    AnalogyEngine,
    PatternDetector,
    RuleInducer,
    Proposition,
    Rule,
    Analogy,
    Pattern,
    Symbol,
    Predicate,
    LogicOperator,
    RelationType,
)


# ---------------------------------------------------------------------------
# Data-class / enum helpers
# ---------------------------------------------------------------------------

class TestDataClasses(unittest.TestCase):
    """Verify that data-classes and enums are constructed correctly."""

    def test_symbol_creation_and_hash(self):
        s1 = Symbol(name="cat")
        s2 = Symbol(name="cat")
        s3 = Symbol(name="dog")
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertEqual(hash(s1), hash(s2))

    def test_symbol_not_equal_to_non_symbol(self):
        s = Symbol(name="cat")
        self.assertNotEqual(s, "cat")

    def test_symbol_properties(self):
        s = Symbol(name="rover", symbol_type="entity", properties={"breed": "lab"})
        self.assertEqual(s.symbol_type, "entity")
        self.assertEqual(s.properties["breed"], "lab")

    def test_predicate_call_creates_proposition(self):
        p = Predicate(name="likes", arity=2)
        prop = p("alice", "bob")
        self.assertIsInstance(prop, Proposition)
        self.assertEqual(prop.predicate, "likes")
        self.assertEqual(list(prop.arguments), ["alice", "bob"])

    def test_predicate_repr(self):
        p = Predicate(name="above", arity=2, arguments=[Symbol("a"), Symbol("b")])
        self.assertIn("above", repr(p))

    def test_proposition_equality(self):
        p1 = Proposition(predicate="likes", arguments=["a", "b"])
        p2 = Proposition(predicate="likes", arguments=["a", "b"])
        p3 = Proposition(predicate="likes", arguments=["a", "c"])
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)

    def test_proposition_hash(self):
        p1 = Proposition(predicate="likes", arguments=["a", "b"])
        p2 = Proposition(predicate="likes", arguments=["a", "b"])
        self.assertEqual(hash(p1), hash(p2))

    def test_proposition_not_equal_to_non_proposition(self):
        p = Proposition(predicate="likes", arguments=["a", "b"])
        self.assertNotEqual(p, "likes(a, b)")

    def test_proposition_repr(self):
        p = Proposition(predicate="parent", arguments=["tom", "bob"])
        self.assertEqual(repr(p), "parent(tom, bob)")

    def test_rule_repr(self):
        ant = [Proposition(predicate="parent", arguments=["X", "Y"])]
        con = Proposition(predicate="ancestor", arguments=["X", "Y"])
        r = Rule(name="r1", antecedent=ant, consequent=con)
        text = repr(r)
        self.assertIn("r1", text)
        self.assertIn("ancestor", text)

    def test_logic_operator_values(self):
        self.assertEqual(LogicOperator.AND.value, "and")
        self.assertEqual(LogicOperator.OR.value, "or")
        self.assertEqual(LogicOperator.NOT.value, "not")
        self.assertEqual(LogicOperator.IMPLIES.value, "implies")
        self.assertEqual(LogicOperator.IFF.value, "iff")
        self.assertEqual(LogicOperator.XOR.value, "xor")

    def test_relation_type_values(self):
        self.assertEqual(RelationType.IS_A.value, "is_a")
        self.assertEqual(RelationType.CAUSES.value, "causes")

    def test_pattern_dataclass(self):
        p = Pattern(pattern_type="arithmetic", elements=[1, 2, 3],
                    structure="a(n) = a(0) + n * 1", confidence=0.9)
        self.assertEqual(p.pattern_type, "arithmetic")
        self.assertAlmostEqual(p.confidence, 0.9)

    def test_analogy_transfer(self):
        analogy = Analogy(
            source_domain="solar",
            target_domain="atom",
            mappings={"sun": "nucleus", "planet": "electron"},
            relation_mappings={"revolves_around": "orbits"},
            strength=0.8,
        )
        src = Proposition(predicate="revolves_around", arguments=["planet", "sun"])
        transferred = analogy.transfer(src)
        self.assertIsNotNone(transferred)
        self.assertEqual(transferred.predicate, "orbits")
        self.assertEqual(list(transferred.arguments), ["electron", "nucleus"])
        self.assertAlmostEqual(transferred.confidence, 0.8)  # 1.0 * 0.8

    def test_analogy_transfer_partial_mapping(self):
        analogy = Analogy(
            source_domain="s", target_domain="t",
            mappings={"a": "x"},
            relation_mappings={},
            strength=0.5,
        )
        src = Proposition(predicate="foo", arguments=["a", "unmapped"])
        transferred = analogy.transfer(src)
        self.assertEqual(transferred.predicate, "foo")
        self.assertEqual(list(transferred.arguments), ["x", "unmapped"])


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class TestKnowledgeBase(unittest.TestCase):
    """Tests for KnowledgeBase: facts, rules, relations, queries."""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_add_fact_and_query_direct(self):
        fact = Proposition(predicate="is_animal", arguments=["cat"])
        self.kb.add_fact(fact)
        found, conf = self.kb.query(fact)
        self.assertTrue(found)
        self.assertGreater(conf, 0.0)

    def test_query_missing_fact(self):
        q = Proposition(predicate="is_animal", arguments=["rock"])
        found, conf = self.kb.query(q)
        self.assertFalse(found)
        self.assertEqual(conf, 0.0)

    def test_add_rule_and_derive(self):
        # Facts: parent(tom, bob)
        self.kb.add_fact(Proposition(predicate="parent", arguments=["tom", "bob"]))
        # Rule: parent(X, Y) -> ancestor(X, Y)
        rule = Rule(
            name="parent_is_ancestor",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        )
        self.kb.add_rule(rule)
        # Query via rule-based derivation
        q = Proposition(predicate="ancestor", arguments=["tom", "bob"])
        found, conf = self.kb.query(q)
        self.assertTrue(found)

    def test_add_relation_and_get_related(self):
        self.kb.add_relation("cat", "animal", RelationType.IS_A)
        related = self.kb.get_related("cat")
        entities = [e for e, _, _ in related]
        self.assertIn("animal", entities)

    def test_get_related_with_filter(self):
        self.kb.add_relation("cat", "animal", RelationType.IS_A)
        self.kb.add_relation("cat", "fur", RelationType.HAS)
        related_is_a = self.kb.get_related("cat", RelationType.IS_A)
        self.assertEqual(len(related_is_a), 1)
        self.assertEqual(related_is_a[0][0], "animal")

    def test_get_related_reverse_lookup(self):
        self.kb.add_relation("dog", "animal", RelationType.IS_A)
        related = self.kb.get_related("animal")
        entities = [e for e, _, _ in related]
        self.assertIn("dog", entities)

    def test_type_hierarchy(self):
        self.kb.add_relation("poodle", "dog", RelationType.IS_A)
        self.kb.add_relation("dog", "animal", RelationType.IS_A)
        ancestors = self.kb.get_type_ancestors("poodle")
        self.assertIn("dog", ancestors)
        self.assertIn("animal", ancestors)

    def test_type_hierarchy_empty(self):
        ancestors = self.kb.get_type_ancestors("unknown")
        self.assertEqual(len(ancestors), 0)

    def test_add_fact_registers_symbols(self):
        self.kb.add_fact(Proposition(predicate="color", arguments=["sky", "blue"]))
        self.assertIn("sky", self.kb.symbols)
        self.assertIn("blue", self.kb.symbols)

    def test_multiple_rules_derive_chain(self):
        # parent(tom, bob), parent(bob, jim)
        self.kb.add_fact(Proposition(predicate="parent", arguments=["tom", "bob"]))
        self.kb.add_fact(Proposition(predicate="parent", arguments=["bob", "jim"]))
        # Rule: parent(X, Y) -> ancestor(X, Y)
        self.kb.add_rule(Rule(
            name="r1",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        ))
        found, _ = self.kb.query(Proposition(predicate="ancestor", arguments=["tom", "bob"]))
        self.assertTrue(found)
        found2, _ = self.kb.query(Proposition(predicate="ancestor", arguments=["bob", "jim"]))
        self.assertTrue(found2)


# ---------------------------------------------------------------------------
# LogicEngine
# ---------------------------------------------------------------------------

class TestLogicEngine(unittest.TestCase):
    """Tests for LogicEngine: evaluate, forward_chain, backward_chain."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.engine = LogicEngine(self.kb)

    # -- evaluate --

    def test_evaluate_known_fact(self):
        self.kb.add_fact(Proposition(predicate="sunny", arguments=["today"]))
        result, conf = self.engine.evaluate(
            Proposition(predicate="sunny", arguments=["today"])
        )
        self.assertTrue(result)

    def test_evaluate_not(self):
        self.kb.add_fact(Proposition(predicate="sunny", arguments=["today"]))
        result, _ = self.engine.evaluate(
            (LogicOperator.NOT, Proposition(predicate="sunny", arguments=["today"]))
        )
        self.assertFalse(result)

    def test_evaluate_and_true(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        self.kb.add_fact(Proposition(predicate="b", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.AND,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)

    def test_evaluate_and_false(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.AND,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertFalse(result)

    def test_evaluate_or(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.OR,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)

    def test_evaluate_implies_true(self):
        # F -> anything is True
        result, _ = self.engine.evaluate((
            LogicOperator.IMPLIES,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)  # false -> false = true

    def test_evaluate_implies_false(self):
        # T -> F is False
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.IMPLIES,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertFalse(result)

    def test_evaluate_iff_both_true(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        self.kb.add_fact(Proposition(predicate="b", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.IFF,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)

    def test_evaluate_iff_both_false(self):
        result, _ = self.engine.evaluate((
            LogicOperator.IFF,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)  # false <-> false = true

    def test_evaluate_xor(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        result, _ = self.engine.evaluate((
            LogicOperator.XOR,
            Proposition(predicate="a", arguments=["1"]),
            Proposition(predicate="b", arguments=["1"]),
        ))
        self.assertTrue(result)  # true xor false = true

    def test_evaluate_invalid_expression(self):
        result, _ = self.engine.evaluate(("bad",))
        self.assertFalse(result)

    def test_evaluate_invalid_tuple_no_third(self):
        result, _ = self.engine.evaluate((LogicOperator.AND, Proposition(predicate="a", arguments=["1"])))
        self.assertFalse(result)

    # -- forward chaining --

    def test_forward_chain_derives_new_facts(self):
        self.kb.add_fact(Proposition(predicate="parent", arguments=["tom", "bob"]))
        self.kb.add_rule(Rule(
            name="r1",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        ))
        new_facts = self.engine.forward_chain()
        predicates = [f.predicate for f in new_facts]
        self.assertIn("ancestor", predicates)
        # verify stat tracking
        self.assertGreaterEqual(self.engine.inferences_made, 1)
        self.assertGreaterEqual(self.engine.rules_applied, 1)

    def test_forward_chain_no_rules(self):
        self.kb.add_fact(Proposition(predicate="a", arguments=["1"]))
        new_facts = self.engine.forward_chain()
        self.assertEqual(len(new_facts), 0)

    def test_forward_chain_multi_antecedent(self):
        self.kb.add_fact(Proposition(predicate="has_feathers", arguments=["tweety"]))
        self.kb.add_fact(Proposition(predicate="can_fly", arguments=["tweety"]))
        self.kb.add_rule(Rule(
            name="bird_rule",
            antecedent=[
                Proposition(predicate="has_feathers", arguments=["X"]),
                Proposition(predicate="can_fly", arguments=["X"]),
            ],
            consequent=Proposition(predicate="is_bird", arguments=["X"]),
        ))
        new_facts = self.engine.forward_chain()
        derived_preds = [f.predicate for f in new_facts]
        self.assertIn("is_bird", derived_preds)
        # Check the derived argument
        bird_facts = [f for f in new_facts if f.predicate == "is_bird"]
        self.assertTrue(any("tweety" in [str(a) for a in f.arguments] for f in bird_facts))

    def test_forward_chain_idempotent(self):
        self.kb.add_fact(Proposition(predicate="parent", arguments=["tom", "bob"]))
        self.kb.add_rule(Rule(
            name="r1",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        ))
        first = self.engine.forward_chain()
        second = self.engine.forward_chain()
        # second run should derive nothing new
        self.assertEqual(len(second), 0)

    # -- backward chaining --

    def test_backward_chain_proves_known_fact(self):
        self.kb.add_fact(Proposition(predicate="is_cat", arguments=["tom"]))
        proven, proof = self.engine.backward_chain(
            Proposition(predicate="is_cat", arguments=["tom"])
        )
        self.assertTrue(proven)
        self.assertGreater(len(proof), 0)

    def test_backward_chain_proves_via_rule(self):
        self.kb.add_fact(Proposition(predicate="parent", arguments=["tom", "bob"]))
        self.kb.add_rule(Rule(
            name="r1",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        ))
        proven, proof = self.engine.backward_chain(
            Proposition(predicate="ancestor", arguments=["tom", "bob"])
        )
        self.assertTrue(proven)
        self.assertGreater(len(proof), 0)

    def test_backward_chain_fails_for_unknown(self):
        proven, proof = self.engine.backward_chain(
            Proposition(predicate="flies", arguments=["rock"])
        )
        self.assertFalse(proven)
        self.assertEqual(len(proof), 0)

    def test_backward_chain_depth_limit(self):
        # Create a rule that references itself indirectly — should not infinite-loop
        self.kb.add_rule(Rule(
            name="loop",
            antecedent=[Proposition(predicate="p", arguments=["X"])],
            consequent=Proposition(predicate="p", arguments=["X"]),
        ))
        proven, _ = self.engine.backward_chain(
            Proposition(predicate="p", arguments=["a"]),
            max_depth=5,
        )
        self.assertFalse(proven)


# ---------------------------------------------------------------------------
# AnalogyEngine
# ---------------------------------------------------------------------------

class TestAnalogyEngine(unittest.TestCase):
    """Tests for AnalogyEngine: find_analogy, transfer_knowledge."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.engine = AnalogyEngine(self.kb)

    def test_find_analogy_structural(self):
        source = [
            Proposition(predicate="orbits", arguments=["earth", "sun"]),
            Proposition(predicate="orbits", arguments=["mars", "sun"]),
        ]
        target = [
            Proposition(predicate="orbits", arguments=["electron", "nucleus"]),
            Proposition(predicate="orbits", arguments=["proton", "nucleus"]),
        ]
        analogy = self.engine.find_analogy(source, target)
        self.assertIsNotNone(analogy)
        self.assertIsInstance(analogy.mappings, dict)
        self.assertGreater(len(analogy.mappings), 0)

    def test_find_analogy_returns_none_for_empty_source(self):
        result = self.engine.find_analogy([], [
            Proposition(predicate="x", arguments=["a", "b"])
        ])
        self.assertIsNone(result)

    def test_find_analogy_returns_none_for_empty_target(self):
        result = self.engine.find_analogy([
            Proposition(predicate="x", arguments=["a", "b"])
        ], [])
        self.assertIsNone(result)

    def test_transfer_knowledge(self):
        analogy = Analogy(
            source_domain="solar",
            target_domain="atom",
            mappings={"sun": "nucleus", "planet": "electron"},
            relation_mappings={"revolves": "orbits"},
            strength=0.8,
        )
        source_facts = [
            Proposition(predicate="revolves", arguments=["planet", "sun"]),
        ]
        transferred = self.engine.transfer_knowledge(analogy, source_facts)
        self.assertEqual(len(transferred), 1)
        self.assertEqual(transferred[0].predicate, "orbits")
        self.assertEqual(list(transferred[0].arguments), ["electron", "nucleus"])

    def test_analogy_stored_in_engine(self):
        source = [
            Proposition(predicate="bigger", arguments=["a", "b"]),
        ]
        target = [
            Proposition(predicate="bigger", arguments=["x", "y"]),
        ]
        self.engine.find_analogy(source, target)
        self.assertGreaterEqual(len(self.engine.analogies), 1)

    def test_infer_domain_name(self):
        facts = [
            Proposition(predicate="flies", arguments=["eagle"]),
            Proposition(predicate="flies", arguments=["sparrow"]),
            Proposition(predicate="swims", arguments=["fish"]),
        ]
        name = self.engine._infer_domain_name(facts)
        self.assertEqual(name, "flies_domain")

    def test_infer_domain_name_empty(self):
        name = self.engine._infer_domain_name([])
        self.assertEqual(name, "unknown_domain")


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------

class TestPatternDetector(unittest.TestCase):
    """Tests for PatternDetector: all five pattern types and prediction."""

    def setUp(self):
        self.detector = PatternDetector()

    # -- arithmetic --

    def test_detect_arithmetic(self):
        seq = [2, 4, 6, 8, 10]
        pattern = self.detector.detect_pattern(seq, 'arithmetic')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'arithmetic')
        self.assertAlmostEqual(pattern.confidence, 1.0)

    def test_detect_arithmetic_negative_diff(self):
        seq = [10, 7, 4, 1]
        pattern = self.detector.detect_pattern(seq, 'arithmetic')
        self.assertIsNotNone(pattern)

    # -- geometric --

    def test_detect_geometric(self):
        seq = [3, 6, 12, 24]
        pattern = self.detector.detect_pattern(seq, 'geometric')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'geometric')

    def test_geometric_zero_element_returns_none(self):
        seq = [0, 1, 2, 4]
        pattern = self.detector.detect_pattern(seq, 'geometric')
        self.assertIsNone(pattern)

    # -- repetition --

    def test_detect_repetition(self):
        seq = ['a', 'b', 'a', 'b', 'a', 'b']
        pattern = self.detector.detect_pattern(seq, 'repetition')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'repetition')
        self.assertIn('period', pattern.structure)

    def test_detect_repetition_period_3(self):
        seq = [1, 2, 3, 1, 2, 3]
        pattern = self.detector.detect_pattern(seq, 'repetition')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.elements, [1, 2, 3])

    # -- alternation --

    def test_detect_alternation(self):
        seq = [0, 1, 0, 1, 0, 1]
        pattern = self.detector.detect_pattern(seq, 'alternation')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'alternation')

    def test_alternation_same_values_returns_none(self):
        seq = [5, 5, 5, 5]
        pattern = self.detector.detect_pattern(seq, 'alternation')
        self.assertIsNone(pattern)

    # -- fibonacci --

    def test_detect_fibonacci(self):
        seq = [1, 1, 2, 3, 5, 8]
        pattern = self.detector.detect_pattern(seq, 'fibonacci')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'fibonacci')

    def test_detect_fibonacci_custom_start(self):
        seq = [2, 3, 5, 8, 13]
        pattern = self.detector.detect_pattern(seq, 'fibonacci')
        self.assertIsNotNone(pattern)

    # -- auto detection (pattern_type=None) --

    def test_auto_detect_arithmetic(self):
        seq = [5, 10, 15, 20]
        pattern = self.detector.detect_pattern(seq)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'arithmetic')

    def test_auto_detect_returns_first_match(self):
        # [2, 4, 8, 16] is geometric; auto-detect tries arithmetic first,
        # which won't match, then geometric should match.
        seq = [2, 4, 8, 16]
        pattern = self.detector.detect_pattern(seq)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'geometric')

    # -- edge cases --

    def test_too_short_sequence(self):
        self.assertIsNone(self.detector.detect_pattern([1, 2]))

    def test_non_numeric_for_arithmetic(self):
        self.assertIsNone(self.detector.detect_pattern(["a", "b", "c"], 'arithmetic'))

    def test_unknown_pattern_type(self):
        self.assertIsNone(self.detector.detect_pattern([1, 2, 3], 'unknown_type'))

    # -- predict_next --

    def test_predict_arithmetic(self):
        seq = [1, 3, 5, 7]
        preds = self.detector.predict_next(seq, 2)
        self.assertEqual(preds, [9, 11])

    def test_predict_geometric(self):
        seq = [2, 6, 18, 54]
        preds = self.detector.predict_next(seq, 1)
        self.assertAlmostEqual(preds[0], 162.0)

    def test_predict_repetition(self):
        seq = ['x', 'y', 'z', 'x', 'y', 'z']
        preds = self.detector.predict_next(seq, 3)
        self.assertEqual(preds, ['x', 'y', 'z'])

    def test_predict_alternation(self):
        seq = [0, 1, 0, 1, 0]
        preds = self.detector.predict_next(seq, 2)
        self.assertEqual(preds, [1, 0])

    def test_predict_fibonacci(self):
        seq = [1, 1, 2, 3, 5]
        preds = self.detector.predict_next(seq, 2)
        self.assertEqual(preds, [8, 13])

    def test_predict_no_pattern(self):
        seq = [3, 1, 4, 1, 5, 9]
        preds = self.detector.predict_next(seq)
        # may or may not detect; we just check it doesn't crash
        self.assertIsInstance(preds, list)

    def test_predict_from_short_sequence(self):
        preds = self.detector.predict_next([1])
        self.assertEqual(preds, [])


# ---------------------------------------------------------------------------
# RuleInducer
# ---------------------------------------------------------------------------

class TestRuleInducer(unittest.TestCase):
    """Tests for RuleInducer: induce_from_examples."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.inducer = RuleInducer(self.kb)

    def test_induce_from_examples_with_background(self):
        # Background knowledge
        bg = [
            Proposition(predicate="has_wings", arguments=["eagle"]),
            Proposition(predicate="has_wings", arguments=["sparrow"]),
        ]
        for f in bg:
            self.kb.add_fact(f)

        positive = [
            Proposition(predicate="can_fly", arguments=["eagle"]),
            Proposition(predicate="can_fly", arguments=["sparrow"]),
        ]
        rules = self.inducer.induce_from_examples(positive, background=bg)
        self.assertIsInstance(rules, list)
        # rules may or may not be induced depending on confidence threshold
        # but the process should complete without error

    def test_induce_no_rules_for_single_example(self):
        positive = [Proposition(predicate="can_fly", arguments=["eagle"])]
        rules = self.inducer.induce_from_examples(positive)
        # Fewer than 2 examples per predicate means no induction
        self.assertEqual(len(rules), 0)

    def test_induced_rules_stored(self):
        bg = [
            Proposition(predicate="feathered", arguments=["a"]),
            Proposition(predicate="feathered", arguments=["b"]),
        ]
        for f in bg:
            self.kb.add_fact(f)
        positive = [
            Proposition(predicate="bird", arguments=["a"]),
            Proposition(predicate="bird", arguments=["b"]),
        ]
        rules = self.inducer.induce_from_examples(positive, background=bg)
        # All induced rules should also be in self.inducer.induced_rules
        self.assertEqual(len(rules), len(self.inducer.induced_rules))

    def test_confidence_with_negatives(self):
        bg = [
            Proposition(predicate="trait", arguments=["x"]),
            Proposition(predicate="trait", arguments=["y"]),
        ]
        for f in bg:
            self.kb.add_fact(f)
        positive = [
            Proposition(predicate="label", arguments=["x"]),
            Proposition(predicate="label", arguments=["y"]),
        ]
        negatives = [
            Proposition(predicate="label", arguments=["z"]),
        ]
        rules = self.inducer.induce_from_examples(positive, negatives, bg)
        # Confidence should reflect presence of negatives
        for r in rules:
            self.assertLessEqual(r.confidence, 1.0)

    def test_extract_patterns_constant_arg(self):
        examples = [
            Proposition(predicate="likes", arguments=["alice", "pizza"]),
            Proposition(predicate="likes", arguments=["bob", "pizza"]),
        ]
        patterns = self.inducer._extract_patterns(examples)
        # Second argument is always "pizza", first varies
        self.assertEqual(patterns[1], "pizza")
        self.assertTrue(patterns[0].startswith("?"))

    def test_extract_patterns_empty(self):
        self.assertEqual(self.inducer._extract_patterns([]), [])


# ---------------------------------------------------------------------------
# AbstractReasoner (unified interface)
# ---------------------------------------------------------------------------

class TestAbstractReasoner(unittest.TestCase):
    """Tests for AbstractReasoner: the top-level reasoning orchestrator."""

    def setUp(self):
        self.reasoner = AbstractReasoner()

    # -- initialization --

    def test_initial_state(self):
        stats = self.reasoner.get_stats()
        self.assertEqual(stats['queries_answered'], 0)
        self.assertEqual(stats['total_facts'], 0)
        self.assertEqual(stats['total_rules'], 0)

    # -- add_knowledge --

    def test_add_facts(self):
        facts = [
            Proposition(predicate="color", arguments=["sky", "blue"]),
            Proposition(predicate="color", arguments=["grass", "green"]),
        ]
        self.reasoner.add_knowledge(facts=facts)
        self.assertEqual(self.reasoner.get_stats()['total_facts'], 2)

    def test_add_rules(self):
        rules = [Rule(
            name="r1",
            antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
            consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
        )]
        self.reasoner.add_knowledge(rules=rules)
        self.assertEqual(self.reasoner.get_stats()['total_rules'], 1)

    def test_add_relations(self):
        relations = [("cat", "animal", RelationType.IS_A)]
        self.reasoner.add_knowledge(relations=relations)
        related = self.reasoner.get_related_concepts("cat")
        self.assertEqual(len(related), 1)

    # -- query --

    def test_query_direct_fact(self):
        self.reasoner.add_knowledge(facts=[
            Proposition(predicate="round", arguments=["ball"]),
        ])
        result, conf, proof = self.reasoner.query(
            Proposition(predicate="round", arguments=["ball"])
        )
        self.assertTrue(result)
        self.assertGreater(conf, 0.0)
        self.assertGreater(len(proof), 0)
        self.assertEqual(self.reasoner.get_stats()['queries_answered'], 1)

    def test_query_derived_via_backward_chaining(self):
        self.reasoner.add_knowledge(
            facts=[Proposition(predicate="parent", arguments=["tom", "bob"])],
            rules=[Rule(
                name="r1",
                antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
                consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
            )],
        )
        result, conf, proof = self.reasoner.query(
            Proposition(predicate="ancestor", arguments=["tom", "bob"])
        )
        self.assertTrue(result)
        self.assertGreater(len(proof), 0)

    def test_query_unknown(self):
        result, conf, proof = self.reasoner.query(
            Proposition(predicate="invisible", arguments=["ghost"])
        )
        self.assertFalse(result)
        self.assertEqual(conf, 0.0)
        self.assertEqual(len(proof), 0)

    # -- pattern detection --

    def test_detect_pattern(self):
        pattern = self.reasoner.detect_pattern([10, 20, 30, 40])
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, 'arithmetic')
        self.assertEqual(self.reasoner.get_stats()['patterns_detected'], 1)

    def test_predict_sequence(self):
        preds = self.reasoner.predict_sequence([1, 2, 3, 4], n=3)
        self.assertEqual(preds, [5, 6, 7])

    # -- analogy --

    def test_find_analogy(self):
        source = [
            Proposition(predicate="flows", arguments=["water", "pipe"]),
            Proposition(predicate="flows", arguments=["current", "wire"]),
        ]
        target = [
            Proposition(predicate="flows", arguments=["blood", "artery"]),
            Proposition(predicate="flows", arguments=["lymph", "vessel"]),
        ]
        analogy = self.reasoner.find_analogy(source, target)
        self.assertIsNotNone(analogy)
        self.assertEqual(self.reasoner.get_stats()['analogies_found'], 1)

    def test_find_analogy_none(self):
        analogy = self.reasoner.find_analogy([], [])
        self.assertIsNone(analogy)
        # counter should NOT increment
        self.assertEqual(self.reasoner.get_stats()['analogies_found'], 0)

    # -- rule induction --

    def test_induce_rules(self):
        self.reasoner.add_knowledge(facts=[
            Proposition(predicate="has_engine", arguments=["car"]),
            Proposition(predicate="has_engine", arguments=["truck"]),
        ])
        positive = [
            Proposition(predicate="is_vehicle", arguments=["car"]),
            Proposition(predicate="is_vehicle", arguments=["truck"]),
        ]
        rules = self.reasoner.induce_rules(positive)
        self.assertIsInstance(rules, list)
        # Induced rules get added to kb
        self.assertEqual(
            self.reasoner.get_stats()['rules_induced'],
            len(rules),
        )

    # -- compose_concepts --

    def test_compose_concepts(self):
        composite = self.reasoner.compose_concepts(["red", "car"], relation="colored")
        self.assertEqual(composite.symbol_type, "composite")
        self.assertIn("red", composite.properties['components'])
        self.assertIn("car", composite.properties['components'])
        self.assertIn(composite.name, self.reasoner.kb.symbols)

    def test_compose_concepts_with_embeddings(self):
        # Pre-register symbols with embeddings
        import numpy as np
        s1 = Symbol(name="a", embedding=np.ones(4))
        s2 = Symbol(name="b", embedding=np.ones(4) * 3)
        self.reasoner.kb.symbols["a"] = s1
        self.reasoner.kb.symbols["b"] = s2
        composite = self.reasoner.compose_concepts(["a", "b"])
        self.assertIsNotNone(composite.embedding)
        # Average of [1,1,1,1] and [3,3,3,3] = [2,2,2,2]
        np.testing.assert_array_almost_equal(composite.embedding, np.ones(4) * 2)

    # -- explain_reasoning --

    def test_explain_known_fact(self):
        prop = Proposition(predicate="sunny", arguments=["today"])
        explanation = self.reasoner.explain_reasoning(prop, True, [prop])
        self.assertIn("known fact", explanation)

    def test_explain_derived(self):
        goal = Proposition(predicate="ancestor", arguments=["tom", "bob"])
        step = Proposition(predicate="parent", arguments=["tom", "bob"])
        explanation = self.reasoner.explain_reasoning(goal, True, [goal, step])
        self.assertIn("To prove", explanation)
        self.assertIn("Therefore", explanation)

    def test_explain_failure(self):
        prop = Proposition(predicate="flies", arguments=["rock"])
        explanation = self.reasoner.explain_reasoning(prop, False, [])
        self.assertIn("Could not prove", explanation)

    # -- get_related_concepts --

    def test_get_related_concepts_with_limit(self):
        for i in range(15):
            self.reasoner.kb.add_relation("center", f"node_{i}", RelationType.RELATED_TO)
        results = self.reasoner.get_related_concepts("center", max_results=5)
        self.assertLessEqual(len(results), 5)

    # -- serialization --

    def test_serialize_deserialize_round_trip(self):
        self.reasoner.add_knowledge(
            facts=[
                Proposition(predicate="parent", arguments=["tom", "bob"]),
                Proposition(predicate="parent", arguments=["bob", "jim"]),
            ],
            rules=[Rule(
                name="r1",
                antecedent=[Proposition(predicate="parent", arguments=["X", "Y"])],
                consequent=Proposition(predicate="ancestor", arguments=["X", "Y"]),
                confidence=0.95,
            )],
        )

        data = self.reasoner.serialize()
        restored = AbstractReasoner.deserialize(data)

        self.assertEqual(len(restored.kb.facts), 2)
        self.assertEqual(len(restored.kb.rules), 1)
        self.assertEqual(restored.kb.rules[0].name, "r1")
        self.assertAlmostEqual(restored.kb.rules[0].confidence, 0.95)

    def test_serialize_empty_reasoner(self):
        data = self.reasoner.serialize()
        self.assertEqual(len(data['facts']), 0)
        self.assertEqual(len(data['rules']), 0)

    def test_deserialize_empty(self):
        restored = AbstractReasoner.deserialize({})
        self.assertEqual(len(restored.kb.facts), 0)
        self.assertEqual(len(restored.kb.rules), 0)


# ---------------------------------------------------------------------------
# Backend import sanity check
# ---------------------------------------------------------------------------

class TestBackendImport(unittest.TestCase):
    """Verify that the backend xp module is usable."""

    def test_xp_array_creation(self):
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertEqual(arr.shape, (3,))

    def test_xp_zeros(self):
        z = xp.zeros((2, 3))
        self.assertEqual(z.shape, (2, 3))
        self.assertAlmostEqual(float(xp.sum(z)), 0.0)


if __name__ == '__main__':
    unittest.main()
