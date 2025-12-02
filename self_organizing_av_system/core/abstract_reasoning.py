"""
Abstract Logic and Reasoning System for ATLAS

Implements higher-order reasoning capabilities:
- Symbolic logic (propositional and predicate)
- Analogy detection and transfer
- Abstract pattern recognition
- Rule induction from examples
- Compositional reasoning
- Relational reasoning

This enables ATLAS to reason about abstract relationships,
detect patterns, and transfer knowledge across domains.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class LogicOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"


class RelationType(Enum):
    """Types of relations between entities"""
    IS_A = "is_a"           # Category membership
    PART_OF = "part_of"     # Meronymy
    HAS = "has"             # Possession
    CAUSES = "causes"       # Causation
    BEFORE = "before"       # Temporal
    AFTER = "after"         # Temporal
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Opposition
    RELATED_TO = "related_to"  # General relation


@dataclass
class Symbol:
    """A symbolic entity in the reasoning system"""
    name: str
    symbol_type: str = "entity"  # entity, predicate, variable, constant
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        return False


@dataclass
class Predicate:
    """A predicate (relation) that can be true or false"""
    name: str
    arity: int  # Number of arguments
    arguments: List[Symbol] = field(default_factory=list)
    truth_value: Optional[bool] = None
    confidence: float = 1.0

    def __call__(self, *args) -> 'Proposition':
        """Create a proposition from this predicate"""
        return Proposition(
            predicate=self.name,
            arguments=list(args),
            truth_value=self.truth_value,
        )

    def __repr__(self):
        args = ", ".join(str(a) for a in self.arguments)
        return f"{self.name}({args})"


@dataclass
class Proposition:
    """A logical proposition that can be true or false"""
    predicate: str
    arguments: List[Any]
    truth_value: Optional[bool] = None
    confidence: float = 1.0

    def __repr__(self):
        args = ", ".join(str(a) for a in self.arguments)
        return f"{self.predicate}({args})"

    def __hash__(self):
        return hash((self.predicate, tuple(str(a) for a in self.arguments)))

    def __eq__(self, other):
        if isinstance(other, Proposition):
            return (self.predicate == other.predicate and
                    len(self.arguments) == len(other.arguments) and
                    all(str(a) == str(b) for a, b in zip(self.arguments, other.arguments)))
        return False


@dataclass
class Rule:
    """A logical rule (implication)"""
    name: str
    antecedent: List[Proposition]  # If these are true...
    consequent: Proposition  # Then this is true
    confidence: float = 1.0
    applications: int = 0  # How many times applied

    def __repr__(self):
        ant = " ∧ ".join(str(p) for p in self.antecedent)
        return f"{self.name}: {ant} → {self.consequent}"


@dataclass
class Analogy:
    """An analogy between two domains"""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str]  # source element -> target element
    relation_mappings: Dict[str, str]  # source relation -> target relation
    strength: float = 0.5

    def transfer(self, source_proposition: Proposition) -> Optional[Proposition]:
        """Transfer a proposition from source to target domain"""
        # Map predicate
        new_predicate = self.relation_mappings.get(
            source_proposition.predicate,
            source_proposition.predicate
        )

        # Map arguments
        new_args = []
        for arg in source_proposition.arguments:
            arg_str = str(arg)
            new_arg = self.mappings.get(arg_str, arg_str)
            new_args.append(new_arg)

        return Proposition(
            predicate=new_predicate,
            arguments=new_args,
            confidence=source_proposition.confidence * self.strength,
        )


@dataclass
class Pattern:
    """An abstract pattern detected in data"""
    pattern_type: str
    elements: List[Any]
    structure: str  # Description of the pattern structure
    confidence: float = 0.5
    examples: List[Any] = field(default_factory=list)


class KnowledgeBase:
    """
    Stores facts, rules, and relationships for logical reasoning.
    """

    def __init__(self):
        """Initialize empty knowledge base."""
        # Facts (ground propositions)
        self.facts: Set[Proposition] = set()

        # Rules
        self.rules: List[Rule] = []

        # Symbols
        self.symbols: Dict[str, Symbol] = {}

        # Relations between symbols
        self.relations: Dict[Tuple[str, str], List[Tuple[RelationType, float]]] = defaultdict(list)

        # Type hierarchy
        self.type_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # child -> parents

    def add_fact(self, proposition: Proposition) -> None:
        """Add a fact to the knowledge base."""
        proposition.truth_value = True
        self.facts.add(proposition)

        # Register symbols
        for arg in proposition.arguments:
            arg_str = str(arg)
            if arg_str not in self.symbols:
                self.symbols[arg_str] = Symbol(name=arg_str)

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base."""
        self.rules.append(rule)

    def add_relation(
        self,
        entity1: str,
        entity2: str,
        relation: RelationType,
        strength: float = 1.0,
    ) -> None:
        """Add a relation between entities."""
        self.relations[(entity1, entity2)].append((relation, strength))

        # Handle special relations
        if relation == RelationType.IS_A:
            self.type_hierarchy[entity1].add(entity2)

    def query(self, proposition: Proposition) -> Tuple[bool, float]:
        """
        Query whether a proposition is true.

        Returns (is_true, confidence).
        """
        # Direct lookup
        for fact in self.facts:
            if fact == proposition:
                return True, fact.confidence

        # Try to derive using rules
        for rule in self.rules:
            if self._matches_consequent(rule.consequent, proposition):
                # Check if antecedent is satisfied
                all_satisfied = True
                min_confidence = rule.confidence

                for ant in rule.antecedent:
                    # Substitute variables
                    substituted = self._substitute(ant, rule.consequent, proposition)
                    result, conf = self.query(substituted)
                    if not result:
                        all_satisfied = False
                        break
                    min_confidence = min(min_confidence, conf)

                if all_satisfied:
                    return True, min_confidence

        return False, 0.0

    def _matches_consequent(self, template: Proposition, target: Proposition) -> bool:
        """Check if target matches the consequent template."""
        return template.predicate == target.predicate and len(template.arguments) == len(target.arguments)

    def _substitute(
        self,
        template: Proposition,
        rule_consequent: Proposition,
        actual: Proposition,
    ) -> Proposition:
        """Substitute variables in template based on actual values."""
        substitution = {}
        for i, (t_arg, a_arg) in enumerate(zip(rule_consequent.arguments, actual.arguments)):
            substitution[str(t_arg)] = str(a_arg)

        new_args = []
        for arg in template.arguments:
            arg_str = str(arg)
            new_args.append(substitution.get(arg_str, arg_str))

        return Proposition(
            predicate=template.predicate,
            arguments=new_args,
        )

    def get_related(
        self,
        entity: str,
        relation: Optional[RelationType] = None,
    ) -> List[Tuple[str, RelationType, float]]:
        """Get entities related to the given entity."""
        results = []

        for (e1, e2), rels in self.relations.items():
            if e1 == entity:
                for rel, strength in rels:
                    if relation is None or rel == relation:
                        results.append((e2, rel, strength))
            elif e2 == entity:
                for rel, strength in rels:
                    if relation is None or rel == relation:
                        results.append((e1, rel, strength))

        return results

    def get_type_ancestors(self, entity: str) -> Set[str]:
        """Get all type ancestors of an entity."""
        ancestors = set()
        stack = list(self.type_hierarchy.get(entity, set()))

        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                stack.extend(self.type_hierarchy.get(parent, set()))

        return ancestors


class LogicEngine:
    """
    Propositional and predicate logic inference engine.
    """

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """Initialize logic engine."""
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()

        # Inference statistics
        self.inferences_made = 0
        self.rules_applied = 0

    def evaluate(
        self,
        expression: Union[Proposition, Tuple],
    ) -> Tuple[bool, float]:
        """
        Evaluate a logical expression.

        Expression can be:
        - A Proposition
        - A tuple: (operator, operand1, operand2) or (NOT, operand)
        """
        if isinstance(expression, Proposition):
            return self.kb.query(expression)

        if not isinstance(expression, tuple) or len(expression) < 2:
            return False, 0.0

        operator = expression[0]

        if operator == LogicOperator.NOT:
            result, conf = self.evaluate(expression[1])
            return not result, conf

        if len(expression) < 3:
            return False, 0.0

        left_result, left_conf = self.evaluate(expression[1])
        right_result, right_conf = self.evaluate(expression[2])
        min_conf = min(left_conf, right_conf)

        if operator == LogicOperator.AND:
            return left_result and right_result, min_conf
        elif operator == LogicOperator.OR:
            return left_result or right_result, max(left_conf, right_conf)
        elif operator == LogicOperator.IMPLIES:
            return (not left_result) or right_result, min_conf
        elif operator == LogicOperator.IFF:
            return left_result == right_result, min_conf
        elif operator == LogicOperator.XOR:
            return left_result != right_result, min_conf

        return False, 0.0

    def forward_chain(self, max_iterations: int = 100) -> List[Proposition]:
        """
        Apply forward chaining to derive new facts.

        Returns list of newly derived facts.
        """
        new_facts = []

        for iteration in range(max_iterations):
            derived_this_round = []

            for rule in self.kb.rules:
                # Find all ways to satisfy antecedent
                bindings = self._find_bindings(rule.antecedent)

                for binding in bindings:
                    # Create consequent with binding
                    new_prop = self._apply_binding(rule.consequent, binding)
                    new_prop.confidence = rule.confidence

                    # Check if already known
                    if new_prop not in self.kb.facts:
                        self.kb.facts.add(new_prop)
                        derived_this_round.append(new_prop)
                        rule.applications += 1
                        self.rules_applied += 1

            if not derived_this_round:
                break  # No new facts derived

            new_facts.extend(derived_this_round)
            self.inferences_made += len(derived_this_round)

        return new_facts

    def backward_chain(
        self,
        goal: Proposition,
        depth: int = 0,
        max_depth: int = 10,
    ) -> Tuple[bool, List[Proposition]]:
        """
        Use backward chaining to prove a goal.

        Returns (proven, proof_path).
        """
        if depth > max_depth:
            return False, []

        # Check if goal is already known
        if goal in self.kb.facts:
            return True, [goal]

        # Try each rule that could derive the goal
        for rule in self.kb.rules:
            if rule.consequent.predicate != goal.predicate:
                continue
            if len(rule.consequent.arguments) != len(goal.arguments):
                continue

            # Create binding from goal to rule
            binding = {}
            for i, (rule_arg, goal_arg) in enumerate(zip(rule.consequent.arguments, goal.arguments)):
                binding[str(rule_arg)] = str(goal_arg)

            # Try to prove all antecedents
            all_proven = True
            proof_path = [goal]

            for ant in rule.antecedent:
                subgoal = self._apply_binding(ant, binding)
                proven, subproof = self.backward_chain(subgoal, depth + 1, max_depth)

                if not proven:
                    all_proven = False
                    break

                proof_path.extend(subproof)

            if all_proven:
                self.inferences_made += 1
                return True, proof_path

        return False, []

    def _find_bindings(self, antecedent: List[Proposition]) -> List[Dict[str, str]]:
        """Find all variable bindings that satisfy the antecedent."""
        if not antecedent:
            return [{}]

        first_prop = antecedent[0]
        bindings = []

        # Find matching facts
        for fact in self.kb.facts:
            if fact.predicate != first_prop.predicate:
                continue
            if len(fact.arguments) != len(first_prop.arguments):
                continue

            # Create binding
            binding = {}
            match = True
            for i, (f_arg, p_arg) in enumerate(zip(fact.arguments, first_prop.arguments)):
                p_arg_str = str(p_arg)
                f_arg_str = str(f_arg)

                # Check if this is a variable (starts with uppercase or ?)
                if p_arg_str.startswith('?') or p_arg_str[0].isupper():
                    if p_arg_str in binding:
                        if binding[p_arg_str] != f_arg_str:
                            match = False
                            break
                    binding[p_arg_str] = f_arg_str
                elif p_arg_str != f_arg_str:
                    match = False
                    break

            if match:
                # Recursively find bindings for rest
                rest_bindings = self._find_bindings_with(antecedent[1:], binding)
                bindings.extend(rest_bindings)

        return bindings

    def _find_bindings_with(
        self,
        antecedent: List[Proposition],
        current_binding: Dict[str, str],
    ) -> List[Dict[str, str]]:
        """Find bindings given an existing partial binding."""
        if not antecedent:
            return [current_binding]

        first_prop = antecedent[0]
        # Apply current binding to first proposition
        bound_prop = self._apply_binding(first_prop, current_binding)

        bindings = []

        for fact in self.kb.facts:
            if fact.predicate != bound_prop.predicate:
                continue
            if len(fact.arguments) != len(bound_prop.arguments):
                continue

            new_binding = current_binding.copy()
            match = True

            for f_arg, p_arg in zip(fact.arguments, first_prop.arguments):
                p_arg_str = str(p_arg)
                f_arg_str = str(f_arg)

                if p_arg_str.startswith('?') or (p_arg_str[0].isupper() if p_arg_str else False):
                    if p_arg_str in new_binding:
                        if new_binding[p_arg_str] != f_arg_str:
                            match = False
                            break
                    new_binding[p_arg_str] = f_arg_str
                elif p_arg_str != f_arg_str:
                    match = False
                    break

            if match:
                rest_bindings = self._find_bindings_with(antecedent[1:], new_binding)
                bindings.extend(rest_bindings)

        return bindings

    def _apply_binding(self, prop: Proposition, binding: Dict[str, str]) -> Proposition:
        """Apply a variable binding to a proposition."""
        new_args = []
        for arg in prop.arguments:
            arg_str = str(arg)
            new_args.append(binding.get(arg_str, arg_str))

        return Proposition(
            predicate=prop.predicate,
            arguments=new_args,
            confidence=prop.confidence,
        )


class AnalogyEngine:
    """
    Detects and uses analogies for reasoning and transfer learning.
    """

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """Initialize analogy engine."""
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()

        # Known analogies
        self.analogies: List[Analogy] = []

        # Embedding dimension for structure mapping
        self.embedding_dim = 64

    def find_analogy(
        self,
        source_facts: List[Proposition],
        target_facts: List[Proposition],
        min_mapping_ratio: float = 0.5,
    ) -> Optional[Analogy]:
        """
        Find an analogy between source and target domains.

        Uses structure mapping theory - finds the mapping that preserves
        the most relational structure.
        """
        # Extract entities from each domain
        source_entities = set()
        target_entities = set()

        for fact in source_facts:
            for arg in fact.arguments:
                source_entities.add(str(arg))

        for fact in target_facts:
            for arg in fact.arguments:
                target_entities.add(str(arg))

        source_entities = list(source_entities)
        target_entities = list(target_entities)

        if not source_entities or not target_entities:
            return None

        # Extract relations
        source_relations = defaultdict(list)
        target_relations = defaultdict(list)

        for fact in source_facts:
            if len(fact.arguments) >= 2:
                source_relations[fact.predicate].append(
                    tuple(str(a) for a in fact.arguments)
                )

        for fact in target_facts:
            if len(fact.arguments) >= 2:
                target_relations[fact.predicate].append(
                    tuple(str(a) for a in fact.arguments)
                )

        # Find best entity mapping using structural consistency
        best_mapping = {}
        best_score = 0.0

        # Try greedy mapping based on relation structure
        for s_ent in source_entities:
            best_match = None
            best_match_score = 0.0

            for t_ent in target_entities:
                if t_ent in best_mapping.values():
                    continue

                # Score based on relation pattern similarity
                score = self._mapping_score(
                    s_ent, t_ent,
                    source_relations, target_relations,
                    best_mapping,
                )

                if score > best_match_score:
                    best_match = t_ent
                    best_match_score = score

            if best_match is not None:
                best_mapping[s_ent] = best_match
                best_score += best_match_score

        if len(best_mapping) < len(source_entities) * min_mapping_ratio:
            return None

        # Find relation mappings
        relation_mappings = {}
        for s_rel in source_relations:
            for t_rel in target_relations:
                if self._relations_align(
                    source_relations[s_rel],
                    target_relations[t_rel],
                    best_mapping,
                ):
                    relation_mappings[s_rel] = t_rel
                    break

        # Determine domain names
        source_domain = self._infer_domain_name(source_facts)
        target_domain = self._infer_domain_name(target_facts)

        analogy = Analogy(
            source_domain=source_domain,
            target_domain=target_domain,
            mappings=best_mapping,
            relation_mappings=relation_mappings,
            strength=best_score / max(len(source_entities), 1),
        )

        self.analogies.append(analogy)
        return analogy

    def _mapping_score(
        self,
        s_ent: str,
        t_ent: str,
        source_relations: Dict[str, List[Tuple]],
        target_relations: Dict[str, List[Tuple]],
        current_mapping: Dict[str, str],
    ) -> float:
        """Score how well s_ent maps to t_ent given current mappings."""
        score = 0.0

        for s_rel, s_tuples in source_relations.items():
            for t_rel, t_tuples in target_relations.items():
                for s_tuple in s_tuples:
                    if s_ent not in s_tuple:
                        continue

                    for t_tuple in t_tuples:
                        if t_ent not in t_tuple:
                            continue

                        # Check if positions match
                        s_pos = s_tuple.index(s_ent)
                        t_pos = t_tuple.index(t_ent) if t_ent in t_tuple else -1

                        if s_pos == t_pos:
                            # Check if other entities in mapping also align
                            aligned = True
                            for i, s_arg in enumerate(s_tuple):
                                if s_arg == s_ent:
                                    continue
                                if s_arg in current_mapping:
                                    if i < len(t_tuple) and current_mapping[s_arg] != t_tuple[i]:
                                        aligned = False
                                        break

                            if aligned:
                                score += 1.0

        return score

    def _relations_align(
        self,
        s_tuples: List[Tuple],
        t_tuples: List[Tuple],
        mapping: Dict[str, str],
    ) -> bool:
        """Check if source and target relations align under the mapping."""
        for s_tuple in s_tuples:
            mapped_tuple = tuple(mapping.get(s, s) for s in s_tuple)
            if mapped_tuple in t_tuples:
                return True
        return False

    def _infer_domain_name(self, facts: List[Proposition]) -> str:
        """Infer a domain name from facts."""
        # Use most common predicate
        pred_counts = defaultdict(int)
        for fact in facts:
            pred_counts[fact.predicate] += 1

        if pred_counts:
            return max(pred_counts, key=pred_counts.get) + "_domain"
        return "unknown_domain"

    def transfer_knowledge(
        self,
        analogy: Analogy,
        source_facts: List[Proposition],
    ) -> List[Proposition]:
        """
        Transfer facts from source to target domain using an analogy.
        """
        transferred = []

        for fact in source_facts:
            new_fact = analogy.transfer(fact)
            if new_fact is not None:
                transferred.append(new_fact)

        return transferred


class PatternDetector:
    """
    Detects abstract patterns in sequences and structures.
    """

    def __init__(self):
        """Initialize pattern detector."""
        # Known patterns
        self.patterns: List[Pattern] = []

        # Pattern templates
        self.templates = {
            'arithmetic': self._detect_arithmetic,
            'geometric': self._detect_geometric,
            'repetition': self._detect_repetition,
            'alternation': self._detect_alternation,
            'fibonacci': self._detect_fibonacci,
        }

    def detect_pattern(
        self,
        sequence: List[Any],
        pattern_type: Optional[str] = None,
    ) -> Optional[Pattern]:
        """
        Detect pattern in a sequence.

        If pattern_type is None, tries all known patterns.
        """
        if len(sequence) < 3:
            return None

        if pattern_type is not None:
            if pattern_type in self.templates:
                return self.templates[pattern_type](sequence)
            return None

        # Try all patterns
        for name, detector in self.templates.items():
            pattern = detector(sequence)
            if pattern is not None:
                return pattern

        return None

    def _detect_arithmetic(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect arithmetic sequence (constant difference)."""
        try:
            nums = [float(x) for x in sequence]
        except (ValueError, TypeError):
            return None

        if len(nums) < 2:
            return None

        # Check for constant difference
        diff = nums[1] - nums[0]
        is_arithmetic = all(
            abs((nums[i] - nums[i-1]) - diff) < 1e-6
            for i in range(2, len(nums))
        )

        if is_arithmetic:
            return Pattern(
                pattern_type='arithmetic',
                elements=nums,
                structure=f'a(n) = a(0) + n * {diff}',
                confidence=1.0,
                examples=sequence,
            )

        return None

    def _detect_geometric(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect geometric sequence (constant ratio)."""
        try:
            nums = [float(x) for x in sequence]
        except (ValueError, TypeError):
            return None

        if len(nums) < 2 or any(n == 0 for n in nums[:-1]):
            return None

        # Check for constant ratio
        ratio = nums[1] / nums[0]
        is_geometric = all(
            abs((nums[i] / nums[i-1]) - ratio) < 1e-6
            for i in range(2, len(nums))
        )

        if is_geometric:
            return Pattern(
                pattern_type='geometric',
                elements=nums,
                structure=f'a(n) = a(0) * {ratio}^n',
                confidence=1.0,
                examples=sequence,
            )

        return None

    def _detect_repetition(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect repeating pattern."""
        n = len(sequence)

        # Try different period lengths
        for period in range(1, n // 2 + 1):
            is_repeating = all(
                sequence[i] == sequence[i % period]
                for i in range(n)
            )

            if is_repeating:
                return Pattern(
                    pattern_type='repetition',
                    elements=sequence[:period],
                    structure=f'repeating with period {period}',
                    confidence=1.0,
                    examples=sequence,
                )

        return None

    def _detect_alternation(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect alternating pattern (A, B, A, B, ...)."""
        if len(sequence) < 2:
            return None

        a, b = sequence[0], sequence[1]
        if a == b:
            return None

        is_alternating = all(
            sequence[i] == (a if i % 2 == 0 else b)
            for i in range(len(sequence))
        )

        if is_alternating:
            return Pattern(
                pattern_type='alternation',
                elements=[a, b],
                structure=f'alternating between {a} and {b}',
                confidence=1.0,
                examples=sequence,
            )

        return None

    def _detect_fibonacci(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect Fibonacci-like pattern (each = sum of previous two)."""
        try:
            nums = [float(x) for x in sequence]
        except (ValueError, TypeError):
            return None

        if len(nums) < 3:
            return None

        is_fibonacci = all(
            abs(nums[i] - (nums[i-1] + nums[i-2])) < 1e-6
            for i in range(2, len(nums))
        )

        if is_fibonacci:
            return Pattern(
                pattern_type='fibonacci',
                elements=nums,
                structure='a(n) = a(n-1) + a(n-2)',
                confidence=1.0,
                examples=sequence,
            )

        return None

    def predict_next(self, sequence: List[Any], n: int = 1) -> List[Any]:
        """Predict the next n elements in a sequence."""
        pattern = self.detect_pattern(sequence)

        if pattern is None:
            return []

        predictions = []
        current = list(sequence)

        for _ in range(n):
            next_val = self._predict_one(pattern, current)
            if next_val is None:
                break
            predictions.append(next_val)
            current.append(next_val)

        return predictions

    def _predict_one(self, pattern: Pattern, sequence: List[Any]) -> Optional[Any]:
        """Predict the next element."""
        if pattern.pattern_type == 'arithmetic':
            diff = sequence[1] - sequence[0]
            return sequence[-1] + diff

        elif pattern.pattern_type == 'geometric':
            ratio = sequence[1] / sequence[0]
            return sequence[-1] * ratio

        elif pattern.pattern_type == 'repetition':
            period = len(pattern.elements)
            return pattern.elements[len(sequence) % period]

        elif pattern.pattern_type == 'alternation':
            return pattern.elements[len(sequence) % 2]

        elif pattern.pattern_type == 'fibonacci':
            return sequence[-1] + sequence[-2]

        return None


class RuleInducer:
    """
    Induces general rules from examples.
    """

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """Initialize rule inducer."""
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()

        # Induced rules
        self.induced_rules: List[Rule] = []

        # Rule confidence threshold
        self.min_confidence = 0.7

    def induce_from_examples(
        self,
        positive_examples: List[Proposition],
        negative_examples: Optional[List[Proposition]] = None,
        background: Optional[List[Proposition]] = None,
    ) -> List[Rule]:
        """
        Induce rules from positive and negative examples.

        Uses a simplified version of ILP (Inductive Logic Programming).
        """
        if negative_examples is None:
            negative_examples = []
        if background is None:
            background = list(self.kb.facts)

        induced = []

        # Group examples by predicate
        pred_examples = defaultdict(list)
        for ex in positive_examples:
            pred_examples[ex.predicate].append(ex)

        for predicate, examples in pred_examples.items():
            # Find common patterns in arguments
            if len(examples) < 2:
                continue

            # Extract argument patterns
            arg_patterns = self._extract_patterns(examples)

            # Try to find supporting conditions from background
            conditions = self._find_conditions(examples, background)

            if conditions:
                # Create rule
                rule = Rule(
                    name=f"induced_{predicate}",
                    antecedent=conditions,
                    consequent=Proposition(
                        predicate=predicate,
                        arguments=arg_patterns,
                    ),
                    confidence=self._compute_confidence(
                        conditions, positive_examples, negative_examples
                    ),
                )

                if rule.confidence >= self.min_confidence:
                    induced.append(rule)
                    self.induced_rules.append(rule)

        return induced

    def _extract_patterns(self, examples: List[Proposition]) -> List[str]:
        """Extract argument patterns from examples."""
        if not examples:
            return []

        num_args = len(examples[0].arguments)
        patterns = []

        for i in range(num_args):
            # Check if all examples have the same value
            values = set(str(ex.arguments[i]) for ex in examples)
            if len(values) == 1:
                patterns.append(list(values)[0])
            else:
                patterns.append(f"?X{i}")  # Variable

        return patterns

    def _find_conditions(
        self,
        examples: List[Proposition],
        background: List[Proposition],
    ) -> List[Proposition]:
        """Find conditions that explain the examples."""
        conditions = []

        # Find background facts that commonly co-occur with examples
        entity_facts = defaultdict(list)

        for fact in background:
            for arg in fact.arguments:
                entity_facts[str(arg)].append(fact)

        # Find shared conditions across examples
        common_predicates = None

        for ex in examples:
            ex_predicates = set()
            for arg in ex.arguments:
                for fact in entity_facts.get(str(arg), []):
                    ex_predicates.add(fact.predicate)

            if common_predicates is None:
                common_predicates = ex_predicates
            else:
                common_predicates &= ex_predicates

        # Create condition propositions
        if common_predicates:
            for pred in list(common_predicates)[:3]:  # Limit conditions
                conditions.append(Proposition(
                    predicate=pred,
                    arguments=["?X"],  # Generalized
                ))

        return conditions

    def _compute_confidence(
        self,
        conditions: List[Proposition],
        positive: List[Proposition],
        negative: List[Proposition],
    ) -> float:
        """Compute rule confidence."""
        if not positive:
            return 0.0

        # Simple coverage-based confidence
        # In a full system, would check which examples the rule covers
        coverage = len(positive) / (len(positive) + len(negative) + 1)
        return coverage


class AbstractReasoner:
    """
    High-level abstract reasoning system that integrates all components.

    Provides:
    - Logical reasoning
    - Analogical reasoning
    - Pattern detection
    - Rule induction
    - Compositional reasoning
    """

    def __init__(self):
        """Initialize abstract reasoner."""
        self.kb = KnowledgeBase()
        self.logic_engine = LogicEngine(self.kb)
        self.analogy_engine = AnalogyEngine(self.kb)
        self.pattern_detector = PatternDetector()
        self.rule_inducer = RuleInducer(self.kb)

        # Statistics
        self.queries_answered = 0
        self.analogies_found = 0
        self.patterns_detected = 0
        self.rules_induced = 0

    def add_knowledge(
        self,
        facts: Optional[List[Proposition]] = None,
        rules: Optional[List[Rule]] = None,
        relations: Optional[List[Tuple[str, str, RelationType]]] = None,
    ) -> None:
        """Add knowledge to the reasoner."""
        if facts:
            for fact in facts:
                self.kb.add_fact(fact)

        if rules:
            for rule in rules:
                self.kb.add_rule(rule)

        if relations:
            for e1, e2, rel in relations:
                self.kb.add_relation(e1, e2, rel)

    def query(self, proposition: Proposition) -> Tuple[bool, float, List[Proposition]]:
        """
        Query whether a proposition is true.

        Returns (is_true, confidence, proof_path).
        """
        self.queries_answered += 1

        # Try direct query
        result, confidence = self.kb.query(proposition)
        if result:
            return True, confidence, [proposition]

        # Try backward chaining
        proven, proof = self.logic_engine.backward_chain(proposition)
        if proven:
            return True, 0.8, proof

        # Try forward chaining first
        new_facts = self.logic_engine.forward_chain(max_iterations=10)
        if proposition in self.kb.facts:
            return True, 0.7, [proposition]

        return False, 0.0, []

    def find_analogy(
        self,
        source: List[Proposition],
        target: List[Proposition],
    ) -> Optional[Analogy]:
        """Find an analogy between source and target."""
        analogy = self.analogy_engine.find_analogy(source, target)
        if analogy:
            self.analogies_found += 1
        return analogy

    def detect_pattern(self, sequence: List[Any]) -> Optional[Pattern]:
        """Detect a pattern in a sequence."""
        pattern = self.pattern_detector.detect_pattern(sequence)
        if pattern:
            self.patterns_detected += 1
        return pattern

    def predict_sequence(self, sequence: List[Any], n: int = 1) -> List[Any]:
        """Predict next elements in a sequence."""
        return self.pattern_detector.predict_next(sequence, n)

    def induce_rules(
        self,
        positive: List[Proposition],
        negative: Optional[List[Proposition]] = None,
    ) -> List[Rule]:
        """Induce rules from examples."""
        rules = self.rule_inducer.induce_from_examples(positive, negative)
        self.rules_induced += len(rules)

        # Add induced rules to knowledge base
        for rule in rules:
            self.kb.add_rule(rule)

        return rules

    def reason_by_analogy(
        self,
        query: Proposition,
        source_domain: List[Proposition],
        target_domain: List[Proposition],
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Answer a query using analogical reasoning.

        Returns (answer, confidence, explanation).
        """
        # Find analogy
        analogy = self.find_analogy(source_domain, target_domain)
        if analogy is None:
            return None

        # Check if query predicate maps to something in source
        inverse_rel_map = {v: k for k, v in analogy.relation_mappings.items()}
        inverse_ent_map = {v: k for k, v in analogy.mappings.items()}

        # Create source query
        source_pred = inverse_rel_map.get(query.predicate, query.predicate)
        source_args = [
            inverse_ent_map.get(str(a), str(a))
            for a in query.arguments
        ]
        source_query = Proposition(predicate=source_pred, arguments=source_args)

        # Check source query
        result, confidence = self.kb.query(source_query)

        explanation = (
            f"By analogy: {source_query} in source maps to {query} in target. "
            f"Analogy strength: {analogy.strength:.2f}"
        )

        return result, confidence * analogy.strength, explanation

    def compose_concepts(
        self,
        concepts: List[str],
        relation: str = "combined",
    ) -> Symbol:
        """
        Compose multiple concepts into a new combined concept.

        This enables compositional reasoning - combining simple
        concepts to form complex ones.
        """
        # Create composite name
        composite_name = f"{relation}({', '.join(concepts)})"

        # Create new symbol
        composite = Symbol(
            name=composite_name,
            symbol_type="composite",
            properties={
                'components': concepts,
                'relation': relation,
            },
        )

        # Compute embedding as combination of component embeddings
        embeddings = []
        for concept in concepts:
            if concept in self.kb.symbols:
                sym = self.kb.symbols[concept]
                if sym.embedding is not None:
                    embeddings.append(sym.embedding)

        if embeddings:
            # Simple average (could use more sophisticated composition)
            composite.embedding = np.mean(embeddings, axis=0)

        self.kb.symbols[composite_name] = composite

        return composite

    def get_related_concepts(
        self,
        concept: str,
        relation: Optional[RelationType] = None,
        max_results: int = 10,
    ) -> List[Tuple[str, RelationType, float]]:
        """Get concepts related to the given concept."""
        return self.kb.get_related(concept, relation)[:max_results]

    def explain_reasoning(
        self,
        query: Proposition,
        result: bool,
        proof: List[Proposition],
    ) -> str:
        """Generate a natural language explanation of reasoning."""
        if not result:
            return f"Could not prove {query}"

        if len(proof) == 1:
            return f"{query} is a known fact."

        # Build explanation
        lines = [f"To prove {query}:"]

        for i, step in enumerate(proof[1:], 1):
            lines.append(f"  {i}. {step}")

        lines.append(f"Therefore, {query} is true.")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        return {
            'queries_answered': self.queries_answered,
            'analogies_found': self.analogies_found,
            'patterns_detected': self.patterns_detected,
            'rules_induced': self.rules_induced,
            'total_facts': len(self.kb.facts),
            'total_rules': len(self.kb.rules),
            'total_symbols': len(self.kb.symbols),
            'inferences_made': self.logic_engine.inferences_made,
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the reasoner state."""
        return {
            'facts': [
                {
                    'predicate': f.predicate,
                    'arguments': [str(a) for a in f.arguments],
                    'confidence': f.confidence,
                }
                for f in self.kb.facts
            ],
            'rules': [
                {
                    'name': r.name,
                    'antecedent': [
                        {'predicate': p.predicate, 'arguments': [str(a) for a in p.arguments]}
                        for p in r.antecedent
                    ],
                    'consequent': {
                        'predicate': r.consequent.predicate,
                        'arguments': [str(a) for a in r.consequent.arguments],
                    },
                    'confidence': r.confidence,
                }
                for r in self.kb.rules
            ],
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AbstractReasoner':
        """Deserialize a reasoner from saved state."""
        reasoner = cls()

        # Restore facts
        for fact_data in data.get('facts', []):
            fact = Proposition(
                predicate=fact_data['predicate'],
                arguments=fact_data['arguments'],
                confidence=fact_data.get('confidence', 1.0),
            )
            reasoner.kb.add_fact(fact)

        # Restore rules
        for rule_data in data.get('rules', []):
            antecedent = [
                Proposition(predicate=p['predicate'], arguments=p['arguments'])
                for p in rule_data['antecedent']
            ]
            consequent = Proposition(
                predicate=rule_data['consequent']['predicate'],
                arguments=rule_data['consequent']['arguments'],
            )
            rule = Rule(
                name=rule_data['name'],
                antecedent=antecedent,
                consequent=consequent,
                confidence=rule_data.get('confidence', 1.0),
            )
            reasoner.kb.add_rule(rule)

        return reasoner
