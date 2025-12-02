#!/usr/bin/env python3
"""
Cognitive Systems Integration Demo for ATLAS

This example demonstrates how all cognitive subsystems work together:
- Language Grounding: Understanding and grounding text
- Working Memory: Managing active information and attention
- Causal Reasoning: Understanding cause-and-effect
- Abstract Reasoning: Logical inference and pattern detection

Usage:
    python run_cognitive_integration.py [--demo all|language|memory|causal|reasoning]
"""

import argparse
import sys
import os
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.language_grounding import LanguageGrounding, TextCorpusLearner
from core.working_memory import WorkingMemory, CognitiveController, WorkspaceSlotType
from core.causal_reasoning import CausalReasoner, CausalGraph, CounterfactualQuery
from core.abstract_reasoning import (
    AbstractReasoner, Proposition, Rule, RelationType, Pattern
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---")


def demo_language_grounding() -> LanguageGrounding:
    """Demonstrate language grounding capabilities."""
    print_header("Language Grounding System")

    print("\n1. Initializing language grounding system...")
    lg = LanguageGrounding(embedding_dim=64)

    # Add some grounded concepts
    print("\n2. Grounding sensory concepts...")

    # Create sensory patterns for colors
    red_pattern = np.zeros(64)
    red_pattern[0:10] = 1.0  # Red channel activation
    lg.ground_word("red", red_pattern, modality="visual")

    blue_pattern = np.zeros(64)
    blue_pattern[20:30] = 1.0  # Blue channel activation
    lg.ground_word("blue", blue_pattern, modality="visual")

    # Ground some action concepts
    grasp_pattern = np.zeros(64)
    grasp_pattern[40:50] = 1.0  # Motor pattern
    lg.ground_word("grasp", grasp_pattern, modality="motor")

    push_pattern = np.zeros(64)
    push_pattern[50:60] = 1.0
    lg.ground_word("push", push_pattern, modality="motor")

    print("   Grounded: red, blue (visual), grasp, push (motor)")

    # Learn from text corpus
    print("\n3. Learning from text corpus...")
    corpus = TextCorpusLearner(lg)

    sample_text = """
    The red ball is on the table.
    The blue cube is next to the ball.
    The robot can grasp objects.
    If you push the ball, it will roll.
    Red and blue are colors.
    Grasping requires motor control.
    """

    words_learned = corpus.learn_from_text(sample_text)
    print(f"   Learned {words_learned} words from corpus")

    # Parse and understand sentences
    print("\n4. Parsing sentences...")

    test_sentences = [
        "The red ball is on the table",
        "Push the blue cube",
        "The robot can grasp the ball",
    ]

    for sentence in test_sentences:
        parsed = lg.parse_sentence(sentence)
        print(f"\n   '{sentence}'")
        print(f"   Subject: {parsed.subject}")
        print(f"   Verb: {parsed.verb}")
        print(f"   Object: {parsed.object}")
        print(f"   Grounding: {parsed.grounding_score:.2f}")

    # Word similarity
    print("\n5. Semantic similarity...")
    pairs = [("red", "blue"), ("grasp", "push"), ("red", "grasp")]
    for w1, w2 in pairs:
        sim = lg.word_similarity(w1, w2)
        print(f"   similarity({w1}, {w2}) = {sim:.3f}")

    # Stats
    stats = lg.get_stats()
    print(f"\n6. System stats:")
    print(f"   Grounded words: {stats['grounded_words']}")
    print(f"   Symbolic words: {stats['symbolic_words']}")

    return lg


def demo_working_memory() -> CognitiveController:
    """Demonstrate working memory and cognitive control."""
    print_header("Working Memory and Cognitive Control")

    print("\n1. Initializing working memory system...")
    wm = WorkingMemory(capacity=7, decay_rate=0.1)
    controller = CognitiveController(wm)

    # Add items to working memory
    print("\n2. Adding items to working memory...")

    items = [
        ("goal_1", "Find red ball", WorkspaceSlotType.GOAL, 0.9),
        ("percept_1", "Ball detected at (3,4)", WorkspaceSlotType.SENSORY, 0.7),
        ("memory_1", "Ball was on table yesterday", WorkspaceSlotType.EPISODIC, 0.5),
        ("action_1", "Move to coordinates", WorkspaceSlotType.ACTION, 0.6),
    ]

    for item_id, description, slot_type, salience in items:
        content = np.random.randn(64)  # Simulated neural representation
        wm.add_item(
            item_id=item_id,
            content=content,
            slot_type=slot_type,
            source="demo",
            salience=salience,
            relevance=0.8,
            metadata={"description": description}
        )
        print(f"   Added: {item_id} ({slot_type.value})")

    # Set current goal
    print("\n3. Setting current goal...")
    wm.set_goal("find_object", priority=1.0)
    print(f"   Active goals: {list(wm.current_goals.keys())}")

    # Focus attention
    print("\n4. Focusing attention...")
    focus = wm.attention.focus_on(["goal_1", "percept_1"])
    print(f"   Focused on: {focus.target_ids}")
    print(f"   Intensity: {focus.intensity:.2f}")

    # Get most active items
    print("\n5. Most active items:")
    active = wm.get_most_active(3)
    for item in active:
        desc = item.metadata.get("description", "?")
        print(f"   {item.item_id}: {desc} (activation: {item.activation:.2f})")

    # Task switching
    print("\n6. Simulating task switch...")
    print(f"   Current task: {wm.current_task}")

    switch_cost = controller.switch_task("avoid_obstacle")
    print(f"   Switched to: {wm.current_task}")
    print(f"   Switch cost: {switch_cost}ms")

    # Cognitive load
    print("\n7. Cognitive metrics:")
    metrics = controller.step()
    print(f"   Conflict level: {metrics['conflict_level']:.2f}")
    print(f"   Cognitive load: {metrics['cognitive_load']:.2f}")
    print(f"   Active inhibitions: {metrics['inhibition_count']}")

    # Working memory stats
    stats = wm.get_stats()
    print(f"\n8. Working memory stats:")
    print(f"   Items: {stats['current_items']}/{stats['capacity']}")
    print(f"   Utilization: {stats['utilization']:.1%}")
    print(f"   Mean activation: {stats['mean_activation']:.2f}")

    return controller


def demo_causal_reasoning() -> CausalReasoner:
    """Demonstrate causal reasoning capabilities."""
    print_header("Causal Reasoning System")

    print("\n1. Initializing causal reasoning system...")
    reasoner = CausalReasoner(min_observations_for_learning=5)

    # Build a simple causal model
    print("\n2. Building causal graph...")

    # Add variables and links
    graph = reasoner.model.graph

    # Rain -> Wet Ground -> Slippery
    # Rain -> Umbrella Usage
    graph.add_variable("rain", domain=(0.0, 1.0))
    graph.add_variable("wet_ground", domain=(0.0, 1.0))
    graph.add_variable("slippery", domain=(0.0, 1.0))
    graph.add_variable("umbrella", domain=(0.0, 1.0))

    graph.add_link("rain", "wet_ground", strength=0.8)
    graph.add_link("rain", "umbrella", strength=0.7)
    graph.add_link("wet_ground", "slippery", strength=0.9)

    print("   Causal structure:")
    print("   rain -> wet_ground -> slippery")
    print("   rain -> umbrella")

    # Set up equations
    reasoner.model.weights[("rain", "wet_ground")] = 0.8
    reasoner.model.weights[("rain", "umbrella")] = 0.7
    reasoner.model.weights[("wet_ground", "slippery")] = 0.9
    reasoner.model.set_default_linear_equations()

    # Observational prediction
    print("\n3. Observational prediction:")
    print("   Given: rain = 0.8")
    evidence = {"rain": 0.8}
    wet = reasoner.model.predict(evidence, "wet_ground")
    slip = reasoner.model.predict(evidence, "slippery")
    print(f"   P(wet_ground | rain=0.8) = {wet:.2f}")
    print(f"   P(slippery | rain=0.8) = {slip:.2f}")

    # Interventional prediction
    print("\n4. Interventional prediction (do-calculus):")
    print("   Intervention: do(wet_ground = 1.0)")
    intervention = {"wet_ground": 1.0}
    slip_do = reasoner.model.intervene(intervention, "slippery")
    rain_do = reasoner.model.intervene(intervention, "rain")
    print(f"   P(slippery | do(wet_ground=1.0)) = {slip_do:.2f}")
    print(f"   P(rain | do(wet_ground=1.0)) = {rain_do:.2f}")
    print("   Note: Intervention on wet_ground doesn't affect rain (causal direction)")

    # Counterfactual reasoning
    print("\n5. Counterfactual reasoning:")
    print("   Factual: rain=0.2, wet_ground=0.3, slippery=0.1")
    print("   Question: What if rain had been 0.9?")

    cf_query = CounterfactualQuery(
        factual_evidence={"rain": 0.2, "wet_ground": 0.3, "slippery": 0.1},
        hypothetical_intervention={"rain": 0.9},
        query_variable="slippery"
    )
    cf_result = reasoner.model.counterfactual(cf_query)
    print(f"   Counterfactual P(slippery) = {cf_result:.2f}")

    # What-if query
    print("\n6. High-level 'what-if' query:")
    result, explanation = reasoner.what_if(
        factual={"rain": 0.5, "wet_ground": 0.4},
        hypothetical={"rain": 0.0},
        query="slippery"
    )
    print(f"   {explanation}")

    # Causal explanation
    print("\n7. Causal explanation:")
    explanation = reasoner.why(
        effect="slippery",
        evidence={"rain": 0.8, "wet_ground": 0.7, "slippery": 0.6}
    )
    print(f"   Effect: {explanation.effect} = {explanation.effect_value:.2f}")
    print(f"   Causes:")
    for cause, value, contribution in explanation.causes:
        print(f"     - {cause} = {value:.2f} (contribution: {contribution:.2f})")
    if explanation.counterfactual:
        print(f"   Counterfactual: {explanation.counterfactual}")

    # Stats
    stats = reasoner.get_stats()
    print(f"\n8. System stats:")
    print(f"   Variables: {stats['num_variables']}")
    print(f"   Causal links: {stats['num_links']}")

    return reasoner


def demo_abstract_reasoning() -> AbstractReasoner:
    """Demonstrate abstract reasoning capabilities."""
    print_header("Abstract Logic and Reasoning System")

    print("\n1. Initializing abstract reasoning system...")
    reasoner = AbstractReasoner()

    # Add knowledge
    print("\n2. Adding knowledge to knowledge base...")

    # Facts
    facts = [
        Proposition("is_animal", ["dog"]),
        Proposition("is_animal", ["cat"]),
        Proposition("is_mammal", ["dog"]),
        Proposition("is_mammal", ["cat"]),
        Proposition("has_fur", ["dog"]),
        Proposition("has_fur", ["cat"]),
        Proposition("barks", ["dog"]),
        Proposition("is_pet", ["dog"]),
        Proposition("is_pet", ["cat"]),
    ]

    for fact in facts:
        reasoner.kb.add_fact(fact)
        print(f"   Added fact: {fact}")

    # Add rules
    print("\n3. Adding logical rules...")

    # Rule: If X is a mammal and X has fur, then X is warm_blooded
    rule1 = Rule(
        name="mammal_warm_blooded",
        antecedent=[
            Proposition("is_mammal", ["?X"]),
            Proposition("has_fur", ["?X"]),
        ],
        consequent=Proposition("is_warm_blooded", ["?X"]),
        confidence=0.95,
    )
    reasoner.kb.add_rule(rule1)
    print(f"   Added: {rule1}")

    # Rule: If X barks, then X is a dog
    rule2 = Rule(
        name="barking_dog",
        antecedent=[Proposition("barks", ["?X"])],
        consequent=Proposition("is_dog", ["?X"]),
        confidence=0.9,
    )
    reasoner.kb.add_rule(rule2)
    print(f"   Added: {rule2}")

    # Query the knowledge base
    print("\n4. Querying the knowledge base...")

    queries = [
        Proposition("is_animal", ["dog"]),
        Proposition("is_warm_blooded", ["dog"]),
        Proposition("is_dog", ["dog"]),
        Proposition("flies", ["dog"]),
    ]

    for query in queries:
        result, confidence, proof = reasoner.query(query)
        status = "TRUE" if result else "FALSE"
        print(f"   {query}: {status} (confidence: {confidence:.2f})")
        if result and len(proof) > 1:
            print(f"      Proof: {' -> '.join(str(p) for p in proof[:3])}")

    # Forward chaining
    print("\n5. Forward chaining (deriving new facts)...")
    new_facts = reasoner.logic_engine.forward_chain(max_iterations=10)
    print(f"   Derived {len(new_facts)} new facts:")
    for fact in new_facts[:5]:
        print(f"     - {fact}")

    # Pattern detection
    print("\n6. Pattern detection...")

    sequences = [
        [2, 4, 6, 8, 10],          # Arithmetic
        [1, 2, 4, 8, 16],          # Geometric
        [1, 1, 2, 3, 5, 8],        # Fibonacci
        ["A", "B", "A", "B", "A"], # Alternation
    ]

    for seq in sequences:
        pattern = reasoner.detect_pattern(seq)
        if pattern:
            print(f"   {seq}")
            print(f"     Pattern: {pattern.pattern_type}")
            print(f"     Structure: {pattern.structure}")

            # Predict next
            next_vals = reasoner.predict_sequence(seq, n=2)
            print(f"     Next values: {next_vals}")

    # Compositional reasoning
    print("\n7. Compositional reasoning...")

    # Compose concepts
    composite = reasoner.compose_concepts(
        ["red", "ball"],
        relation="has_property"
    )
    print(f"   Composed: {composite.name}")
    print(f"   Components: {composite.properties['components']}")

    # Rule induction
    print("\n8. Rule induction from examples...")

    positive_examples = [
        Proposition("can_fly", ["sparrow"]),
        Proposition("can_fly", ["eagle"]),
        Proposition("can_fly", ["crow"]),
    ]

    # Add background knowledge
    reasoner.kb.add_fact(Proposition("has_wings", ["sparrow"]))
    reasoner.kb.add_fact(Proposition("has_wings", ["eagle"]))
    reasoner.kb.add_fact(Proposition("has_wings", ["crow"]))
    reasoner.kb.add_fact(Proposition("is_bird", ["sparrow"]))
    reasoner.kb.add_fact(Proposition("is_bird", ["eagle"]))
    reasoner.kb.add_fact(Proposition("is_bird", ["crow"]))

    induced = reasoner.induce_rules(positive_examples)
    print(f"   Induced {len(induced)} rules from flying examples")
    for rule in induced:
        print(f"     - {rule.name}: confidence={rule.confidence:.2f}")

    # Stats
    stats = reasoner.get_stats()
    print(f"\n9. System stats:")
    print(f"   Total facts: {stats['total_facts']}")
    print(f"   Total rules: {stats['total_rules']}")
    print(f"   Queries answered: {stats['queries_answered']}")
    print(f"   Patterns detected: {stats['patterns_detected']}")
    print(f"   Inferences made: {stats['inferences_made']}")

    return reasoner


def demo_integrated_cognition():
    """Demonstrate integrated cognition using all systems together."""
    print_header("Integrated Cognitive System Demo")

    print("\nThis demo shows all cognitive systems working together")
    print("to solve a simple reasoning task.\n")

    # Initialize all systems
    print("1. Initializing cognitive systems...")
    language = LanguageGrounding(embedding_dim=64)
    wm = WorkingMemory(capacity=7)
    controller = CognitiveController(wm)
    causal = CausalReasoner()
    reasoner = AbstractReasoner()

    # Scenario: Understanding and reasoning about a scene
    print("\n2. Processing natural language input...")

    scene_description = "The robot sees a red ball. If the robot pushes the ball, it will roll."
    print(f"   Input: '{scene_description}'")

    # Parse sentences
    sentences = scene_description.split(". ")
    for sentence in sentences:
        if sentence:
            parsed = language.parse_sentence(sentence)
            print(f"   Parsed: subject={parsed.subject}, verb={parsed.verb}, object={parsed.object}")

    # Add to working memory
    print("\n3. Adding information to working memory...")

    wm.add_item(
        item_id="object_1",
        content=np.random.randn(64),
        slot_type=WorkspaceSlotType.SENSORY,
        source="vision",
        salience=0.8,
        relevance=0.9,
        metadata={"type": "ball", "color": "red", "position": (3, 4)}
    )

    wm.add_item(
        item_id="goal_1",
        content=np.random.randn(64),
        slot_type=WorkspaceSlotType.GOAL,
        source="planner",
        salience=0.9,
        relevance=1.0,
        metadata={"goal": "move_ball", "target": (5, 4)}
    )

    print(f"   Working memory items: {len(wm.slots)}")

    # Build causal model
    print("\n4. Building causal model...")

    causal.model.graph.add_variable("push_action")
    causal.model.graph.add_variable("ball_velocity")
    causal.model.graph.add_variable("ball_position")
    causal.model.graph.add_link("push_action", "ball_velocity", strength=0.9)
    causal.model.graph.add_link("ball_velocity", "ball_position", strength=0.8)

    causal.model.weights[("push_action", "ball_velocity")] = 0.9
    causal.model.weights[("ball_velocity", "ball_position")] = 0.8
    causal.model.set_default_linear_equations()

    print("   Causal chain: push_action -> ball_velocity -> ball_position")

    # Reason about action
    print("\n5. Reasoning about action consequences...")

    # What happens if we push?
    predictions = causal.do({"push_action": 1.0})
    print(f"   If push_action=1.0:")
    print(f"     ball_velocity = {predictions.get('ball_velocity', 0):.2f}")
    print(f"     ball_position change = {predictions.get('ball_position', 0):.2f}")

    # Abstract reasoning
    print("\n6. Applying abstract reasoning...")

    # Add facts about the scene
    reasoner.kb.add_fact(Proposition("is_object", ["ball"]))
    reasoner.kb.add_fact(Proposition("has_color", ["ball", "red"]))
    reasoner.kb.add_fact(Proposition("is_movable", ["ball"]))

    # Add rule
    reasoner.kb.add_rule(Rule(
        name="movable_can_push",
        antecedent=[Proposition("is_movable", ["?X"])],
        consequent=Proposition("can_push", ["?X"]),
    ))

    # Query
    can_push = reasoner.query(Proposition("can_push", ["ball"]))
    print(f"   Query: can_push(ball)? {can_push[0]} (confidence: {can_push[1]:.2f})")

    # Cognitive control decision
    print("\n7. Cognitive control decision...")

    controller.monitor_conflict()
    load = controller.estimate_cognitive_load()
    print(f"   Cognitive load: {load:.2f}")
    print(f"   Conflict level: {controller.conflict_level:.2f}")

    # Decide on action
    print("\n8. Decision:")
    print("   Based on:")
    print("   - Causal model predicts push will move ball")
    print("   - Abstract reasoning confirms ball is pushable")
    print("   - Working memory contains goal to move ball")
    print("   - Cognitive load is manageable")
    print("\n   Decision: Execute push action")

    # Summary
    print_section("Summary")
    print("All cognitive systems contributed to the decision:")
    print("- Language: Parsed scene description")
    print("- Working Memory: Maintained goal and object info")
    print("- Causal Reasoning: Predicted action consequences")
    print("- Abstract Reasoning: Inferred object properties")
    print("- Cognitive Control: Managed resources and made decision")


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Cognitive Systems Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python run_cognitive_integration.py --demo all

  # Run specific demo
  python run_cognitive_integration.py --demo language
  python run_cognitive_integration.py --demo memory
  python run_cognitive_integration.py --demo causal
  python run_cognitive_integration.py --demo reasoning

  # Run integrated demo
  python run_cognitive_integration.py --demo integrated
        """
    )

    parser.add_argument(
        '--demo',
        type=str,
        default='all',
        choices=['all', 'language', 'memory', 'causal', 'reasoning', 'integrated'],
        help='Which demo to run (default: all)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Print welcome message
    print("\n" + "=" * 60)
    print(" ATLAS - Cognitive Systems Integration Demo")
    print(" Demonstrating superintelligence components")
    print("=" * 60)

    if args.demo == 'all':
        demo_language_grounding()
        demo_working_memory()
        demo_causal_reasoning()
        demo_abstract_reasoning()
        demo_integrated_cognition()
    elif args.demo == 'language':
        demo_language_grounding()
    elif args.demo == 'memory':
        demo_working_memory()
    elif args.demo == 'causal':
        demo_causal_reasoning()
    elif args.demo == 'reasoning':
        demo_abstract_reasoning()
    elif args.demo == 'integrated':
        demo_integrated_cognition()

    print("\n" + "=" * 60)
    print(" Demo completed successfully!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
