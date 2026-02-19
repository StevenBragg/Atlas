#!/usr/bin/env python3
"""
Reasoning Systems Demo for Atlas

This example demonstrates Atlas's reasoning capabilities:
- Causal Reasoning: Understanding cause-effect relationships
- Abstract Reasoning: Logic, analogy, and pattern recognition

Usage:
    python reasoning_demo.py
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_organizing_av_system.core.causal_reasoning import (
    CausalReasoner, CausalGraph, CausalRelationType, CounterfactualQuery
)
from self_organizing_av_system.core.abstract_reasoning import (
    AbstractReasoner, Proposition, Rule, RelationType, KnowledgeBase
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---")


def demo_causal_reasoning():
    """Demonstrate causal reasoning capabilities."""
    print_header("Causal Reasoning Demo")
    print("(Pearl's Causal Hierarchy: Association → Intervention → Counterfactual)")
    
    # Initialize causal reasoner
    print_section("Initializing Causal Reasoner")
    reasoner = CausalReasoner()
    print("✓ Created causal reasoning engine")
    
    # Build a causal graph for weather-garden system
    print_section("Building Causal Graph: Weather-Garden System")
    
    # Add variables
    variables = [
        ("rain", True, True),      # Observable, manipulable
        ("sunlight", True, False), # Observable, not manipulable
        ("soil_moisture", True, True),
        ("plant_growth", True, False),
        ("weed_growth", True, False),
    ]
    
    for name, obs, manip in variables:
        reasoner.graph.add_variable(name, is_observable=obs, is_manipulable=manip)
        print(f"  ✓ Added variable: {name}")
    
    # Add causal links
    links = [
        ("rain", "soil_moisture", CausalRelationType.CAUSES, 0.8),
        ("sunlight", "soil_moisture", CausalRelationType.PREVENTS, 0.6),
        ("soil_moisture", "plant_growth", CausalRelationType.ENABLES, 0.7),
        ("sunlight", "plant_growth", CausalRelationType.CAUSES, 0.9),
        ("rain", "weed_growth", CausalRelationType.CAUSES, 0.5),
        ("soil_moisture", "weed_growth", CausalRelationType.ENABLES, 0.6),
    ]
    
    for cause, effect, rel_type, strength in links:
        reasoner.graph.add_link(cause, effect, rel_type, strength)
        print(f"  ✓ {cause} {rel_type.value} {effect} (strength={strength})")
    
    # Level 1: Observational (Association)
    print_section("Level 1: Observational Reasoning (P(Y|X))")
    
    print("Observations: What do we see when it rains?")
    effects = reasoner.predict_effects("rain")
    for effect, prob in effects[:4]:
        print(f"  P({effect}|rain) = {prob:.2f}")
    
    print("\nFinding causes of plant growth:")
    causes = reasoner.find_causes("plant_growth")
    for cause, strength in causes[:4]:
        print(f"  {cause} contributes with strength {strength:.2f}")
    
    # Level 2: Interventional (Do-calculus)
    print_section("Level 2: Interventional Reasoning (P(Y|do(X)))")
    
    print("Intervention: What if we water the soil (do(soil_moisture=high))?")
    result = reasoner.intervene({"soil_moisture": 0.9})
    print(f"  Plant growth after intervention: {result.get('plant_growth', 0):.2f}")
    print(f"  Weed growth after intervention: {result.get('weed_growth', 0):.2f}")
    
    print("\nIntervention vs Observation:")
    print("  P(plant_growth|rain) - observing rain")
    obs_result = reasoner.predict_effects("rain")
    for effect, prob in obs_result[:2]:
        print(f"    {effect}: {prob:.2f}")
    
    print("  P(plant_growth|do(rain)) - forcing rain")
    int_result = reasoner.intervene({"rain": 1.0})
    for var in ["plant_growth", "weed_growth"]:
        print(f"    {var}: {int_result.get(var, 0):.2f}")
    
    # Level 3: Counterfactual
    print_section("Level 3: Counterfactual Reasoning (What if?)")
    
    # Evidence: It rained, plants grew well
    evidence = {"rain": 1.0, "plant_growth": 0.9, "sunlight": 0.8}
    
    # Query: What if it hadn't rained?
    query = CounterfactualQuery(
        factual_evidence=evidence,
        hypothetical_intervention={"rain": 0.0},
        query_variable="plant_growth"
    )
    
    result = reasoner.counterfactual(query)
    print(f"Factual: Plants grew at {evidence['plant_growth']:.1f} with rain")
    print(f"Counterfactual: Plants would have grown at {result:.2f} without rain")
    
    # Generate explanation
    print_section("Causal Explanation Generation")
    explanation = reasoner.explain("plant_growth", evidence)
    print(f"Why did the plants grow well?")
    print(f"  Effect value: {explanation.effect_value:.2f}")
    print(f"  Contributing causes:")
    for cause, value, contrib in explanation.causes[:3]:
        print(f"    - {cause}: value={value:.2f}, contribution={contrib:.2f}")
    
    return reasoner


def demo_abstract_reasoning():
    """Demonstrate abstract reasoning capabilities."""
    print_header("Abstract Reasoning Demo")
    print("(Logic, Analogy, and Pattern Recognition)")
    
    # Initialize abstract reasoner
    print_section("Initializing Abstract Reasoner")
    reasoner = AbstractReasoner()
    print("✓ Created abstract reasoning engine")
    
    # Build knowledge base
    print_section("Building Knowledge Base")
    kb = reasoner.knowledge_base
    
    # Add facts
    facts = [
        ("Socrates", "is_a", "human"),
        ("Plato", "is_a", "human"),
        ("human", "is_a", "mortal"),
        ("human", "is_a", "animal"),
        ("animal", "is_a", "living_thing"),
    ]
    
    for subject, relation, obj in facts:
        kb.add_fact(Proposition(predicate=relation, arguments=[subject, obj]))
        print(f"  ✓ Fact: {subject} {relation} {obj}")
    
    # Add rules
    rules = [
        Rule(
            name="transitivity",
            antecedent=[
                Proposition(predicate="is_a", arguments=["X", "Y"]),
                Proposition(predicate="is_a", arguments=["Y", "Z"])
            ],
            consequent=Proposition(predicate="is_a", arguments=["X", "Z"]),
            confidence=0.95
        ),
        Rule(
            name="inheritance",
            antecedent=[
                Proposition(predicate="is_a", arguments=["X", "Y"]),
                Proposition(predicate="has_property", arguments=["Y", "P"])
            ],
            consequent=Proposition(predicate="has_property", arguments=["X", "P"]),
            confidence=0.9
        )
    ]
    
    for rule in rules:
        kb.add_rule(rule)
        print(f"  ✓ Rule: {rule.name}")
    
    # Logical inference
    print_section("Logical Inference")
    
    print("Query: Is Socrates mortal?")
    result = reasoner.query(Proposition(predicate="is_a", arguments=["Socrates", "mortal"]))
    print(f"  Result: {result}")
    
    print("\nQuery: What is Socrates?")
    types = reasoner.infer_types("Socrates")
    print(f"  Socrates is: {types}")
    
    # Analogy reasoning
    print_section("Analogical Reasoning")
    
    # Set up source domain (solar system)
    solar_system = {
        "sun": "center",
        "planet": "orbits_sun",
        "moon": "orbits_planet",
        "gravity": "binding_force"
    }
    
    # Set up target domain (atom)
    atom = {
        "nucleus": "center",
        "electron": "orbits_nucleus",
        "force": "electromagnetic"
    }
    
    print("Source domain: Solar System")
    for k, v in solar_system.items():
        print(f"  {k}: {v}")
    
    print("\nTarget domain: Atom")
    for k, v in atom.items():
        print(f"  {k}: {v}")
    
    # Create analogy
    analogy = reasoner.create_analogy(solar_system, atom)
    print(f"\nAnalogy mapping:")
    for source, target in analogy.mappings.items():
        print(f"  {source} → {target}")
    
    # Transfer knowledge
    print("\nKnowledge transfer:")
    source_fact = Proposition(predicate="orbits", arguments=["moon", "planet"])
    transferred = analogy.transfer(source_fact)
    if transferred:
        print(f"  '{source_fact}' → '{transferred}'")
    
    # Pattern recognition
    print_section("Pattern Recognition")
    
    # Provide examples of a pattern
    examples = [
        [2, 4, 6, 8],
        [3, 6, 9, 12],
        [5, 10, 15, 20],
    ]
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex}")
    
    pattern = reasoner.induce_pattern(examples)
    print(f"\nInduced pattern: {pattern.structure}")
    print(f"Pattern type: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.2f}")
    
    # Test pattern on new example
    test = [4, 8, 12, 16]
    matches = reasoner.match_pattern(pattern, test)
    print(f"\nTest {test} against pattern: {matches}")
    
    # Complex reasoning
    print_section("Complex Reasoning: Raven's Matrix Style")
    
    # Simple matrix completion
    matrix = [
        ["circle", "square"],
        ["triangle", "?"]
    ]
    
    print("Matrix:")
    for row in matrix:
        print(f"  {row}")
    
    # Infer missing element based on pattern
    # In this case, each row/column has different shapes
    answer = reasoner.complete_matrix(matrix)
    print(f"\nInferred missing element: {answer}")
    
    return reasoner


def demo_integrated_reasoning():
    """Demonstrate how causal and abstract reasoning work together."""
    print_header("Integrated Reasoning Demo")
    print("(Combining causal and logical reasoning)")
    
    # Create both reasoners
    causal = CausalReasoner()
    abstract = AbstractReasoner()
    
    print_section("Scenario: Medical Diagnosis")
    
    # Build causal model of disease
    print("Building causal model...")
    causal.graph.add_variable("fever", True, False)
    causal.graph.add_variable("infection", True, False)
    causal.graph.add_variable("antibiotics", True, True)
    causal.graph.add_variable("recovery", True, False)
    
    causal.graph.add_link("infection", "fever", CausalRelationType.CAUSES, 0.8)
    causal.graph.add_link("fever", "recovery", CausalRelationType.PREVENTS, 0.3)
    causal.graph.add_link("antibiotics", "infection", CausalRelationType.PREVENTS, 0.9)
    causal.graph.add_link("antibiotics", "recovery", CausalRelationType.ENABLES, 0.85)
    
    # Add logical rules
    print("Adding diagnostic rules...")
    kb = abstract.knowledge_base
    kb.add_fact(Proposition(predicate="symptom_of", arguments=["fever", "infection"]))
    kb.add_fact(Proposition(predicate="treatment_for", arguments=["antibiotics", "infection"]))
    
    # Combined reasoning
    print_section("Diagnostic Reasoning")
    
    # Patient presents with fever
    symptoms = {"fever": 0.9}
    print(f"Patient symptoms: {symptoms}")
    
    # Causal: What could cause fever?
    print("\nCausal analysis (possible causes):")
    causes = causal.find_causes("fever")
    for cause, strength in causes[:3]:
        print(f"  - {cause}: likelihood {strength:.2f}")
    
    # Logical: What does fever imply?
    print("\nLogical inference:")
    result = abstract.query(Proposition(predicate="symptom_of", arguments=["fever", "infection"]))
    print(f"  Fever is a symptom of infection: {result}")
    
    # Treatment planning
    print_section("Treatment Planning")
    
    # Causal: What intervention leads to recovery?
    print("Causal intervention analysis:")
    intervention = causal.intervene({"antibiotics": 1.0})
    print(f"  Recovery probability with antibiotics: {intervention.get('recovery', 0):.2f}")
    
    no_treatment = causal.intervene({})
    print(f"  Recovery probability without treatment: {no_treatment.get('recovery', 0):.2f}")
    
    # Counterfactual: What if we didn't treat?
    print("\nCounterfactual analysis:")
    evidence = {"fever": 0.9, "infection": 0.8, "antibiotics": 1.0, "recovery": 0.9}
    query = CounterfactualQuery(
        factual_evidence=evidence,
        hypothetical_intervention={"antibiotics": 0.0},
        query_variable="recovery"
    )
    
    counterfactual = causal.counterfactual(query)
    print(f"  Actual recovery: {evidence['recovery']:.1f}")
    print(f"  Recovery without antibiotics: {counterfactual:.2f}")
    print(f"  Treatment effect: {evidence['recovery'] - counterfactual:.2f}")


def main():
    """Run all reasoning demos."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ATLAS Reasoning Systems Demo                       ║
    ║                                                              ║
    ║  Atlas implements multiple reasoning systems:                ║
    ║                                                              ║
    ║  • CAUSAL REASONING (Pearl's Causal Hierarchy)               ║
    ║    - Association: P(Y|X) - seeing/observing                  ║
    ║    - Intervention: P(Y|do(X)) - doing/manipulating           ║
    ║    - Counterfactual: P(Y_x|X',Y') - imagining               ║
    ║                                                              ║
    ║  • ABSTRACT REASONING                                        ║
    ║    - Symbolic logic and inference                            ║
    ║    - Analogical reasoning and transfer                       ║
    ║    - Pattern induction and recognition                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run demos
    causal_reasoner = demo_causal_reasoning()
    abstract_reasoner = demo_abstract_reasoning()
    demo_integrated_reasoning()
    
    print_header("Demo Complete!")
    print("""
Key Takeaways:
• Causal reasoning distinguishes correlation from causation
• Three levels: observation → intervention → counterfactual
• Abstract reasoning handles logic, analogy, and patterns
• Combined reasoning enables complex problem-solving

Next steps:
• Try building larger causal graphs
• Experiment with different analogy domains
• Combine with working memory for multi-step reasoning
    """)


if __name__ == "__main__":
    main()
