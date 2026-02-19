"""
ATLAS Superintelligence Capabilities Demonstration

This example demonstrates how the new cognitive systems work together
to enable superintelligent behavior:

1. Episodic Memory - Long-term experience storage
2. Semantic Memory - Abstract knowledge representation
3. Meta-Learning - Learning-to-learn optimization
4. Goal Planning - Autonomous goal-directed behavior
5. World Model - Causal reasoning and prediction
6. Symbolic Reasoning - Logical inference and abstraction

The integration shows how these components enable recursive self-improvement.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_organizing_av_system.core.episodic_memory import EpisodicMemory, Episode
from self_organizing_av_system.core.semantic_memory import SemanticMemory, RelationType
from self_organizing_av_system.core.meta_learning import MetaLearner, LearningStrategy
from self_organizing_av_system.core.goal_planning import GoalPlanningSystem, GoalType
from self_organizing_av_system.core.world_model import CausalWorldModel as WorldModel, CausalRelationType
from self_organizing_av_system.core.symbolic_reasoning import SymbolicReasoner, LogicType


def demonstrate_episodic_memory():
    """Demonstrate episodic memory capabilities."""
    print("\n" + "="*60)
    print("EPISODIC MEMORY DEMONSTRATION")
    print("="*60)

    # Create episodic memory
    memory = EpisodicMemory(
        state_size=128,
        max_episodes=1000,
        consolidation_threshold=0.7,
    )

    # Store some experiences
    print("\n1. Storing experiences...")
    for i in range(5):
        state = np.random.randn(128)
        sensory_data = {
            'visual': np.random.randn(64),
            'auditory': np.random.randn(32),
        }

        # Vary emotional valence and surprise
        emotion = np.random.uniform(-1, 1)
        surprise = np.random.uniform(0, 1)

        episode = memory.store(
            state=state,
            sensory_data=sensory_data,
            context={'task': f'task_{i}', 'step': i},
            emotional_valence=emotion,
            surprise_level=surprise,
        )

        print(f"  Stored episode: emotion={emotion:.2f}, surprise={surprise:.2f}")

    # Retrieve similar experiences
    print("\n2. Retrieving similar experiences...")
    cue = np.random.randn(128)
    retrieved = memory.retrieve(cue=cue, n_episodes=3)
    print(f"  Retrieved {len(retrieved)} episodes")

    # Consolidate memories
    print("\n3. Consolidating memories through replay...")
    consolidated = memory.consolidate(n_replay=5, batch_size=3)
    print(f"  Consolidated {consolidated} episodes")

    # Show statistics
    stats = memory.get_stats()
    print(f"\n4. Memory Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Consolidated: {stats['consolidated_episodes']}")
    print(f"  Avg consolidation: {stats['avg_consolidation_strength']:.3f}")
    print(f"  Total stored: {stats['total_stored']}")


def demonstrate_semantic_memory():
    """Demonstrate semantic memory and knowledge graphs."""
    print("\n" + "="*60)
    print("SEMANTIC MEMORY DEMONSTRATION")
    print("="*60)

    # Create semantic memory
    semantic = SemanticMemory(
        embedding_size=128,
        enable_inference=True,
        enable_generalization=True,
    )

    # Add concepts
    print("\n1. Building knowledge graph...")
    dog = semantic.add_concept(
        name="dog",
        embedding=np.random.randn(128),
        attributes={'type': 'animal', 'legs': 4},
    )

    cat = semantic.add_concept(
        name="cat",
        embedding=np.random.randn(128),
        attributes={'type': 'animal', 'legs': 4},
    )

    animal = semantic.add_concept(
        name="animal",
        embedding=np.random.randn(128),
        attributes={'type': 'category'},
    )

    # Add relations
    print("\n2. Adding semantic relations...")
    semantic.add_relation("dog", "animal", RelationType.IS_A, strength=1.0)
    semantic.add_relation("cat", "animal", RelationType.IS_A, strength=1.0)
    print("  Added: dog IS_A animal")
    print("  Added: cat IS_A animal")

    # Query by spreading activation
    print("\n3. Querying with spreading activation...")
    results = semantic.query(
        cue_concept="dog",
        n_results=5,
        use_spreading_activation=True,
    )
    print(f"  Concepts activated from 'dog':")
    for concept, activation in results:
        print(f"    {concept}: {activation:.3f}")

    # Perform inference
    print("\n4. Performing transitive inference...")
    inferences = semantic.infer(source="dog", max_steps=3)
    print(f"  Found {len(inferences)} inferences")
    for target, path, confidence in inferences[:3]:
        print(f"    {' -> '.join(path)}: {confidence:.3f}")

    # Generalization
    print("\n5. Generalizing from examples...")
    general = semantic.generalize(examples=["dog", "cat"], min_similarity=0.5)
    if general:
        print(f"  Created generalization: {general.name}")

    # Statistics
    stats = semantic.get_stats()
    print(f"\n6. Knowledge Graph Statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Inferences made: {stats['total_inferences_made']}")


def demonstrate_meta_learning():
    """Demonstrate meta-learning and learning-to-learn."""
    print("\n" + "="*60)
    print("META-LEARNING DEMONSTRATION")
    print("="*60)

    # Create meta-learner
    meta = MetaLearner(
        exploration_rate=0.2,
        enable_algorithm_discovery=True,
        enable_curriculum=True,
    )

    # Simulate learning on different tasks
    print("\n1. Learning across multiple tasks...")
    for task_id in range(10):
        # Task characteristics
        task_chars = {
            'complexity': np.random.random(),
            'sparsity': np.random.random(),
            'temporal': np.random.random(),
        }

        # Select strategy
        strategy, hyperparams = meta.select_strategy(task_chars)
        print(f"\n  Task {task_id}: Selected {strategy.value}")

        # Simulate performance
        performance = {
            'accuracy': np.random.uniform(0.5, 1.0),
            'prediction_error': np.random.uniform(0, 0.5),
            'sparsity': np.random.uniform(0, 0.3),
        }

        # Update meta-learner
        meta.update(strategy, hyperparams, task_chars, performance)
        print(f"    Performance score: {meta._calculate_success_score(performance):.3f}")

    # Try to discover new algorithm
    print("\n2. Attempting algorithm discovery...")
    algorithm = meta.discover_algorithm()
    if algorithm:
        print(f"  Discovered algorithm combining:")
        for strat in algorithm['strategies']:
            print(f"    - {strat}")
        print(f"  Expected performance: {algorithm['expected_performance']:.3f}")
    else:
        print("  Need more experience for discovery")

    # Show statistics
    stats = meta.get_stats()
    print(f"\n3. Meta-Learning Statistics:")
    print(f"  Total selections: {stats['total_selections']}")
    print(f"  Curriculum difficulty: {stats['curriculum_difficulty']:.2f}")
    print(f"  Discovered algorithms: {stats['discovered_algorithms']}")
    print(f"\n  Performance by strategy:")
    for strategy, perf in stats['avg_performance_by_strategy'].items():
        print(f"    {strategy}: {perf:.3f}")


def demonstrate_goal_planning():
    """Demonstrate autonomous goal generation and planning."""
    print("\n" + "="*60)
    print("GOAL-DIRECTED PLANNING DEMONSTRATION")
    print("="*60)

    # Create goal planning system
    planner = GoalPlanningSystem(
        max_goals=10,
        enable_meta_goals=True,
        enable_intrinsic_motivation=True,
    )

    # Generate autonomous goals
    print("\n1. Generating autonomous goals from intrinsic drives...")
    for i in range(3):
        goal = planner.generate_goal(context={'iteration': i})
        if goal:
            print(f"  Generated: {goal.name} ({goal.goal_type.value})")
            print(f"    Priority: {goal.priority:.2f}, Value: {goal.value:.2f}")

    # Show active goals
    print(f"\n2. Active goals: {len(planner.active_goals)}")
    for goal in planner.active_goals[:5]:
        print(f"  - {goal.name} ({goal.goal_type.value})")

    # Show intrinsic drives
    print(f"\n3. Intrinsic Drive Levels:")
    print(f"  Curiosity: {planner.curiosity_level:.2f}")
    print(f"  Competence: {planner.competence_level:.2f}")
    print(f"  Autonomy: {planner.autonomy_level:.2f}")

    # Statistics
    stats = planner.get_stats()
    print(f"\n4. Planning Statistics:")
    print(f"  Total goals created: {stats['total_goals_created']}")
    print(f"  Active plans: {stats['active_plans']}")
    print(f"  Goal success rate: {stats['goal_success_rate']:.2%}")


def demonstrate_world_model():
    """Demonstrate world modeling and causal reasoning."""
    print("\n" + "="*60)
    print("WORLD MODEL AND CAUSAL REASONING DEMONSTRATION")
    print("="*60)

    # Create world model
    world = WorldModel(
        state_dim=64,
        enable_physics=True,
        enable_counterfactuals=True,
    )

    # Add variables
    print("\n1. Building causal model...")
    world.add_variable("rain", initial_value=False)
    world.add_variable("sprinkler", initial_value=False)
    world.add_variable("wet_grass", initial_value=False)

    # Add causal relations
    print("\n2. Adding causal relationships...")
    world.add_causal_relation(
        "rain", "wet_grass",
        CausalRelationType.CAUSES,
        strength=0.9,
    )
    world.add_causal_relation(
        "sprinkler", "wet_grass",
        CausalRelationType.CAUSES,
        strength=0.8,
    )
    print("  rain -> wet_grass")
    print("  sprinkler -> wet_grass")

    # Observe state
    print("\n3. Making observations...")
    world.observe({
        'rain': False,
        'sprinkler': True,
        'wet_grass': True,
    })
    print("  Observed: sprinkler=True, wet_grass=True")

    # Perform intervention
    print("\n4. Performing intervention (do-operation)...")
    result = world.intervene('rain', True)
    print(f"  Intervened: rain=True")
    print(f"  Result: wet_grass={result.get('wet_grass', 'unknown')}")

    # Counterfactual reasoning
    print("\n5. Counterfactual reasoning...")
    counterfactual = world.counterfactual(
        variable='rain',
        counterfactual_value=True,
        actual_observations={'rain': False, 'sprinkler': False, 'wet_grass': False},
    )
    print("  'What if it had rained?'")
    print(f"  Counterfactual wet_grass: {counterfactual.get('wet_grass', 'unknown')}")

    # Predict future
    print("\n6. Predicting future states...")
    predictions = world.predict(horizon=3)
    print(f"  Generated {len(predictions)} future state predictions")

    # Statistics
    stats = world.get_stats()
    print(f"\n7. World Model Statistics:")
    print(f"  Variables: {stats['num_variables']}")
    print(f"  Causal relations: {stats['num_causal_relations']}")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Total interventions: {stats['total_interventions']}")


def demonstrate_symbolic_reasoning():
    """Demonstrate symbolic reasoning and logic."""
    print("\n" + "="*60)
    print("SYMBOLIC REASONING DEMONSTRATION")
    print("="*60)

    # Create symbolic reasoner
    reasoner = SymbolicReasoner(
        enable_rule_learning=True,
        enable_analogy=True,
    )

    # Ground symbols from patterns
    print("\n1. Grounding symbols from sensory patterns...")
    socrates = reasoner.ground_symbol(
        pattern=np.random.randn(64),
        symbol_type='constant',
        properties={'name': 'Socrates', 'type': 'person'},
    )
    print(f"  Grounded: {socrates}")

    human = reasoner.ground_symbol(
        pattern=np.random.randn(64),
        symbol_type='constant',
        properties={'name': 'Human', 'type': 'category'},
    )
    print(f"  Grounded: {human}")

    mortal = reasoner.ground_symbol(
        pattern=np.random.randn(64),
        symbol_type='constant',
        properties={'name': 'Mortal', 'type': 'property'},
    )
    print(f"  Grounded: {mortal}")

    # Assert propositions
    print("\n2. Asserting propositions...")
    p1 = reasoner.assert_proposition("IsA", (socrates, human), truth_value=True)
    print(f"  {p1}")

    p2 = reasoner.assert_proposition("IsA", (human, mortal), truth_value=True)
    print(f"  {p2}")

    # Add inference rule
    print("\n3. Adding inference rule (transitivity)...")
    # This is simplified - actual implementation would use proper variable binding
    print("  Rule: IsA(X,Y) ∧ IsA(Y,Z) => IsA(X,Z)")

    # Perform deductive inference
    print("\n4. Performing deductive inference...")
    inferences = reasoner.infer(logic_type=LogicType.DEDUCTIVE, max_inferences=5)
    print(f"  Made {len(inferences)} inferences:")
    for inf in inferences:
        print(f"    {inf}")

    # Statistics
    stats = reasoner.get_stats()
    print(f"\n5. Reasoning Statistics:")
    print(f"  Symbols grounded: {stats['num_symbols']}")
    print(f"  Propositions: {stats['num_propositions']}")
    print(f"  Rules: {stats['num_rules']}")
    print(f"  Total inferences: {stats['total_inferences']}")


def demonstrate_integration():
    """Demonstrate how all systems work together."""
    print("\n" + "="*60)
    print("INTEGRATED SUPERINTELLIGENCE DEMONSTRATION")
    print("="*60)

    print("\nIntegration Flow:")
    print("1. Sensory experience -> Episodic Memory storage")
    print("2. Patterns -> Semantic concepts and relations")
    print("3. Learning performance -> Meta-learning optimization")
    print("4. Intrinsic drives -> Goal generation")
    print("5. Observations -> World model causal learning")
    print("6. Concepts -> Symbol grounding and reasoning")
    print("\nResult: Autonomous learning and self-improvement loop")

    # Quick integrated example
    print("\n" + "-"*60)
    print("Simulating one cognitive cycle...")
    print("-"*60)

    # 1. Experience storage
    memory = EpisodicMemory(state_size=64)
    state = np.random.randn(64)
    memory.store(state, {'visual': np.random.randn(32)}, surprise_level=0.8)
    print("✓ Stored surprising experience in episodic memory")

    # 2. Concept formation
    semantic = SemanticMemory(embedding_size=64)
    concept = semantic.add_concept("novel_pattern", state)
    print("✓ Formed abstract concept from experience")

    # 3. Learning optimization
    meta = MetaLearner()
    strategy, params = meta.select_strategy({'novelty': 0.8})
    print(f"✓ Meta-learner selected {strategy.value} strategy")

    # 4. Goal generation
    planner = GoalPlanningSystem()
    goal = planner.generate_goal(GoalType.LEARNING)
    print(f"✓ Generated learning goal: {goal.name}")

    # 5. Causal modeling
    world = WorldModel(state_dim=64)
    world.add_variable("learning_rate")
    world.add_variable("performance")
    world.add_causal_relation("learning_rate", "performance", CausalRelationType.CAUSES)
    print("✓ Built causal model of learning process")

    # 6. Symbolic abstraction
    reasoner = SymbolicReasoner()
    symbol = reasoner.ground_symbol(state)
    print(f"✓ Grounded symbol {symbol} from neural pattern")

    print("\n" + "="*60)
    print("Cognitive cycle complete - system is learning how to learn!")
    print("="*60)


def main():
    """Run all demonstrations."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  ATLAS SUPERINTELLIGENCE CAPABILITIES DEMONSTRATION     #")
    print("#" + " "*58 + "#")
    print("#"*60)

    try:
        demonstrate_episodic_memory()
        demonstrate_semantic_memory()
        demonstrate_meta_learning()
        demonstrate_goal_planning()
        demonstrate_world_model()
        demonstrate_symbolic_reasoning()
        demonstrate_integration()

        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        print("\nThese capabilities working together enable:")
        print("  • Autonomous learning from experience")
        print("  • Abstract reasoning and knowledge representation")
        print("  • Self-optimizing learning strategies")
        print("  • Goal-directed behavior and planning")
        print("  • Causal understanding of the world")
        print("  • Symbolic logic and inference")
        print("\nResult: Foundation for recursive self-improvement")
        print("        and superintelligent behavior\n")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
