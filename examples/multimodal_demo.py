#!/usr/bin/env python3
"""
Multimodal Learning Demo for Atlas

This example demonstrates Atlas's audio-visual learning capabilities:
- Cross-modal associations between vision and audio
- Predictive coding across modalities
- Emergent multimodal representations

Note: This demo uses synthetic data for demonstration.
For live webcam/microphone processing, use the main Atlas system.

Usage:
    python multimodal_demo.py
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_organizing_av_system.core.cross_modal_association import CrossModalAssociation
from self_organizing_av_system.core.multimodal_association import MultimodalAssociation, AssociationMode


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---")


def generate_sensory_pattern(modality: str, pattern_type: str, size: int) -> np.ndarray:
    """Generate synthetic sensory patterns."""
    pattern = np.zeros(size)
    
    if modality == "visual":
        if pattern_type == "red_circle":
            pattern[0:10] = 1.0  # Red channel
            pattern[40:50] = 0.8  # Circular shape
        elif pattern_type == "blue_square":
            pattern[20:30] = 1.0  # Blue channel
            pattern[50:60] = 0.8  # Square shape
        elif pattern_type == "green_triangle":
            pattern[10:20] = 1.0  # Green channel
            pattern[60:70] = 0.8  # Triangular shape
        else:
            pattern = np.random.randn(size) * 0.3
    
    elif modality == "audio":
        if pattern_type == "high_pitch":
            pattern[0:20] = 1.0  # High frequency
        elif pattern_type == "low_pitch":
            pattern[40:60] = 1.0  # Low frequency
        elif pattern_type == "rhythmic":
            pattern[::5] = 0.8  # Rhythmic pattern
        else:
            pattern = np.random.randn(size) * 0.3
    
    return pattern


def demo_cross_modal_learning():
    """Demonstrate cross-modal association learning."""
    print_header("Cross-Modal Association Demo")
    print("(Hebbian learning between visual and auditory pathways)")
    
    # Initialize cross-modal association
    print_section("Initializing Cross-Modal Association")
    cross_modal = CrossModalAssociation(
        visual_size=100,
        audio_size=80,
        learning_rate=0.01,
        association_threshold=0.1,
        temporal_window_size=5,
        bidirectional=True
    )
    print("✓ Created cross-modal association (visual: 100, audio: 80)")
    
    # Training: Pair visual patterns with audio patterns
    print_section("Training Cross-Modal Associations")
    
    # Define paired patterns (like objects with characteristic sounds)
    pairings = [
        ("red_circle", "high_pitch"),
        ("blue_square", "low_pitch"),
        ("green_triangle", "rhythmic"),
    ]
    
    print("Training pairs (visual → audio):")
    for visual_type, audio_type in pairings:
        print(f"  {visual_type} ↔ {audio_type}")
    
    # Train on multiple presentations
    print("\nTraining for 50 epochs...")
    for epoch in range(50):
        for visual_type, audio_type in pairings:
            visual = generate_sensory_pattern("visual", visual_type, 100)
            audio = generate_sensory_pattern("audio", audio_type, 80)
            
            # Add some noise for robustness
            visual += np.random.randn(100) * 0.1
            audio += np.random.randn(80) * 0.1
            
            cross_modal.update(visual, audio, learn=True)
    
    print("✓ Training complete")
    
    # Test cross-modal prediction
    print_section("Cross-Modal Prediction")
    
    print("Testing: Predict audio from visual input")
    for visual_type, expected_audio in pairings:
        visual = generate_sensory_pattern("visual", visual_type, 100)
        predictions = cross_modal.predict_audio_from_visual(visual)
        
        # Find strongest predicted audio features
        top_features = np.argsort(predictions)[-3:]
        print(f"\n  Visual: {visual_type}")
        print(f"    Top predicted audio features: {top_features}")
        print(f"    Prediction strength: {np.max(predictions):.3f}")
    
    print("\nTesting: Predict visual from audio input")
    for visual_type, audio_type in pairings:
        audio = generate_sensory_pattern("audio", audio_type, 80)
        predictions = cross_modal.predict_visual_from_audio(audio)
        
        top_features = np.argsort(predictions)[-3:]
        print(f"\n  Audio: {audio_type}")
        print(f"    Top predicted visual features: {top_features}")
        print(f"    Prediction strength: {np.max(predictions):.3f}")
    
    # Show association statistics
    print_section("Association Statistics")
    stats = cross_modal.get_stats()
    print(f"Update count: {stats['update_count']}")
    print(f"Average association strength: {np.mean(stats['association_strength']):.3f}")
    print(f"Visual→Audio prediction accuracy: {np.mean(stats['v2a_accuracy']):.3f}")
    print(f"Audio→Visual prediction accuracy: {np.mean(stats['a2v_accuracy']):.3f}")
    
    return cross_modal


def demo_multimodal_integration():
    """Demonstrate multimodal integration at a higher level."""
    print_header("Multimodal Integration Demo")
    print("(Higher-level multimodal association and binding)")
    
    # Initialize multimodal association
    print_section("Initializing Multimodal Association")
    multimodal = MultimodalAssociation(
        visual_size=64,
        audio_size=48,
        output_size=32,
        mode=AssociationMode.BINDING
    )
    print("✓ Created multimodal association (output: 32)")
    
    # Create some multimodal events
    print_section("Processing Multimodal Events")
    
    events = [
        {"name": "Bell Ringing", "visual": "red_circle", "audio": "high_pitch"},
        {"name": "Drum Beat", "visual": "blue_square", "audio": "rhythmic"},
        {"name": "Whistle", "visual": "green_triangle", "audio": "high_pitch"},
    ]
    
    integrated_representations = []
    
    for event in events:
        visual = generate_sensory_pattern("visual", event["visual"], 64)
        audio = generate_sensory_pattern("audio", event["audio"], 48)
        
        # Integrate modalities
        integrated = multimodal.bind(visual, audio)
        integrated_representations.append((event["name"], integrated))
        
        print(f"\n  Event: {event['name']}")
        print(f"    Visual pattern: {event['visual']}")
        print(f"    Audio pattern: {event['audio']}")
        print(f"    Integrated representation shape: {integrated.shape}")
        print(f"    Activation: {np.mean(integrated):.3f}")
    
    # Test retrieval
    print_section("Cross-Modal Retrieval")
    
    # Query with only visual input
    query_visual = generate_sensory_pattern("visual", "red_circle", 64)
    retrieved = multimodal.retrieve_from_visual(query_visual)
    
    print("Query: Visual pattern 'red_circle'")
    print(f"  Retrieved multimodal representation activation: {np.mean(retrieved):.3f}")
    
    # Query with only audio input
    query_audio = generate_sensory_pattern("audio", "high_pitch", 48)
    retrieved = multimodal.retrieve_from_audio(query_audio)
    
    print("\nQuery: Audio pattern 'high_pitch'")
    print(f"  Retrieved multimodal representation activation: {np.mean(retrieved):.3f}")
    
    return multimodal


def demo_temporal_binding():
    """Demonstrate temporal binding of multimodal events."""
    print_header("Temporal Binding Demo")
    print("(Binding events across time)")
    
    cross_modal = CrossModalAssociation(
        visual_size=50,
        audio_size=40,
        temporal_window_size=3,
        learning_rate=0.02
    )
    
    print_section("Simulating Temporal Sequence")
    
    # Simulate a sequence: visual event followed by audio event
    sequence_length = 10
    
    print("Sequence: Visual flash → Audio beep (repeated)")
    
    for i in range(sequence_length):
        # Visual flash at even steps
        if i % 2 == 0:
            visual = generate_sensory_pattern("visual", "red_circle", 50)
            audio = np.random.randn(40) * 0.1  # Quiet audio
            print(f"  Step {i}: FLASH (visual active)")
        # Audio beep at odd steps (predicted from previous visual)
        else:
            visual = np.random.randn(50) * 0.1  # Dim visual
            audio = generate_sensory_pattern("audio", "high_pitch", 40)
            print(f"  Step {i}: BEEP (audio active)")
        
        cross_modal.update(visual, audio, learn=True)
        time.sleep(0.01)  # Small delay
    
    # Test prediction
    print_section("Testing Temporal Prediction")
    
    # Present only visual, expect audio prediction
    visual = generate_sensory_pattern("visual", "red_circle", 50)
    predictions = cross_modal.predict_audio_from_visual(visual)
    
    print("Presenting visual flash alone...")
    print(f"  Predicted audio activity: {np.mean(predictions):.3f}")
    print(f"  Strong prediction: {np.mean(predictions) > 0.3}")
    
    return cross_modal


def demo_emergent_properties():
    """Demonstrate emergent properties from multimodal learning."""
    print_header("Emergent Properties Demo")
    print("(Capabilities that emerge from multimodal learning)")
    
    cross_modal = CrossModalAssociation(
        visual_size=60,
        audio_size=50,
        learning_rate=0.01
    )
    
    print_section("Training on Correlated Patterns")
    
    # Train on strongly correlated patterns
    training_pairs = [
        ("circle", "round_sound"),
        ("square", "angular_sound"),
    ]
    
    for epoch in range(100):
        for visual_type, audio_type in training_pairs:
            visual = generate_sensory_pattern("visual", visual_type, 60)
            
            # Create audio pattern based on visual
            if audio_type == "round_sound":
                audio = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.5 + 0.5
            else:
                audio = np.abs(np.random.randn(50)) * 0.8
            
            cross_modal.update(visual, audio, learn=True)
    
    print("✓ Trained on 2 pattern pairs for 100 epochs")
    
    # Test emergent property: robustness to noise
    print_section("Emergent Property: Noise Robustness")
    
    visual = generate_sensory_pattern("visual", "circle", 60)
    
    noise_levels = [0.0, 0.2, 0.5, 0.8]
    print("Testing prediction with increasing noise:")
    
    for noise in noise_levels:
        noisy_visual = visual + np.random.randn(60) * noise
        prediction = cross_modal.predict_audio_from_visual(noisy_visual)
        print(f"  Noise level {noise:.1f}: prediction strength = {np.mean(prediction):.3f}")
    
    # Test emergent property: pattern completion
    print_section("Emergent Property: Pattern Completion")
    
    # Present partial visual pattern
    partial_visual = generate_sensory_pattern("visual", "circle", 60)
    partial_visual[30:] = 0  # Mask half
    
    print("Presenting partial visual pattern (50% occluded)...")
    
    # Cross-modal prediction can help complete the pattern
    audio_prediction = cross_modal.predict_audio_from_visual(partial_visual)
    visual_completion = cross_modal.predict_visual_from_audio(audio_prediction)
    
    print(f"  Original pattern strength: {np.mean(partial_visual):.3f}")
    print(f"  Completed pattern strength: {np.mean(visual_completion):.3f}")
    
    return cross_modal


def main():
    """Run all multimodal demos."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ATLAS Multimodal Learning Demo                     ║
    ║                                                              ║
    ║  Atlas learns from synchronized audio-visual streams:        ║
    ║                                                              ║
    ║  • Cross-Modal Associations                                  ║
    ║    - Hebbian learning between visual and audio               ║
    ║    - Bidirectional prediction (see → hear, hear → see)       ║
    ║    - Temporal binding across modalities                      ║
    ║                                                              ║
    ║  • Emergent Capabilities                                     ║
    ║    - Noise robustness through multimodal redundancy          ║
    ║    - Pattern completion via cross-modal cues                 ║
    ║    - Invariant representations across modalities             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Note: This demo uses synthetic data.")
    print("For live webcam/microphone processing, run:")
    print("  python self_organizing_av_system/main.py\n")
    
    # Run demos
    cross_modal = demo_cross_modal_learning()
    multimodal = demo_multimodal_integration()
    temporal = demo_temporal_binding()
    emergent = demo_emergent_properties()
    
    print_header("Demo Complete!")
    print("""
Key Takeaways:
• Cross-modal associations form through temporal coincidence
• Bidirectional prediction enables sensory completion
• Multimodal learning is more robust than unimodal
• Emergent properties arise from statistical structure

Next steps:
• Try the live demo with real webcam/microphone
• Experiment with different learning rates
• Combine with episodic memory for event learning
    """)


if __name__ == "__main__":
    main()
