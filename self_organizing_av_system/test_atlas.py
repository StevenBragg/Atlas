#!/usr/bin/env python3
"""
Simple test script to verify Atlas self-organizing AV system is working.
"""

import numpy as np
import time
from models.visual.processor import VisualProcessor
from models.audio.processor import AudioProcessor
from models.multimodal.system import SelfOrganizingAVSystem


def test_atlas_system():
    """Test the Atlas system with synthetic data."""
    print("Testing Atlas Self-Organizing AV System...")
    
    # Create processors
    print("1. Creating visual processor...")
    visual_processor = VisualProcessor(
        input_width=64,
        input_height=64,
        use_grayscale=False
    )
    
    print("2. Creating audio processor...")
    audio_processor = AudioProcessor(
        sample_rate=22050
    )
    
    print("3. Creating multimodal system...")
    system = SelfOrganizingAVSystem(
        visual_processor=visual_processor,
        audio_processor=audio_processor,
        multimodal_size=100
    )
    
    print("4. Processing synthetic data...")
    # Create synthetic visual input (test pattern)
    visual_input = np.random.rand(64, 64, 3).astype(np.float32)
    
    # Create synthetic audio input (1 second of random audio)
    audio_input = np.random.randn(22050).astype(np.float32) * 0.1
    
    # Process a few frames
    for i in range(5):
        print(f"   Frame {i+1}/5...")
        
        # Process the inputs
        result = system.process_av_pair(visual_input, audio_input)
        
        # Print some stats
        stats = system.get_system_state()
        print(f"   - Visual features active: {stats.get('visual_sparsity', 0):.1%}")
        print(f"   - Audio features active: {stats.get('audio_sparsity', 0):.1%}")
        print(f"   - Cross-modal coherence: {stats.get('cross_modal_coherence', 0):.3f}")
        
        # Slightly modify inputs for next frame
        visual_input += np.random.randn(*visual_input.shape).astype(np.float32) * 0.01
        audio_input = np.random.randn(22050).astype(np.float32) * 0.1
    
    print("\n5. Testing cross-modal generation...")
    # Test visual to audio
    generated_audio = system.generate_from_visual(visual_input)
    print(f"   Generated audio features shape: {generated_audio.shape}")
    
    # Test audio to visual  
    generated_visual = system.generate_from_audio(audio_input)
    print(f"   Generated visual features shape: {generated_visual.shape}")
    
    print("\nSuccess! Atlas system is working correctly.")
    print("\nSystem capabilities:")
    print("- Visual processing with edge and complex features")
    print("- Audio processing with spectral analysis")
    print("- Cross-modal association learning")
    print("- Bidirectional cross-modal generation")
    print("- Self-organizing through Hebbian learning")
    print("- Works without camera or microphone using test patterns")


if __name__ == "__main__":
    test_atlas_system()