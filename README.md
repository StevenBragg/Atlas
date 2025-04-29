# ATLAS - Autonomously Teaching, Learning And Self-organizing

A biologically inspired self-organizing architecture for audio-visual learning that operates without supervision or pre-training.

## Project Overview

ATLAS is a neuromorphic architecture that learns from raw, synchronized audio-video streams without any pretraining or labeled data. This system autonomously discovers structure in sensory inputs through self-organization, using principles inspired by the brain's learning mechanisms.

ATLAS treats incoming camera frames and microphone waveforms as raw physical signals, with no built-in notions of objects, speech, or other human-centric features. It discovers structure through exposure and experience alone.

## Key Principles

- **No Backpropagation or Task Bias**: All learning uses local synaptic plasticity rules (Hebbian learning, Oja's rule, Spike-Timing-Dependent Plasticity) operating on correlations and spike timings. There is no end-to-end error gradient or human-defined objective; the system's only "teacher" is the temporal structure of the sensory stream itself.

- **Modular Sensory Pathways**: Separate video and audio processing pathways handle each modality's input, initially learning independent representations. Each pathway consists of layers of neurons that self-organize their connectivity and tuning based on statistical regularities in that modality.

- **Cross-Modal Integration**: As learning progresses, the system forms bridges between modalities through local associations. Neurons across the two pathways develop connections if their patterns of activity consistently coincide.

- **Emergent Encoding, Memory, and Prediction**: Higher-level layers act as an internal memory for temporal patterns. Neurons learn to encode not just static features, but also to predict upcoming sensory input based on recent activity.

- **Stability and Diversity**: Mechanisms include weight normalization, homeostatic scaling, inhibitory competition, and structural plasticity to promote stable learning and diverse representations.

- **Structural Plasticity**: The network can rewire and expand itself by adding new neurons or synapses in response to novel patterns, and pruning those that are redundant.

## Architecture Details

### Visual Pathway

The visual pathway processes raw video frames through a hierarchy of neural layers:

1. **Layer 1 (Low-Level Vision)**: Neurons receive raw pixel inputs and strengthen connections for pixels that often activate together. Using Oja's rule and lateral inhibition, these neurons specialize to detect basic visual primitives like edges, contrasts, or color blobs.

2. **Layer 2+ (Higher Visual Features)**: These layers capture combinations or temporal groupings of lower-level features, learning to recognize contours, textures, or motion patterns.

3. **Temporal Context and Prediction**: Using STDP, neurons form predictive connections where one visual feature tends to precede another, allowing the system to anticipate visual outcomes.

### Auditory Pathway

The auditory pathway processes raw audio signals in parallel:

1. **Input Preprocessing**: The raw audio waveform is segmented into time windows and may be filtered into frequency bands to create a cochleagram-like representation.

2. **Layer 1 (Low-Level Auditory Features)**: Neurons specialize to detect particular patterns of frequencies or temporal dynamics through Hebbian learning and competition.

3. **Layer 2 (Higher Auditory Features)**: These neurons learn more complex or invariant features, such as sequences of tones or combinations of frequency patterns.

4. **Temporal Sequencing**: As with vision, STDP enables the learning of temporal relationships between sounds, allowing prediction of auditory sequences.

### Cross-Modal Association

The system learns to associate visual and auditory patterns that consistently co-occur:

1. **Convergence Zone**: A multimodal association layer receives inputs from both visual and auditory pathways, strengthening connections when patterns co-occur.

2. **Direct Hebbian Links**: Direct synaptic connections form between visual and auditory neurons that frequently fire together.

3. **Temporal Window and Alignment**: STDP-like rules accommodate slight lead/lag relationships between modalities.

4. **Reentrant Loops**: Bidirectional connections enable one modality to trigger representations in the other, creating a distributed memory of multimodal events.

## Implementation Features

### Temporal Memory and Prediction

- **Modality-Specific Prediction**: Each pathway predicts its next input based on learned temporal sequences.
- **Joint Multimodal Prediction**: Cross-modal connections enable predictions across modalities.
- **Prediction-Driven Learning**: Differences between predicted and actual patterns drive synaptic changes.

### Stability Mechanisms

- **Homeostatic Plasticity**: Neurons maintain target firing rates by adjusting excitability.
- **Hebbian Normalization**: Oja's rule prevents unbounded weight growth.
- **Lateral Inhibition**: Competitive dynamics ensure neurons specialize to different patterns.
- **Sparse Activity**: Global inhibitory feedback maintains sparse network activity.
- **Synaptic Competition**: Unused connections are weakened and removed.

### Structural Plasticity

- **New Neuron Recruitment**: The network adds neurons for novel patterns.
- **Synapse Sprouting**: New connections form between correlated neurons.
- **Pruning**: Unnecessary neurons and connections are removed to improve efficiency.

## Installation

```bash
# Clone the repository
git clone https://github.com/StevenBragg/atlas.git
cd atlas

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

```bash
# Run the demo with a webcam input
python run_live.py

# Run with a prerecorded file
python run_file_demo.py path/to/video/file.mp4
```

## Project Status

ATLAS is currently under active development. The core architecture is implemented and functional, but we continue to refine the learning mechanisms and expand the system's capabilities.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This work draws inspiration from neuroscience research on self-organizing systems, predictive coding, and multimodal integration in the brain. 
