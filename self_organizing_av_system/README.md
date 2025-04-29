# Self-Organizing Audio-Visual Learning System

A biologically-inspired self-organizing system for learning audiovisual representations from synchronized sensory data without supervised training.

## Architecture Overview

This system uses biologically-inspired mechanisms to learn from raw sensory input without backpropagation or labeled data:

1. **Sensory Pathways**: Separate processing streams for visual and auditory data
2. **Hierarchical Feature Extraction**: Layered representation learning in each pathway
3. **Cross-Modal Integration**: Associative binding between visual and audio features
4. **Predictive Learning**: Temporal prediction mechanisms for sequence learning
5. **Self-Organizing Learning Rules**: Local Hebbian learning without global error signals
6. **Stability Mechanisms**: Homeostatic plasticity and normalization for balanced learning
7. **Structural Plasticity**: Dynamic network growth and pruning based on activity

## Key Features

- **No Backpropagation**: Uses only local learning rules
- **Self-Organization**: Emergent structure from unlabeled data
- **Stability**: Homeostatic mechanisms maintain balanced activity
- **Structural Plasticity**: Network structure adapts to input complexity
- **Cross-Modal Binding**: Automatic discovery of audiovisual correlations
- **Temporal Prediction**: Learns sequences and predicts future states
- **Unsupervised**: Works without labels or supervision

## Installation

Requires Python 3.8+ and the following libraries:
- NumPy
- PyTorch (for visual processing)
- Librosa (for audio processing)
- Matplotlib (for visualization)
- OpenCV (for video capture)

```bash
pip install -r requirements.txt
```

## Usage

### Running a Live Demo

Process live webcam and microphone input:

```bash
python -m self_organizing_av_system.examples.run_live_demo
```

Optional arguments:
- `--config PATH`: Path to custom configuration file
- `--save_dir PATH`: Directory to save output files
- `--no_vis`: Disable visualization
- `--no_learning`: Disable learning (inference only)

### Processing a Video File

```bash
python -m self_organizing_av_system.examples.run_file_demo path/to/video.mp4
```

Optional arguments:
- `--config PATH`: Path to custom configuration file
- `--save_dir PATH`: Directory to save output files 
- `--start_time SEC`: Start time in seconds
- `--duration SEC`: Processing duration in seconds

## Configuration

The system's parameters can be configured via YAML files. See `config/default_config.yaml` for a complete example.

Key configuration sections:
- `system`: Global parameters
- `visual_processor`: Visual pathway settings
- `audio_processor`: Audio pathway settings
- `multimodal_association`: Cross-modal binding settings
- `temporal_prediction`: Sequence learning settings
- `stability`: Homeostatic plasticity settings
- `structural_plasticity`: Network growth settings

## Project Structure

- `self_organizing_av_system/`: Main package directory
  - `core/`: Core system components
    - `system.py`: Main system integration
    - `multimodal_association.py`: Cross-modal binding
    - `temporal_prediction.py`: Sequence learning
    - `stability.py`: Homeostatic mechanisms
    - `structural_plasticity.py`: Network adaptation
    - `multimodal.py`: Legacy implementation (for reference only)
  - `processors/`: Sensory processing modules
    - `visual_processor.py`: Visual pathway
    - `audio_processor.py`: Auditory pathway
  - `utils/`: Utility functions and helpers
    - `monitor.py`: Visualization and monitoring
    - `av_capture.py`: Audio-video capture tools
  - `examples/`: Example scripts
    - `run_live_demo.py`: Live webcam/mic processing
    - `run_file_demo.py`: Video file processing
  - `config/`: Configuration files

## Code Organization Notes

This project has evolved through multiple iterations. Some important notes:

- Current architecture uses modular components with consistent interfaces
- The file `multimodal.py` contains a legacy implementation (renamed to `LegacyMultimodalAssociation`) kept for reference
- The current implementation uses `multimodal_association.py` instead
- Some older components like `neuron.py`, `layer.py`, and `pathway.py` have been replaced with more integrated designs

## How it Works

1. **Visual Processing**: Extracts hierarchical features from images
2. **Audio Processing**: Extracts spectral and temporal features from audio
3. **Cross-Modal Association**: Discovers correlations between modalities
4. **Temporal Prediction**: Learns to predict future states
5. **Homeostatic Plasticity**: Maintains stable activity patterns
6. **Structural Plasticity**: Grows and prunes network connections

The system learns incrementally as it processes synchronized audio-visual data, forming cross-modal associations between correlated features.

## Biological Inspiration

This system is inspired by several aspects of brain organization:

- **Sensory Pathways**: Similar to visual and auditory cortical streams
- **Local Learning**: Inspired by STDP and Hebbian plasticity
- **Lateral Inhibition**: Competitive processes like in cortical columns
- **Homeostasis**: Similar to synaptic scaling in neural circuits
- **Multimodal Integration**: Like multisensory neurons in superior temporal sulcus
- **Predictive Processing**: Similar to predictive coding in the brain
- **Structural Plasticity**: Inspired by neural pruning and synaptogenesis

## License

MIT License 