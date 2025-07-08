# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATLAS (Autonomously Teaching, Learning And Self-organizing) is a biologically-inspired neuromorphic architecture for unsupervised audio-visual learning. The system uses local Hebbian learning rules without backpropagation and implements structural plasticity for dynamic network growth.

## Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Windows users: Set up conda environment
install.bat
```

### Running the System
```bash
# Live demo with webcam/microphone
python main.py
python examples/run_live_demo.py

# Process video file
python main.py --video-file path/to/video.mp4
python examples/run_file_demo.py path/to/video.mp4

# Windows shortcuts available
run_live.bat
run_file_demo.bat
```

### Testing
```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python tests/test_full_pipeline.py

# Run individual test
python -m unittest test_full_pipeline.TestFullPipeline.test_full_pipeline_with_resizing
```

## Architecture

The system consists of parallel visual and auditory processing pathways that converge in association areas:

1. **Visual Pathway** (`/models/visual_system.py`):
   - Edge detection layer → Complex feature maps
   - Structural plasticity enables emergent topographic organization

2. **Audio Pathway** (`/models/audio_system.py`):
   - Spectral feature extraction → Temporal pattern detection
   - Adapts to discovered sound patterns

3. **Convergence Zones** (`/core/convergence_zone.py`):
   - Cross-modal association through Hebbian learning
   - No supervision required

4. **Plasticity Mechanisms** (`/core/structural_plasticity.py`):
   - Dynamic synapse formation/elimination
   - Network self-organizes based on input statistics

## Key Directories

- `/core/` - Neural components (neurons, layers, plasticity)
- `/models/` - Sensory systems (visual, audio, multimodal)
- `/config/` - YAML-based configuration (default: `config/default_config.yaml`)
- `/gui/` - Real-time monitoring interfaces
- `/utils/` - Capture and monitoring utilities
- `/checkpoints/` - Saved model states for resuming sessions

## Development Notes

- The system uses PyTorch for neural computations but implements custom learning rules (no autograd)
- Checkpoint system allows saving/loading learned representations
- GUI monitoring shows learning progress in real-time
- All sensory processing is unsupervised - no labels or targets needed
- Configuration via `config/default_config.yaml` controls network parameters