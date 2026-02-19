# Atlas Tutorial - Getting Started Guide

Welcome to Atlas! This tutorial will guide you through using Atlas's self-organizing audio-visual learning system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding Atlas](#understanding-atlas)
4. [Text Learning](#text-learning)
5. [Memory Systems](#memory-systems)
6. [Reasoning](#reasoning)
7. [Multimodal Learning](#multimodal-learning)
8. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Webcam and microphone (for live demos)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/StevenBragg/Atlas.git
cd Atlas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r self_organizing_av_system/requirements.txt
```

### Installation Options

| Requirements File | Use Case |
|-------------------|----------|
| `requirements.txt` | Full installation with audio support |
| `requirements_no_audio.txt` | Without audio dependencies |
| `requirements_headless.txt` | Server deployment (no GUI/audio) |

---

## Quick Start

### 1. Run a Demo

The fastest way to understand Atlas is to run the examples:

```bash
cd examples

# Text learning demo
python text_learning_demo.py

# Memory systems demo
python memory_demo.py

# Reasoning demo
python reasoning_demo.py

# Multimodal demo
python multimodal_demo.py
```

### 2. Live Learning (with webcam)

```bash
cd self_organizing_av_system
python main.py
```

This starts Atlas with live webcam and microphone input. The system will begin learning immediately from your sensory environment.

### 3. Process a Video File

```bash
python main.py --video-file path/to/video.mp4
```

---

## Understanding Atlas

### What Makes Atlas Different?

Traditional machine learning requires:
- ❌ Labeled training data
- ❌ Predefined objectives
- ❌ Human-designed features
- ❌ Fixed network architecture

Atlas uses **self-organizing principles**:
- ✅ Learns from raw sensory streams
- ✅ Discovers structure without labels
- ✅ Adapts its architecture dynamically
- ✅ Develops representations through experience

### Core Concepts

#### 1. Predictive Coding
Atlas learns by predicting future sensory input. Prediction errors drive learning.

```
Prediction Error = Actual Input - Predicted Input
Learning Update ∝ Prediction Error × Learning Rate
```

#### 2. Hebbian Learning
"Neurons that fire together, wire together." Concurrent activity strengthens connections.

```python
# When visual and audio patterns co-occur:
visual_activity = process_visual(input)
audio_activity = process_audio(input)
cross_modal_strength += learning_rate * visual_activity * audio_activity
```

#### 3. Structural Plasticity
The network grows and prunes connections based on experience:
- **Neurogenesis**: New neurons for novel patterns
- **Synaptogenesis**: New connections for correlated activity
- **Pruning**: Remove weak connections

---

## Text Learning

Atlas can learn language structure from text, similar to how it learns from audio-visual streams.

### Basic Usage

```python
from self_organizing_av_system.core.text_learning import TextLearningModule

# Create text learner
text_learner = TextLearningModule(
    embedding_dim=128,
    context_window=5,
    learning_rate=0.01
)

# Learn from text
result = text_learner.learn_from_text("The cat sat on the mat")
print(f"Learned {result['unique_tokens']} unique tokens")

# Generate text
generated = text_learner.generate_text(prompt="The cat", max_length=20)
print(generated)
```

### How It Works

1. **Tokenization**: Text is split into tokens (words, punctuation)
2. **Context Learning**: The system learns which tokens predict others
3. **Embedding Formation**: Each token develops a distributed representation
4. **Generation**: Predict next token from context, repeat

### Example: Training on a Corpus

```python
# Train on multiple texts
corpus = [
    "Dogs are loyal companions.",
    "Cats are independent animals.",
    "Birds can fly in the sky.",
]

for text in corpus:
    text_learner.learn_from_text(text)

# Check what was learned
stats = text_learner.get_stats()
print(f"Vocabulary size: {stats['vocabulary_size']}")
print(f"Most common: {stats['most_common_tokens'][:5]}")
```

---

## Memory Systems

Atlas implements a dual memory system inspired by human cognition:

### Episodic Memory

Stores specific experiences with context (like the hippocampus):

```python
from self_organizing_av_system.core.episodic_memory import EpisodicMemory

# Create episodic memory
memory = EpisodicMemory(state_size=64, max_episodes=1000)

# Store an experience
episode = memory.store(
    state=current_neural_state,
    sensory_data={"visual": visual_input, "audio": audio_input},
    context={"location": "kitchen", "time": "morning"},
    emotional_valence=0.5
)

# Retrieve similar experiences
similar = memory.retrieve(query_state, k=5)

# Pattern completion (recall from partial cue)
completed = memory.complete(partial_state)
```

### Semantic Memory

Stores abstract concepts and relationships (like the cortex):

```python
from self_organizing_av_system.core.semantic_memory import SemanticMemory, RelationType

# Create semantic memory
semantic = SemanticMemory(embedding_size=64)

# Add concepts
semantic.add_concept("dog", embedding=dog_features)
semantic.add_concept("animal", embedding=animal_features)

# Add relationships
semantic.add_relationship("dog", RelationType.IS_A, "animal")

# Query with spreading activation
activated = semantic.spreading_activation("dog", depth=2)

# Inference
result = semantic.infer("dog", RelationType.IS_A, "animal")  # True
```

### Memory Integration

The two memory systems work together:

```python
# Experiences in episodic memory ground semantic concepts
dog_experiences = episodic.retrieve_by_context({"animal": "dog"})
dog_embedding = np.mean([ep.state for ep in dog_experiences], axis=0)
semantic.add_concept("dog", dog_embedding)

# Semantic knowledge helps interpret new experiences
dog_concept = semantic.get_concept("dog")
similar_episodes = episodic.retrieve(dog_concept.embedding)
```

---

## Reasoning

### Causal Reasoning

Understand cause-effect relationships:

```python
from self_organizing_av_system.core.causal_reasoning import (
    CausalReasoner, CausalRelationType
)

# Create reasoner
reasoner = CausalReasoner()

# Build causal graph
reasoner.graph.add_variable("rain")
reasoner.graph.add_variable("wet_ground")
reasoner.graph.add_link("rain", "wet_ground", CausalRelationType.CAUSES, 0.9)

# Observational: P(Y|X)
effects = reasoner.predict_effects("rain")

# Interventional: P(Y|do(X))
result = reasoner.intervene({"rain": 1.0})

# Counterfactual: What if?
query = CounterfactualQuery(
    factual_evidence={"rain": 1.0, "wet_ground": 1.0},
    hypothetical_intervention={"rain": 0.0},
    query_variable="wet_ground"
)
result = reasoner.counterfactual(query)
```

### Abstract Reasoning

Logical inference, analogy, and pattern recognition:

```python
from self_organizing_av_system.core.abstract_reasoning import (
    AbstractReasoner, Proposition, Rule
)

# Create reasoner
reasoner = AbstractReasoner()

# Add facts
reasoner.knowledge_base.add_fact(
    Proposition(predicate="is_a", arguments=["Socrates", "human"])
)

# Add rules
rule = Rule(
    name="mortality",
    antecedent=[Proposition(predicate="is_a", arguments=["X", "human"])],
    consequent=Proposition(predicate="is_mortal", arguments=["X"]),
    confidence=1.0
)
reasoner.knowledge_base.add_rule(rule)

# Query
result = reasoner.query(
    Proposition(predicate="is_mortal", arguments=["Socrates"])
)
```

---

## Multimodal Learning

### Cross-Modal Associations

Learn relationships between vision and audio:

```python
from self_organizing_av_system.core.cross_modal_association import CrossModalAssociation

# Create association module
cross_modal = CrossModalAssociation(
    visual_size=100,
    audio_size=80,
    learning_rate=0.01
)

# Learn from paired inputs
cross_modal.update(visual_activity, audio_activity, learn=True)

# Predict audio from visual
predicted_audio = cross_modal.predict_audio_from_visual(visual_input)

# Predict visual from audio
predicted_visual = cross_modal.predict_visual_from_audio(audio_input)
```

### Multimodal Integration

Higher-level multimodal binding:

```python
from self_organizing_av_system.core.multimodal_association import (
    MultimodalAssociation, AssociationMode
)

# Create multimodal integrator
multimodal = MultimodalAssociation(
    visual_size=64,
    audio_size=48,
    output_size=32,
    mode=AssociationMode.BINDING
)

# Bind modalities into unified representation
integrated = multimodal.bind(visual_input, audio_input)

# Retrieve from partial input
retrieved = multimodal.retrieve_from_visual(visual_input)
```

---

## Next Steps

### 1. Experiment with Parameters

Try different configurations:

```python
# Larger embeddings for richer representations
text_learner = TextLearningModule(embedding_dim=256)

# Longer context for more complex patterns
text_learner = TextLearningModule(context_window=10)

# Faster or slower learning
text_learner = TextLearningModule(learning_rate=0.001)
```

### 2. Combine Systems

Build more complex cognitive architectures:

```python
# Text learning + Semantic memory
text_learner.learn_from_text(corpus)
for word in text_learner.vocabulary:
    semantic.add_concept(word, text_learner.embeddings[word])

# Episodic + Causal reasoning
for episode in episodic.episodes:
    causal.observe(episode.context, episode.outcome)
```

### 3. Build Custom Applications

```python
# Personal knowledge base
class PersonalAtlas:
    def __init__(self):
        self.text_learner = TextLearningModule()
        self.episodic = EpisodicMemory(state_size=128)
        self.semantic = SemanticMemory(embedding_size=128)
        self.causal = CausalReasoner()
    
    def learn_from_day(self, experiences):
        for exp in experiences:
            self.episodic.store(...)
            self.causal.observe(...)
    
    def answer_question(self, question):
        # Use learned knowledge to answer
        ...
```

### 4. Explore Advanced Features

- **Working Memory**: Active information storage and attention
- **Goal Planning**: Autonomous goal generation and planning
- **Meta-Learning**: Learning to learn - optimize learning strategies
- **World Model**: Internal simulation of environment dynamics

### 5. Join the Community

- GitHub Issues: Report bugs and request features
- GitHub Discussions: Share ideas and get help
- Contribute: Submit pull requests with improvements

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'self_organizing_av_system'**
```bash
# Make sure you're in the Atlas directory
cd /path/to/Atlas
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA out of memory**
```python
# Reduce model size
config = SystemConfig(visual_layer_sizes=[50, 25])
```

**No audio input**
```bash
# Check audio devices
python -c "import pyaudio; p=pa.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"
```

### Getting Help

1. Check the [README.md](README.md) for detailed documentation
2. Run the examples to understand expected behavior
3. Open an issue on GitHub with error details

---

## Summary

Atlas provides a fundamentally different approach to machine learning:

1. **Self-Organizing**: No labels, no predefined objectives
2. **Biologically-Inspired**: Based on how brains actually learn
3. **Multimodal**: Integrates vision, audio, and text
4. **Cognitive**: Includes memory, reasoning, and planning

Start with the demos, experiment with the code, and build your own applications!

Happy learning! 🧠✨
