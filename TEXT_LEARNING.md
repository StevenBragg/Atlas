# Text Learning Feature Documentation

## Overview

The Text Learning feature enables Atlas to learn from and generate text, extending its self-organizing capabilities beyond audio-visual processing. Text is treated as a sequential sensory stream where characters and words become sensory tokens, context provides temporal structure, and meaning emerges from statistical patterns.

### Key Capabilities

1. **Language Structure Learning** - Learn from text corpora using predictive coding principles
2. **Conceptual Grounding** - Ground words in conceptual space through distributed representations
3. **Text Generation** - Generate coherent text responses from learned patterns
4. **Human Communication** - Enable conversational interaction with humans

### How It Works

The Text Learning Module uses biologically-inspired learning mechanisms similar to Atlas's audio-visual pathways:

- **Predictive Coding**: Predicts the next token based on context
- **Hebbian Learning**: Co-occurring tokens strengthen their connections
- **Structural Plasticity**: Vocabulary grows dynamically as needed
- **Distributed Representations**: Each token has an embedding vector

---

## API Endpoint Documentation

### Base URL

```
http://localhost:8000  (when running locally)
```

### Endpoints

#### 1. POST /text/learn

Learn from provided text input.

**Request Body:**
```json
{
  "text": "The quick brown fox jumps over the lazy dog"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "tokens_processed": 9,
    "unique_tokens": 8,
    "vocabulary_size": 42,
    "avg_prediction_error": 0.234,
    "predictions": [
      {
        "context": ["the"],
        "predicted": "quick",
        "error": 0.189
      }
    ]
  }
}
```

**Fields:**
- `tokens_processed`: Number of tokens learned from
- `unique_tokens`: Count of unique tokens in input
- `vocabulary_size`: Current total vocabulary size
- `avg_prediction_error`: Mean prediction error across tokens
- `predictions`: Sample predictions showing context → target mappings

---

#### 2. POST /text/generate

Generate text based on a prompt.

**Request Body:**
```json
{
  "prompt": "Hello world",
  "max_length": 50
}
```

**Response:**
```json
{
  "success": true,
  "generated": "hello world this is a test of neural networks",
  "prompt": "Hello world"
}
```

**Fields:**
- `prompt`: The input prompt (optional, defaults to empty)
- `max_length`: Maximum number of tokens to generate (default: 50)
- `generated`: The generated text output

---

#### 3. GET /text/stats

Retrieve learning statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "vocabulary_size": 156,
    "total_tokens_seen": 5000,
    "unique_contexts": 892,
    "most_common_tokens": [
      ["the", 450],
      ["is", 320],
      ["a", 280]
    ]
  }
}
```

**Fields:**
- `vocabulary_size`: Number of unique tokens in vocabulary
- `total_tokens_seen`: Cumulative tokens processed
- `unique_contexts`: Number of learned context patterns
- `most_common_tokens`: Top 10 most frequent tokens with counts

---

#### 4. POST /chat

Conversational interface that learns from input and generates a response.

**Request Body:**
```json
{
  "message": "How does machine learning work?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "machine learning works by finding patterns in data",
  "learned_tokens": 6
}
```

**Fields:**
- `message`: User's input message
- `response`: Generated response
- `learned_tokens`: Number of tokens learned from the input

---

## How to Teach Atlas Through Text

### Basic Learning

Atlas learns from text through exposure, similar to how it learns from audio-visual streams:

```python
from self_organizing_av_system.core.text_learning import TextLearningModule

# Initialize the text learning module
text_module = TextLearningModule(
    embedding_dim=128,      # Size of token representations
    max_vocabulary=10000,   # Maximum vocabulary size
    context_window=5,       # Tokens of context for predictions
    learning_rate=0.01      # Learning rate for weight updates
)

# Teach Atlas with text
text_module.learn_from_text("Machine learning is fascinating")
text_module.learn_from_text("Neural networks process information")
```

### Progressive Learning

For best results, teach Atlas progressively:

1. **Start Simple**: Begin with short, clear sentences
2. **Build Vocabulary**: Expose to domain-specific terminology
3. **Provide Context**: Related texts help build coherent associations
4. **Reinforce Patterns**: Repeated exposure strengthens connections

### Example Teaching Session

```python
# Initialize
from self_organizing_av_system.core.text_learning import TextLearningModule
text_module = TextLearningModule()

# Teaching sequence
teaching_texts = [
    "Hello world this is a test",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is fascinating",
    "Neural networks process information",
    "Deep learning uses multiple layers",
    "Artificial intelligence mimics human cognition"
]

# Learn from each text
for text in teaching_texts:
    result = text_module.learn_from_text(text)
    print(f"Learned: {result['tokens_processed']} tokens")

# Check statistics
stats = text_module.get_stats()
print(f"Vocabulary size: {stats['vocabulary_size']}")
```

### Saving and Loading Progress

```python
# Save learned state
text_module.save_state("text_model.pkl")

# Load in a new session
new_module = TextLearningModule()
new_module.load_state("text_model.pkl")
```

---

## Example curl Commands

### Learn from Text

```bash
curl -X POST http://localhost:8000/text/learn \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog"
  }'
```

### Generate Text

```bash
curl -X POST http://localhost:8000/text/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_length": 30
  }'
```

### Get Statistics

```bash
curl http://localhost:8000/text/stats
```

### Chat Interaction

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?"
  }'
```

### Batch Learning Example

```bash
# Learn multiple texts in sequence
curl -X POST http://localhost:8000/text/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "Artificial intelligence is transforming technology"}'

curl -X POST http://localhost:8000/text/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "Neural networks learn from data patterns"}'

curl -X POST http://localhost:8000/text/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "Deep learning enables complex problem solving"}'

# Generate based on learned patterns
curl -X POST http://localhost:8000/text/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Artificial intelligence", "max_length": 20}'
```

---

## Expected Behavior

### Learning Phase

1. **Tokenization**: Text is split into words and punctuation
2. **Vocabulary Growth**: New tokens are added to vocabulary automatically
3. **Context Learning**: The system learns which tokens predict others
4. **Error Reduction**: Prediction errors decrease with exposure

### Generation Phase

1. **Context-Based**: Generated text depends on learned contexts
2. **Probabilistic**: Output varies based on learned patterns
3. **Coherence**: Quality improves with more training data
4. **Vocabulary-Limited**: Can only use learned tokens

### Typical Learning Curve

- **Initial**: High prediction errors, limited vocabulary
- **After 100 tokens**: Basic word associations form
- **After 1000 tokens**: Simple phrase generation possible
- **After 10000+ tokens**: Coherent sentence generation

---

## Limitations

### Current Limitations

1. **Simple Tokenization**: Uses basic regex-based word splitting
   - No subword tokenization (BPE, WordPiece)
   - Limited handling of contractions and special characters

2. **Context Window**: Fixed-size context (default: 5 tokens)
   - Cannot capture long-range dependencies
   - May lose coherence in longer sequences

3. **Vocabulary Management**: 
   - Maximum vocabulary size is fixed
   - Rare tokens may be replaced when limit reached

4. **No Semantic Understanding**:
   - Learns statistical patterns, not meaning
   - Cannot reason about learned content
   - No world knowledge integration

5. **Training Data Requirements**:
   - Requires significant text for quality generation
   - Quality depends on training corpus diversity

6. **Computational Constraints**:
   - Embedding-based similarity search can be slow with large vocabularies
   - No GPU acceleration in current implementation

### Best Practices

1. **Preprocess Text**: Clean input text for best results
2. **Consistent Domain**: Train on related content for coherent output
3. **Monitor Vocabulary**: Check stats to avoid vocabulary overflow
4. **Save Checkpoints**: Regularly save learned state
5. **Start Small**: Test with small datasets before scaling

### Comparison to Traditional NLP

| Aspect | Atlas Text Learning | Traditional LLMs |
|--------|---------------------|------------------|
| Training | Unsupervised, online | Pre-trained, batch |
| Architecture | Self-organizing | Transformer-based |
| Scale | Small vocabulary | Billions of parameters |
| Understanding | Statistical patterns | Semantic reasoning |
| Use Case | Learning demonstration | Production NLP |

---

## Integration with Atlas

The Text Learning Module integrates with Atlas's broader cognitive architecture:

```python
from self_organizing_av_system.core.text_learning import TextLearningModule
from cloud.text_api import TextLearningAPI

# Create module
text_module = TextLearningModule()

# Create API handler
text_api = TextLearningAPI(text_module)

# Use in your application
result = text_api.handle_learn({"text": "Hello world"})
response = text_api.handle_chat({"message": "Hi there"})
```

### Future Enhancements

- Integration with multimodal association layer
- Cross-modal grounding (text ↔ visual concepts)
- Hierarchical language understanding
- Attention mechanisms for better context handling
- Subword tokenization support
