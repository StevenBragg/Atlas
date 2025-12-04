"""Test episode persistence to ChromaDB."""
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_organizing_av_system.database.vector_store import VectorStore

DATA_DIR = r'C:\Users\sabragg\Desktop\self_learn\Atlas\atlas_data'

print("=" * 60)
print("TESTING EPISODE PERSISTENCE")
print("=" * 60)

# Create VectorStore
vs = VectorStore(data_dir=DATA_DIR, embedding_dim=128)
print(f"VectorStore created: using_chromadb={vs._using_chromadb}")

# Current episode count
count_before = vs.count_episodes()
print(f"Episodes before test: {count_before}")

# Test adding an episode
test_id = f"test_episode_{np.random.randint(10000)}"
test_embedding = np.random.randn(128).astype(np.float32)
test_metadata = {
    'timestamp': 1234567890.0,
    'emotional_valence': 0.5,
    'surprise_level': 0.3,
    'replay_count': 0,
    'consolidation_strength': 0.0,
    'context': {'challenge_name': 'test', 'source': 'test'},
}

print(f"\nAdding test episode: {test_id}")
print(f"  Embedding shape: {test_embedding.shape}")
print(f"  Metadata: {test_metadata}")

try:
    result = vs.add_episode(
        episode_id=test_id,
        embedding=test_embedding,
        metadata=test_metadata
    )
    print(f"  Result: {result}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Check count after
count_after = vs.count_episodes()
print(f"\nEpisodes after test: {count_after}")

if count_after > count_before:
    print("SUCCESS - Episode was added!")
else:
    print("FAILED - Episode was NOT added")

# Try to retrieve it
print(f"\nSearching for test episode...")
results = vs.search_episodes(test_embedding, n_results=5)
print(f"Search returned {len(results)} results")
for r in results:
    print(f"  - {r.id}: similarity={r.similarity:.3f}")

vs.close()
print("\n" + "=" * 60)
