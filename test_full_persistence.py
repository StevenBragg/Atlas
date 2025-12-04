"""Test full persistence path from KnowledgeBase to databases."""
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_organizing_av_system.database.vector_store import VectorStore
from self_organizing_av_system.database.graph_store import GraphStore
from self_organizing_av_system.core.knowledge_base import KnowledgeBase

DATA_DIR = r'C:\Users\sabragg\Desktop\self_learn\Atlas\atlas_data'

print("=" * 60)
print("TESTING FULL PERSISTENCE PATH")
print("=" * 60)

# Create stores
print("\n1. Creating stores...")
vs = VectorStore(data_dir=DATA_DIR, embedding_dim=128)
gs = GraphStore(data_dir=DATA_DIR)
print(f"   VectorStore: using_chromadb={vs._using_chromadb}")
print(f"   GraphStore: created")

# Check initial counts
print("\n2. Initial counts:")
print(f"   Episodes: {vs.count_episodes()}")
print(f"   Concepts: {vs.count_concepts()}")

# Create KnowledgeBase with stores
print("\n3. Creating KnowledgeBase with stores...")
kb = KnowledgeBase(
    state_dim=128,
    vector_store=vs,
    graph_store=gs,
    enable_consolidation=False,  # Disable to simplify test
)

# Check episodic memory persistence flag
print(f"   KB._persistence_enabled: {kb._persistence_enabled}")
print(f"   KB.episodic._persistence_enabled: {kb.episodic._persistence_enabled}")
print(f"   KB.episodic.vector_store is not None: {kb.episodic.vector_store is not None}")
print(f"   KB.semantic._persistence_enabled: {kb.semantic._persistence_enabled}")

# Store a test experience
print("\n4. Storing test experience...")
test_state = np.random.randn(128).astype(np.float32)
kb.store_experience(
    state=test_state,
    context={'challenge_name': 'test_challenge', 'accuracy': 0.85},
    emotional_valence=0.5,
    source='curriculum',
)
print(f"   Experience stored. total_experiences_stored: {kb.total_experiences_stored}")

# Check counts after
print("\n5. Counts after store_experience:")
print(f"   Episodes in memory: {len(kb.episodic.episodes)}")
print(f"   Episodes in ChromaDB: {vs.count_episodes()}")
print(f"   Concepts in memory: {len(kb.semantic.concepts)}")
print(f"   Concepts in ChromaDB: {vs.count_concepts()}")

# Check if episode was actually persisted
print("\n6. Checking if episode was persisted...")
if kb.episodic.episodes:
    last_episode = kb.episodic.episodes[-1]
    print(f"   Last episode ID: {last_episode.episode_id}")

    # Try to find it in ChromaDB
    results = vs.search_episodes(last_episode.state, n_results=3)
    print(f"   Search results: {len(results)}")
    for r in results:
        print(f"     - {r.id}: similarity={r.similarity:.3f}")
        if r.id == last_episode.episode_id:
            print("       FOUND! Episode was persisted correctly!")

# Clean up
kb.stop()
vs.close()
gs.close()

print("\n" + "=" * 60)
