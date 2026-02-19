#!/usr/bin/env python3
"""
Atlas Story Trainer

Trains Atlas to generate coherent text by feeding it stories, books,
and natural language instead of code.
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from shared_brain import get_shared_brain, save_shared_brain

# Classic stories and literature for Atlas to learn
STORIES = [
    """Once upon a time, in a land far away, there lived a young prince who dreamed of adventure. 
    Every day he would gaze out from the castle towers, wondering what lay beyond the mountains. 
    One morning, a traveling bard arrived with tales of dragons and treasure. 
    The prince knew his destiny awaited him beyond the castle walls.""",
    
    """The old lighthouse keeper had watched the sea for forty years. 
    He knew every storm, every ship, every soul that passed his light. 
    On moonless nights, he would tell stories to the waves, and sometimes, 
    he swore the waves whispered back their own ancient tales.""",
    
    """In the heart of the enchanted forest stood a tree unlike any other. 
    Its leaves shimmered with silver light, and its roots reached deep into the earth's memories. 
    Travelers who rested beneath its branches would wake with visions of the past and future, 
    their dreams woven together like threads in an infinite tapestry.""",
    
    """The library was silent except for the whisper of turning pages. 
    Sarah had discovered a book that seemed to write itself as she read, 
    its words changing to tell her own story, her own fears and hopes. 
    She realized the book was magic, but more than that, it was alive.""",
    
    """Long ago, before maps were drawn, there was a city built on clouds. 
    Its people walked on mist and drank from rainbows. 
    They had forgotten the ground existed until a child fell through the clouds 
    and discovered the world below, a world of green and brown and endless wonder.""",
    
    """The clockmaker's daughter could hear time. 
    To her, every tick was a note, every chime a melody. 
    She spent her days tuning the clocks of the city, 
    ensuring that time sang in harmony rather than chaos. 
    But one clock, deep in the cellar, played a song she'd never heard before.""",
    
    """Every night, the stars would come down to the lake to drink. 
    They would shimmer and dance on the water's surface, 
    sharing stories of distant worlds and ancient light. 
    A lonely fisherman once stayed awake to watch them, 
    and in return, they taught him the language of the cosmos.""",
    
    """The garden had been abandoned for a hundred years, 
    yet the flowers still bloomed in impossible colors. 
    A young botanist discovered that each plant remembered the hands that planted it, 
    growing not from soil but from love and memory. 
    She spent her life learning their stories, one petal at a time."""
]

def train_on_stories():
    """Train Atlas on narrative text"""
    brain = get_shared_brain()
    
    print("📚 Training Atlas on stories...")
    initial_vocab = brain.get_stats()['vocabulary_size']
    
    for i, story in enumerate(STORIES, 1):
        print(f"\n[{i}/{len(STORIES)}] Learning story...")
        brain.learn_from_text(story)
        
        # Test generation after each story
        if i % 2 == 0:
            test = brain.generate_text("Once upon a time", max_length=50)
            print(f"   Test: {test[:80]}...")
    
    save_shared_brain()
    
    final_vocab = brain.get_stats()['vocabulary_size']
    print(f"\n✅ Training complete!")
    print(f"   Vocabulary: {initial_vocab} → {final_vocab} (+{final_vocab - initial_vocab})")
    
    # Final story test
    print("\n📝 Atlas writes a story:")
    story = brain.generate_text("Once upon a time", max_length=100)
    print(story)

if __name__ == "__main__":
    train_on_stories()
