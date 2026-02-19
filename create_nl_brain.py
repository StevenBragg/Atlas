#!/usr/bin/env python3
"""
Atlas Natural Language Brain

Creates a separate brain instance focused purely on natural language,
without the code pollution from the main brain.
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')
from core.text_learning import TextLearningModule

# Classic literature and stories
CORPUS = [
    # Fairy tales
    "Once upon a time, in a kingdom far away, there lived a princess who loved to explore the forests near her castle.",
    "The little girl walked through the woods, carrying a basket of food for her grandmother who lived in a cottage.",
    "A young boy climbed a magical beanstalk that reached up into the clouds where a giant lived in a castle.",
    
    # Adventure
    "The ship sailed across the vast ocean, its crew searching for new lands and treasures beyond the horizon.",
    "Deep in the jungle, the explorer discovered ancient ruins covered in vines and mysterious symbols.",
    "The mountain climber reached the summit just as the sun rose, painting the sky in brilliant colors.",
    
    # Fantasy
    "The wizard raised his staff and spoke words of power, causing the air to shimmer with magical energy.",
    "Dragons soared through the sky, their scales gleaming like precious gems in the sunlight.",
    "The enchanted forest was filled with creatures that spoke in whispers and danced in moonlight.",
    
    # Mystery
    "The detective examined the clues carefully, piecing together the puzzle of the missing painting.",
    "In the old mansion, doors creaked open by themselves and shadows moved in the corners of vision.",
    "The secret message was hidden in plain sight, written in a code that only the initiated could read.",
    
    # Romance
    "Their eyes met across the crowded room, and in that moment, the world seemed to fade away.",
    "She waited by the window, watching for his return, her heart filled with hope and longing.",
    "The two lovers walked hand in hand through the garden, speaking of dreams and future days.",
    
    # Science Fiction
    "The spaceship traveled through the stars, its crew in cryogenic sleep for the long journey ahead.",
    "Robots served the city, maintaining order while humans pursued art and leisure.",
    "The time traveler stepped through the portal, not knowing what era awaited on the other side.",
    
    # Nature
    "The river flowed gently through the valley, carrying leaves and memories toward the distant sea.",
    "Autumn painted the forest in shades of gold and crimson as the wind whispered through the trees.",
    "The wolf howled at the moon, its voice echoing across the silent, snow-covered mountains.",
    
    # Emotions
    "Joy bubbled up inside her like a spring, overflowing into laughter that filled the room.",
    "Grief lay heavy on his heart, a stone that time would slowly wear away but never fully remove.",
    "Hope flickered like a candle in the darkness, fragile but refusing to be extinguished."
]

def create_natural_language_brain():
    """Create a fresh brain trained only on natural language"""
    
    print("🧠 Creating natural language brain...")
    brain = TextLearningModule(embedding_dim=256, max_vocabulary=5000)
    
    # Learn all stories
    for i, text in enumerate(CORPUS, 1):
        print(f"Learning text {i}/{len(CORPUS)}...")
        brain.learn_from_text(text)
    
    # Save the NL brain
    nl_brain_path = Path('/root/.openclaw/workspace/Atlas/natural_language_brain.pkl')
    brain.save_state(nl_brain_path)
    
    stats = brain.get_stats()
    print(f"\n✅ Natural language brain created!")
    print(f"   Vocabulary: {stats['vocabulary_size']} words")
    print(f"   Total tokens: {stats['total_tokens_seen']}")
    
    # Test story generation
    print("\n📝 Generated story:")
    story = brain.generate_text("Once upon a time", max_length=80)
    print(story)
    
    return brain

if __name__ == "__main__":
    create_natural_language_brain()
