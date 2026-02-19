#!/usr/bin/env python3
"""
Atlas Game Trainer

Trains Atlas to play video games using reinforcement learning
and imitation learning. All learning goes to the shared brain.
"""

import sys
import time
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from core.game_learning import GameLearningModule
from shared_brain import get_shared_brain, save_shared_brain

def main():
    print("🎮 Atlas Game Trainer")
    print("=" * 50)
    
    # Get shared brain
    brain = get_shared_brain()
    print(f"Loaded brain with {brain.get_stats()['vocabulary_size']} words")
    
    # Create game learner
    gamer = GameLearningModule(shared_brain=brain)
    
    # Available games for learning
    games = [
        'CartPole-v1',  # Easy balancing game
        # 'ALE/Pong-v5',  # Classic Atari (requires ROM)
        # 'ALE/Breakout-v5',
    ]
    
    print("\nAvailable games:")
    for i, game in enumerate(games, 1):
        print(f"  {i}. {game}")
    
    # Train on each game
    for game_name in games:
        print(f"\n🎯 Training on {game_name}...")
        try:
            gamer.train_on_game(game_name, episodes=10)
            
            # Save progress
            save_shared_brain()
            gamer.save(f'/root/.openclaw/workspace/Atlas/game_state_{game_name.replace("/", "_")}.pkl')
            
            print(f"✅ {game_name} training complete!")
            
        except Exception as e:
            print(f"❌ Error training on {game_name}: {e}")
            continue
        
        time.sleep(2)
    
    # Show final stats
    stats = gamer.get_stats()
    print("\n📊 Final Game Learning Stats:")
    print(f"  Games played: {stats['games_played']}")
    print(f"  Total score: {stats['total_score']}")
    print(f"  Best scores: {stats['best_scores']}")
    print(f"  Visual patterns learned: {stats['visual_patterns_learned']}")
    
    print("\n🧠 Game learning added to shared brain!")
    print("Atlas can now understand game concepts like:")
    print("  - Score, reward, winning/losing")
    print("  - Controller inputs, actions")
    print("  - Visual patterns on screen")
    print("  - Strategy and timing")

if __name__ == "__main__":
    main()
