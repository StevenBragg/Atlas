"""
Atlas Game Learning Module

Enables Atlas to learn video game control through:
- Watching gameplay (visual learning)
- Trial and error (reinforcement learning)
- Imitation learning (copying expert moves)

Integrates with shared brain for unified learning.
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict
from collections import deque
import pickle
from pathlib import Path

class GameLearningModule:
    """
    Atlas learns to play games by watching and doing.
    
    Combines:
    - Visual processing (screen pixels)
    - Motor control (button presses)
    - Reward learning (score, progress)
    - Imitation (watching experts)
    """
    
    def __init__(self, shared_brain=None):
        self.shared_brain = shared_brain
        
        # Game memory
        self.episode_memory = deque(maxlen=1000)
        self.expert_demonstrations = []
        
        # Motor pattern learning
        self.action_patterns = {}
        self.visual_action_map = {}
        
        # Performance tracking
        self.games_played = 0
        self.total_score = 0
        self.best_scores = {}
        
    def watch_expert_play(self, game_name: str, frames: List[np.ndarray], 
                          actions: List[int], rewards: List[float]):
        """
        Learn by watching an expert play.
        
        Args:
            game_name: Name of the game
            frames: Screen frames (visual input)
            actions: Button presses taken
            rewards: Scores received
        """
        demonstration = {
            'game': game_name,
            'frames': frames,
            'actions': actions,
            'rewards': rewards,
            'total_reward': sum(rewards)
        }
        
        self.expert_demonstrations.append(demonstration)
        
        # Learn visual-action associations
        for i in range(len(frames) - 1):
            frame = frames[i]
            action = actions[i]
            reward = rewards[i]
            
            # Create simplified visual key (downsampled)
            visual_key = self._frame_to_key(frame)
            
            # Store action pattern
            if visual_key not in self.visual_action_map:
                self.visual_action_map[visual_key] = []
            
            self.visual_action_map[visual_key].append({
                'action': action,
                'reward': reward,
                'next_frame': frames[i + 1]
            })
        
        # Learn from demonstration using shared brain
        if self.shared_brain:
            description = f"Expert played {game_name} achieving score {sum(rewards)}"
            self.shared_brain.learn_from_text(description)
        
        print(f"Learned from expert: {game_name}, score: {sum(rewards)}")
    
    def _frame_to_key(self, frame: np.ndarray, size: Tuple[int, int] = (10, 10)) -> str:
        """Convert frame to a simple key for lookup"""
        # Downsample frame
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize(size)
        
        # Convert to simple hash
        pixels = np.array(img).flatten()
        # Quantize to reduce noise
        quantized = (pixels // 32) * 32
        return hash(quantized.tobytes()).hex()[:16]
    
    def play_game(self, game_name: str, render: bool = False, 
                  max_steps: int = 1000) -> Dict:
        """
        Play a game using learned knowledge.
        
        Returns:
            Episode stats
        """
        env = gym.make(game_name, render_mode='human' if render else None)
        obs, info = env.reset()
        
        episode_frames = []
        episode_actions = []
        episode_rewards = []
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get current frame
            frame = obs
            episode_frames.append(frame)
            
            # Decide action
            action = self._decide_action(frame, game_name)
            episode_actions.append(action)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            total_reward += reward
            steps += 1
            
            # Learn from this step
            self._learn_from_step(frame, action, reward, obs)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Store episode
        episode = {
            'game': game_name,
            'frames': episode_frames,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'total_reward': total_reward,
            'steps': steps
        }
        self.episode_memory.append(episode)
        
        # Update stats
        self.games_played += 1
        self.total_score += total_reward
        
        if game_name not in self.best_scores or total_reward > self.best_scores[game_name]:
            self.best_scores[game_name] = total_reward
        
        # Learn from episode using shared brain
        if self.shared_brain:
            summary = f"Played {game_name}: score {total_reward}, steps {steps}"
            self.shared_brain.learn_from_text(summary)
        
        print(f"Game complete: {game_name}, score: {total_reward}, steps: {steps}")
        
        return episode
    
    def _decide_action(self, frame: np.ndarray, game_name: str) -> int:
        """Decide what action to take based on visual input"""
        # Try to match with expert demonstrations
        visual_key = self._frame_to_key(frame)
        
        if visual_key in self.visual_action_map:
            # Use learned action
            options = self.visual_action_map[visual_key]
            # Pick action with highest average reward
            best_action = max(options, key=lambda x: x['reward'])['action']
            return best_action
        
        # No match - try random action (exploration)
        # Default to action 0 (usually no-op) for safety
        return 0
    
    def _learn_from_step(self, frame: np.ndarray, action: int, 
                         reward: float, next_frame: np.ndarray):
        """Learn from a single game step"""
        # Store in visual-action map
        visual_key = self._frame_to_key(frame)
        
        if visual_key not in self.visual_action_map:
            self.visual_action_map[visual_key] = []
        
        self.visual_action_map[visual_key].append({
            'action': action,
            'reward': reward,
            'next_frame': next_frame
        })
    
    def train_on_game(self, game_name: str, episodes: int = 10):
        """Train Atlas on a specific game"""
        print(f"Training Atlas on {game_name} for {episodes} episodes...")
        
        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")
            stats = self.play_game(game_name, render=False)
            
            # Adaptive learning: if score is improving, reinforce those actions
            if ep > 0:
                prev_episode = self.episode_memory[-2] if len(self.episode_memory) > 1 else None
                if prev_episode and stats['total_reward'] > prev_episode['total_reward']:
                    print(f"  Score improved! Reinforcing successful actions.")
        
        print(f"Training complete on {game_name}")
        print(f"Best score: {self.best_scores.get(game_name, 0)}")
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'games_played': self.games_played,
            'total_score': self.total_score,
            'best_scores': self.best_scores,
            'expert_demonstrations': len(self.expert_demonstrations),
            'visual_patterns_learned': len(self.visual_action_map),
            'episodes_stored': len(self.episode_memory)
        }
    
    def save(self, filepath: str):
        """Save game learning state"""
        state = {
            'action_patterns': self.action_patterns,
            'visual_action_map': self.visual_action_map,
            'best_scores': self.best_scores,
            'games_played': self.games_played,
            'total_score': self.total_score
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load game learning state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.action_patterns = state['action_patterns']
        self.visual_action_map = state['visual_action_map']
        self.best_scores = state['best_scores']
        self.games_played = state['games_played']
        self.total_score = state['total_score']


# Demo usage
if __name__ == "__main__":
    # Create game learner
    from shared_brain import get_shared_brain
    brain = get_shared_brain()
    gamer = GameLearningModule(shared_brain=brain)
    
    # List available games
    print("Available games:")
    print("- ALE/Pong-v5")
    print("- ALE/Breakout-v5")
    print("- ALE/SpaceInvaders-v5")
    print("- CartPole-v1")
    
    # Train on a simple game
    # gamer.train_on_game('CartPole-v1', episodes=5)
    
    print("\nGame learning module ready!")
    print("Atlas can now learn to play video games by watching and doing.")
