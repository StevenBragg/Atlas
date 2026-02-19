#!/usr/bin/env python3
"""
Interactive Atlas Teacher

Chat with Atlas and teach it through conversation.
It learns from your messages and responds based on what you've taught it.
"""

import os
import sys
import json
import readline
from datetime import datetime
from pathlib import Path

# Add Atlas to path
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from core.text_learning import TextLearningModule
from core.episodic_memory import EpisodicMemory
from core.semantic_memory import SemanticMemory

class InteractiveAtlas:
    """Atlas that learns from chatting with you"""
    
    def __init__(self):
        print("🧠 Loading Atlas...")
        
        # Initialize learning systems
        self.text_learner = TextLearningModule(embedding_dim=256)
        self.episodic_memory = EpisodicMemory(state_size=256, max_episodes=1000)
        self.semantic_memory = SemanticMemory(embedding_size=256)
        
        # Conversation history
        self.conversation_file = Path('/root/.openclaw/workspace/Atlas/conversations.jsonl')
        self.load_conversations()
        
        # Teaching stats
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'words_learned': 0,
            'conversations': 0
        }
        
        print("✅ Atlas is ready! Type 'help' for commands.\n")
        
    def load_conversations(self):
        """Load previous conversations"""
        if self.conversation_file.exists():
            with open(self.conversation_file) as f:
                for line in f:
                    try:
                        msg = json.loads(line)
                        # Learn from previous conversations
                        if msg.get('role') == 'user':
                            self.text_learner.learn_from_text(msg['content'])
                    except:
                        pass
            print(f"📚 Loaded previous conversations")
    
    def save_message(self, role, content):
        """Save message to conversation log"""
        with open(self.conversation_file, 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'role': role,
                'content': content
            }, f)
            f.write('\n')
    
    def learn_from_message(self, message):
        """Learn from user's message"""
        result = self.text_learner.learn_from_text(message)
        self.stats['messages_received'] += 1
        self.stats['words_learned'] += result['tokens_processed']
        
        # Store in episodic memory
        import numpy as np
        self.episodic_memory.store(
            state=np.random.randn(256),  # Simplified for now
            sensory_data={'text': np.array([0.0])},
            context={
                'type': 'conversation',
                'message': message[:100],
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return result
    
    def generate_response(self, message):
        """Generate a response based on what Atlas has learned"""
        # Learn from the message first
        learn_result = self.learn_from_message(message)
        
        # Generate response
        vocab_size = self.text_learner.get_stats()['vocabulary_size']
        
        # Simple response logic based on message content
        message_lower = message.lower()
        
        if 'hello' in message_lower or 'hi' in message_lower:
            response = f"Hello! I've learned {vocab_size} words so far. What would you like to teach me?"
            
        elif 'how are you' in message_lower:
            response = f"I'm learning! I've processed {self.stats['messages_received']} messages and learned {self.stats['words_learned']} words."
            
        elif 'what have you learned' in message_lower:
            stats = self.text_learner.get_stats()
            response = f"So far I've learned {stats['vocabulary_size']} unique words from {stats['total_tokens_seen']} total words. Keep teaching me!"
            
        elif 'help' in message_lower:
            response = """Available commands:
- Just chat with me normally
- 'stats' - See my learning progress
- 'save' - Save my current state
- 'generate' - I'll generate some text
- 'quit' - Exit the conversation"""
            
        elif message_lower == 'stats':
            stats = self.text_learner.get_stats()
            response = f"""📊 My Learning Stats:
- Vocabulary: {stats['vocabulary_size']} words
- Total words seen: {stats['total_tokens_seen']}
- Your messages: {self.stats['messages_received']}
- Words learned from you: {self.stats['words_learned']}"""
            
        elif message_lower == 'generate':
            generated = self.text_learner.generate_text(max_length=50)
            response = f"📝 Generated: {generated}"
            
        elif message_lower == 'save':
            self.save_state()
            response = "💾 State saved!"
            
        else:
            # Generate a contextual response
            prompt = message[:20]
            generated = self.text_learner.generate_text(prompt, max_length=30)
            
            if generated and len(generated) > len(prompt):
                response = generated
            else:
                response = f"That's interesting! I'm learning from everything you tell me. I've now learned {vocab_size} words."
        
        self.stats['messages_sent'] += 1
        return response
    
    def save_state(self):
        """Save learning state"""
        state_dir = Path('/root/.openclaw/workspace/Atlas/teacher_state')
        state_dir.mkdir(exist_ok=True)
        
        self.text_learner.save_state(state_dir / 'text_learner.pkl')
        
        with open(state_dir / 'stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print("💾 State saved!")
    
    def chat_loop(self):
        """Main chat loop"""
        print("=" * 60)
        print("🎓 Interactive Atlas Teacher")
        print("=" * 60)
        print("Teach Atlas by chatting with it!")
        print("Type 'quit' to exit, 'help' for commands.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Save user message
                self.save_message('user', user_input)
                
                # Check for quit
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Atlas will remember what you taught it.")
                    self.save_state()
                    break
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"Atlas: {response}\n")
                
                # Save Atlas response
                self.save_message('assistant', response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.save_state()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    atlas = InteractiveAtlas()
    atlas.chat_loop()

if __name__ == "__main__":
    main()
