#!/usr/bin/env python3
"""
Atlas Autonomous Learning System

Runs Atlas continuously, teaching it to read/write/improve its own code.
Designed to run 24/7 and make progress while you sleep.
"""

import os
import sys
import time
import json
import random
import logging
from datetime import datetime
from pathlib import Path

# Add Atlas to path
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from core.text_learning import TextLearningModule
from core.episodic_memory import EpisodicMemory
from core.semantic_memory import SemanticMemory
from core.creativity import CreativityEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/.openclaw/workspace/Atlas/logs/autonomous.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AtlasAutonomous')


class AutonomousAtlas:
    """
    Self-improving Atlas that learns from:
    1. Reading its own source code
    2. Generating new ideas
    3. Testing improvements
    4. Remembering what works
    """
    
    def __init__(self):
        self.text_learner = TextLearningModule(embedding_dim=256)
        self.episodic_memory = EpisodicMemory(state_size=256, max_episodes=1000)
        self.semantic_memory = SemanticMemory(embedding_size=256)
        self.creativity = CreativityEngine(embedding_dim=256)
        
        self.codebase_path = Path('/root/.openclaw/workspace/Atlas/self_organizing_av_system')
        self.improvements_path = Path('/root/.openclaw/workspace/Atlas/improvements')
        self.improvements_path.mkdir(exist_ok=True)
        
        self.learning_stats = {
            'code_files_read': 0,
            'lines_learned': 0,
            'ideas_generated': 0,
            'improvements_attempted': 0,
            'start_time': datetime.now().isoformat()
        }
        
    def read_and_learn_from_codebase(self):
        """Read Python files and learn from them"""
        logger.info("📚 Reading codebase...")
        
        python_files = list(self.codebase_path.rglob('*.py'))
        random.shuffle(python_files)
        
        for file_path in python_files[:10]:  # Learn from 10 files per cycle
            try:
                with open(file_path) as f:
                    content = f.read()
                    
                # Learn from the code
                result = self.text_learner.learn_from_text(content)
                
                # Store in episodic memory
                import numpy as np
                self.episodic_memory.store(
                    state=np.array(self.text_learner.embeddings.get(0, [0] * 256)),
                    sensory_data={'text': np.array([0.0])},
                    context={
                        'type': 'code_reading',
                        'file': str(file_path),
                        'timestamp': datetime.now().isoformat(),
                        'lines': len(content.split('\n'))
                    }
                )
                
                self.learning_stats['code_files_read'] += 1
                self.learning_stats['lines_learned'] += len(content.split('\n'))
                
                logger.info(f"  Learned from {file_path.name}: {result['tokens_processed']} tokens")
                
            except Exception as e:
                logger.warning(f"  Failed to read {file_path}: {e}")
                
    def generate_improvement_ideas(self):
        """Generate ideas for improving Atlas"""
        logger.info("💡 Generating improvement ideas...")
        
        # Get stats to inform generation
        stats = self.text_learner.get_stats()
        
        # Generate ideas using creativity engine
        import numpy as np
        problem_vector = np.random.randn(256)  # Random seed for idea generation
        
        ideas = self.creativity.divergent_think(
            problem=problem_vector,
            n_solutions=5,
            diversity_weight=0.7
        )
        
        self.learning_stats['ideas_generated'] += len(ideas)
        
        # Store ideas
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ideas_file = self.improvements_path / f'ideas_{timestamp}.json'
        
        with open(ideas_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'stats': stats,
                'ideas': len(ideas)
            }, f, indent=2)
            
        logger.info(f"  Generated {len(ideas)} ideas, saved to {ideas_file}")
        return ideas
        
    def attempt_code_improvement(self):
        """Try to write a small improvement"""
        logger.info("🔧 Attempting code improvement...")
        
        # Generate some code based on what was learned
        prompt = "def improve_atlas_"
        generated = self.text_learner.generate_text(prompt, max_length=100)
        
        if generated and len(generated) > len(prompt):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            improvement_file = self.improvements_path / f'improvement_{timestamp}.py'
            
            with open(improvement_file, 'w') as f:
                f.write(f'# Autonomous improvement generated by Atlas\n')
                f.write(f'# Timestamp: {datetime.now().isoformat()}\n\n')
                f.write(generated)
                
            self.learning_stats['improvements_attempted'] += 1
            logger.info(f"  Created improvement: {improvement_file}")
            
    def self_reflection(self):
        """Reflect on learning progress"""
        logger.info("🤔 Self-reflection...")
        
        stats = self.text_learner.get_stats()
        
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'learning_stats': self.learning_stats,
            'text_stats': stats,
            'memory_stats': {
                'episodic_memories': len(self.episodic_memory.memories) if hasattr(self.episodic_memory, 'memories') else 'N/A',
            }
        }
        
        reflection_file = self.improvements_path / 'reflection_latest.json'
        with open(reflection_file, 'w') as f:
            json.dump(reflection, f, indent=2)
            
        logger.info(f"  Vocabulary: {stats['vocabulary_size']}")
        logger.info(f"  Files read: {self.learning_stats['code_files_read']}")
        logger.info(f"  Ideas generated: {self.learning_stats['ideas_generated']}")
        
    def run_learning_cycle(self):
        """Run one complete learning cycle"""
        logger.info("=" * 60)
        logger.info("🚀 Starting Atlas Autonomous Learning Cycle")
        logger.info("=" * 60)
        
        try:
            # 1. Learn from codebase
            self.read_and_learn_from_codebase()
            
            # 2. Generate improvement ideas
            self.generate_improvement_ideas()
            
            # 3. Attempt code improvements
            self.attempt_code_improvement()
            
            # 4. Self-reflection
            self.self_reflection()
            
            # 5. Save state
            self.save_state()
            
            logger.info("✅ Learning cycle complete")
            
        except Exception as e:
            logger.error(f"❌ Error in learning cycle: {e}", exc_info=True)
            
    def save_state(self):
        """Save learning state"""
        state_path = self.improvements_path / 'state'
        state_path.mkdir(exist_ok=True)
        
        self.text_learner.save_state(state_path / 'text_learner.pkl')
        self.episodic_memory.save(state_path / 'episodic_memory.pkl')
        
        with open(state_path / 'learning_stats.json', 'w') as f:
            json.dump(self.learning_stats, f, indent=2)
            
        logger.info("  State saved")
        
    def load_state(self):
        """Load previous learning state"""
        state_path = self.improvements_path / 'state'
        
        try:
            if (state_path / 'text_learner.pkl').exists():
                self.text_learner.load_state(state_path / 'text_learner.pkl')
                logger.info("Loaded text learner state")
                
            if (state_path / 'episodic_memory.pkl').exists():
                self.episodic_memory.load(state_path / 'episodic_memory.pkl')
                logger.info("Loaded episodic memory")
                
            if (state_path / 'learning_stats.json').exists():
                with open(state_path / 'learning_stats.json') as f:
                    self.learning_stats = json.load(f)
                logger.info("Loaded learning stats")
                
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            
    def run_continuously(self, cycle_interval_minutes=30):
        """Run learning cycles continuously"""
        logger.info("🌙 Atlas Autonomous Mode Started")
        logger.info(f"Running learning cycles every {cycle_interval_minutes} minutes")
        logger.info("Press Ctrl+C to stop")
        
        # Load previous state
        self.load_state()
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('='*60)
            
            self.run_learning_cycle()
            
            logger.info(f"💤 Sleeping for {cycle_interval_minutes} minutes...")
            time.sleep(cycle_interval_minutes * 60)


def main():
    """Main entry point"""
    atlas = AutonomousAtlas()
    
    # Run continuously with 30-minute cycles
    try:
        atlas.run_continuously(cycle_interval_minutes=30)
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping Atlas Autonomous Mode")
        atlas.save_state()
        logger.info("Final state saved. Goodbye!")


if __name__ == "__main__":
    main()
