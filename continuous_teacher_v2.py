#!/usr/bin/env python3
"""
Atlas Continuous Teacher - Fixed Version v2

Uses the ACTUAL shared brain singleton to ensure all learning
is persisted to the same brain instance.

FIXES:
- Added file locking to prevent race conditions
- Added single instance check using PID file
- Forces brain reload before each session to get latest data
- Improved logging for debugging
"""

import sys
import os
import random
import time
import fcntl
from pathlib import Path

# CRITICAL: Remove any paths that might cause importing the wrong shared_brain
workspace_root = '/root/.openclaw/workspace'
if workspace_root in sys.path:
    sys.path.remove(workspace_root)

# Add Atlas paths FIRST (highest priority)
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

# Import the CORRECT shared_brain
from shared_brain import get_shared_brain, save_shared_brain, reset_shared_brain

# Constants
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
PID_FILE = ATLAS_DIR / 'teacher_state' / 'continuous_teacher.pid'
LOCK_FILE = ATLAS_DIR / 'teacher_state' / 'shared_brain.lock'
LOG_FILE = ATLAS_DIR / 'logs' / 'continuous_teacher.log'

# Teaching topics
TOPICS = {
    'math': [
        "Fibonacci Sequence", "Prime Numbers", "Geometry", "Algebra",
        "Calculus", "Statistics", "Probability", "Linear Algebra",
        "Number Theory", "The Golden Ratio"
    ],
    'science': [
        "Photosynthesis", "Newton's Laws", "Quantum Mechanics", "DNA",
        "Evolution", "The Water Cycle", "Chemical Bonds", "Thermodynamics",
        "Cell Biology", "Astronomy"
    ],
    'programming': [
        "Variables and Data Types", "Functions", "Recursion", "Object-Oriented Design",
        "Algorithms", "Data Structures", "Big O Notation", "Concurrency",
        "Functional Programming", "Debugging"
    ],
    'language': [
        "Etymology", "Grammar", "Syntax", "Semantics",
        "Rhetoric", "Pragmatics", "Linguistics", "Phonetics",
        "Morphology", "Discourse Analysis"
    ],
    'logic': [
        "Deductive Reasoning", "Inductive Reasoning", "Logical Fallacies",
        "Boolean Algebra", "Set Theory", "Modal Logic", "Argumentation",
        "Critical Thinking", "Problem Solving", "Decision Theory"
    ]
}

LESSON_TEMPLATES = {
    'math': """
{topic}:

{topic} is a fundamental concept in mathematics. Let me explain how it works.

Key principles:
1. First principle of {topic}
2. Second principle of {topic}
3. Applications in real world

Example: When we study {topic}, we discover patterns that appear throughout nature.
The mathematical relationships help us understand the underlying structure.

Practice problem: Apply {topic} to solve x + 5 = 12.
Solution: Using algebraic manipulation, x = 7.
""",
    'science': """
{topic}:

{topic} describes how the natural world operates. This process is essential to understanding.

Key concepts:
1. Mechanism of {topic}
2. Components involved
3. Outcomes and effects

Example: In {topic}, we observe cause and effect relationships.
The scientific method helps us verify these observations through experimentation.

Real world application: Technology uses {topic} principles every day.
""",
    'programming': """
{topic}:

{topic} is essential for writing effective code. Understanding this concept makes you a better programmer.

Key aspects:
1. Definition and syntax
2. Best practices
3. Common pitfalls

Example code:
```python
def example():
    # Demonstrating {topic}
    result = apply_concept()
    return result
```

When to use: Apply {topic} when you need clean, maintainable code.
""",
    'language': """
{topic}:

{topic} explores how we use and understand language. This field reveals the structure of communication.

Key elements:
1. Core definition
2. Historical development
3. Modern applications

Example: Analyzing {topic} shows how meaning emerges from structure.
Language evolves through usage patterns across communities.

Connection: {topic} relates to psychology, sociology, and cognition.
""",
    'logic': """
{topic}:

{topic} provides tools for clear thinking and valid argumentation. These skills apply everywhere.

Key principles:
1. Basic structure
2. Valid forms
3. Common errors to avoid

Example argument:
Premise 1: All humans are mortal
Premise 2: Socrates is human
Conclusion: Socrates is mortal

This demonstrates {topic} in action.
"""
}

class FileLock:
    """Simple file-based lock for cross-process synchronization"""
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.fd = None
    
    def acquire(self):
        """Acquire the lock"""
        self.fd = open(self.lock_file, 'w')
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        self.fd.write(str(os.getpid()))
        self.fd.flush()
        return True
    
    def release(self):
        """Release the lock"""
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        self.release()


def check_single_instance():
    """Check if another instance is already running"""
    try:
        if PID_FILE.exists():
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if process is still running
            try:
                os.kill(old_pid, 0)
                # If we get here, process exists
                print(f"[ERROR] Another instance is already running (PID: {old_pid})")
                return False
            except OSError:
                # Process doesn't exist, stale PID file
                pass
        
        # Write our PID
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to check PID file: {e}")
        return False


def cleanup_pid():
    """Remove PID file on exit"""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except:
        pass


def log_message(msg):
    """Log to both console and file"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    
    # Append to log file
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(full_msg + '\n')
    except:
        pass


def get_brain_stats_direct():
    """Get brain stats directly from file (bypassing singleton)"""
    import pickle
    brain_path = ATLAS_DIR / 'shared_brain.pkl'
    if brain_path.exists():
        try:
            with open(brain_path, 'rb') as f:
                state = pickle.load(f)
            return {
                'vocab_size': len(state.get('token_to_idx', {})),
                'total_tokens': state.get('total_tokens', 0),
                'next_idx': state.get('next_idx', 0)
            }
        except:
            pass
    return {'vocab_size': 0, 'total_tokens': 0, 'next_idx': 0}


def teach_lesson(topic, category, brain):
    """Teach a single lesson and update shared brain"""
    # Generate lesson content
    template = LESSON_TEMPLATES.get(category, LESSON_TEMPLATES['math'])
    content = template.format(topic=topic)
    
    # Learn from the content
    result = brain.learn_from_text(content)
    
    # Generate a question
    question = f"What is the key concept of {topic}?"
    
    # Get Atlas's response
    context_text = topic.lower()
    response = brain.generate_text(context_text, max_length=30)
    
    # Provide feedback
    feedback = f"Good! Your response shows understanding of {topic}."
    brain.learn_from_text(feedback)
    
    return result


def run_teaching_session():
    """Run one complete teaching session"""
    log_message("=" * 60)
    log_message("🎓 Atlas Continuous Teacher - Starting Session")
    log_message("=" * 60)
    
    # Get file-based stats first (source of truth)
    file_stats = get_brain_stats_direct()
    log_message(f"[FILE] Brain file: {file_stats['vocab_size']} vocab, {file_stats['total_tokens']} tokens")
    
    # Reset singleton to force reload from file
    reset_shared_brain()
    
    # Now get the brain (will load fresh from file)
    brain = get_shared_brain()
    
    # Verify we're using the correct shared_brain module
    import shared_brain
    log_message(f"[DEBUG] Using shared_brain from: {shared_brain.__file__}")
    
    # Get initial stats from loaded brain
    initial_stats = brain.get_stats()
    initial_vocab = initial_stats['vocabulary_size']
    log_message(f"[BRAIN] Loaded vocabulary: {initial_vocab} words")
    log_message(f"[BRAIN] Total tokens: {initial_stats['total_tokens_seen']:,}")
    
    # Verify consistency
    if initial_vocab != file_stats['vocab_size']:
        log_message(f"[WARNING] Brain vocab ({initial_vocab}) != File vocab ({file_stats['vocab_size']})")
    
    # Teach lessons
    lessons_taught = 0
    vocab_growth = []
    
    categories = list(TOPICS.keys())
    random.shuffle(categories)
    
    for i in range(10):
        category = categories[i % len(categories)]
        topic = random.choice(TOPICS[category])
        
        log_message(f"[{i+1}/10] Teaching: {topic} ({category})")
        
        # Acquire lock before teaching and saving
        with FileLock(LOCK_FILE):
            result = teach_lesson(topic, category, brain)
            current_vocab = result['vocabulary_size']
            vocab_growth.append(current_vocab)
            lessons_taught += 1
            
            log_message(f"  -> Tokens processed: {result['tokens_processed']}, Vocab: {current_vocab}")
            
            # Save checkpoint every 5 lessons
            if lessons_taught % 5 == 0:
                save_shared_brain()
                log_message(f"💾 Checkpoint saved after {lessons_taught} lessons")
    
    # Final save with lock
    with FileLock(LOCK_FILE):
        save_shared_brain()
    
    # Report results
    final_stats = brain.get_stats()
    final_vocab = final_stats['vocabulary_size']
    vocab_added = final_vocab - initial_vocab
    
    log_message("=" * 60)
    log_message("📊 Session Complete")
    log_message(f"Lessons taught: {lessons_taught}")
    log_message(f"Vocabulary: {initial_vocab} → {final_vocab} (+{vocab_added})")
    log_message(f"Total tokens: {final_stats['total_tokens_seen']:,}")
    log_message("=" * 60)
    
    return vocab_added > 0


def main():
    # Check for single instance
    if not check_single_instance():
        sys.exit(1)
    
    try:
        success = run_teaching_session()
        return 0 if success else 1
    except Exception as e:
        log_message(f"[ERROR] {e}")
        import traceback
        log_message(traceback.format_exc())
        return 1
    finally:
        cleanup_pid()


if __name__ == "__main__":
    sys.exit(main())
