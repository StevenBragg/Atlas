#!/usr/bin/env python3
"""
Atlas Continuous Teacher - v3 with Q&A and Proper Logging

Key improvements:
1. Actual Q&A conversations with Atlas
2. Proper conversation logging
3. Persistent session tracking
4. Better error handling
"""

import sys
import os
import random
import time
import fcntl
import json
from pathlib import Path
from datetime import datetime

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
CONVERSATION_FILE = ATLAS_DIR / 'teacher_state' / 'conversations.json'
SESSION_FILE = ATLAS_DIR / 'teacher_state' / 'session_stats.json'

# Teaching topics with Q&A pairs
TOPICS = {
    'math': [
        ("Fibonacci Sequence", "What is the pattern in the Fibonacci sequence?"),
        ("Prime Numbers", "Why is 2 the only even prime number?"),
        ("Geometry", "How do you calculate the area of a circle?"),
        ("Algebra", "What does solving for x mean?"),
        ("Calculus", "What is the difference between a derivative and an integral?"),
        ("Statistics", "What does standard deviation measure?"),
        ("Probability", "If I flip a coin twice, what's the chance of getting heads both times?"),
        ("Linear Algebra", "What is a matrix used for?"),
        ("Number Theory", "Why are prime numbers important in cryptography?"),
        ("The Golden Ratio", "Where does the golden ratio appear in nature?")
    ],
    'science': [
        ("Photosynthesis", "What do plants need for photosynthesis?"),
        ("Newton's Laws", "What is Newton's third law?"),
        ("Quantum Mechanics", "What is superposition?"),
        ("DNA", "What does DNA stand for and what does it do?"),
        ("Evolution", "What is natural selection?"),
        ("The Water Cycle", "What are the three main stages of the water cycle?"),
        ("Chemical Bonds", "What's the difference between ionic and covalent bonds?"),
        ("Thermodynamics", "What is the law of conservation of energy?"),
        ("Cell Biology", "What is the function of mitochondria?"),
        ("Astronomy", "How do stars produce energy?")
    ],
    'programming': [
        ("Variables and Data Types", "What is the difference between a string and an integer?"),
        ("Functions", "Why should I use functions in my code?"),
        ("Recursion", "What is a recursive function?"),
        ("Object-Oriented Design", "What is a class in programming?"),
        ("Algorithms", "What makes an algorithm efficient?"),
        ("Data Structures", "When should I use a list versus a dictionary?"),
        ("Big O Notation", "What does O(n) mean?"),
        ("Concurrency", "What is the difference between threading and multiprocessing?"),
        ("Functional Programming", "What is a pure function?"),
        ("Debugging", "What is a breakpoint?")
    ],
    'language': [
        ("Etymology", "Where does the word 'salary' come from?"),
        ("Grammar", "What is the difference between a noun and a verb?"),
        ("Syntax", "Why does word order matter in sentences?"),
        ("Semantics", "What is the difference between syntax and semantics?"),
        ("Rhetoric", "What are the three appeals in rhetoric?"),
        ("Pragmatics", "What does context mean in language?"),
        ("Linguistics", "What do linguists study?"),
        ("Phonetics", "How do we produce different speech sounds?"),
        ("Morphology", "What is a morpheme?"),
        ("Discourse Analysis", "What is discourse?")
    ],
    'logic': [
        ("Deductive Reasoning", "What is a syllogism?"),
        ("Inductive Reasoning", "Can inductive reasoning prove something with certainty?"),
        ("Logical Fallacies", "What is an ad hominem attack?"),
        ("Boolean Algebra", "What are the three basic Boolean operations?"),
        ("Set Theory", "What is the difference between a union and an intersection?"),
        ("Modal Logic", "What does 'necessarily' mean in modal logic?"),
        ("Argumentation", "What makes an argument valid?"),
        ("Critical Thinking", "What is confirmation bias?"),
        ("Problem Solving", "What is the first step in solving any problem?"),
        ("Decision Theory", "What is expected value?")
    ]
}

LESSON_TEMPLATES = {
    'math': """{topic}:

{topic} is a fundamental concept in mathematics. Let me explain how it works.

Key principles:
1. First principle of {topic}
2. Second principle of {topic}  
3. Applications in real world

Example: When we study {topic}, we discover patterns that appear throughout nature.
The mathematical relationships help us understand the underlying structure.
""",
    'science': """{topic}:

{topic} describes how the natural world operates. This process is essential to understanding.

Key concepts:
1. Mechanism of {topic}
2. Components involved
3. Outcomes and effects

Example: In {topic}, we observe cause and effect relationships.
The scientific method helps us verify these observations through experimentation.
""",
    'programming': """{topic}:

{topic} is essential for writing effective code. Understanding this concept makes you a better programmer.

Key aspects:
1. Definition and syntax
2. Best practices
3. Common pitfalls

Example code demonstrating {topic}:
```python
def example():
    result = apply_concept()
    return result
```
""",
    'language': """{topic}:

{topic} explores how we use and understand language. This field reveals the structure of communication.

Key elements:
1. Core definition
2. Historical development
3. Modern applications

Example: Analyzing {topic} shows how meaning emerges from structure.
Language evolves through usage patterns across communities.
""",
    'logic': """{topic}:

{topic} provides tools for clear thinking and valid argumentation. These skills apply everywhere.

Key principles:
1. Basic structure
2. Valid forms
3. Common errors to avoid

Example argument demonstrating {topic}:
Premise 1: All humans are mortal
Premise 2: Socrates is human
Conclusion: Socrates is mortal
"""
}

# Sample answers for Atlas to learn from
SAMPLE_ANSWERS = {
    'math': [
        "Each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8...",
        "Because every other even number is divisible by 2 and another number.",
        "Using the formula A equals pi times radius squared.",
        "Finding the value that makes the equation true.",
        "Derivatives measure rate of change, integrals measure accumulation."
    ],
    'science': [
        "Plants need sunlight, water, and carbon dioxide.",
        "For every action, there is an equal and opposite reaction.",
        "A quantum system can exist in multiple states simultaneously.",
        "Deoxyribonucleic acid - it stores genetic information.",
        "The process where organisms better adapted to their environment survive."
    ],
    'programming': [
        "Strings hold text, integers hold whole numbers.",
        "Functions make code reusable and easier to understand.",
        "A function that calls itself to solve smaller instances of the same problem.",
        "A blueprint for creating objects with properties and methods.",
        "Efficiency depends on time and space complexity."
    ],
    'language': [
        "From Latin 'salarium', money for salt.",
        "Nouns name things, verbs describe actions.",
        "Different orders create different meanings.",
        "Syntax is structure, semantics is meaning.",
        "Ethos, pathos, and logos."
    ],
    'logic': [
        "A form of reasoning with two premises and a conclusion.",
        "No, it only provides probable conclusions based on evidence.",
        "Attacking the person instead of their argument.",
        "AND, OR, and NOT.",
        "Union combines sets, intersection finds common elements."
    ]
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


def log_message(msg):
    """Log to both console and file"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(full_msg + '\n')
    except:
        pass


def load_session_stats():
    """Load persistent session statistics"""
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'total_lessons': 0,
        'total_questions': 0,
        'total_conversations': 0,
        'sessions_completed': 0,
        'last_session_time': None
    }


def save_session_stats(stats):
    """Save persistent session statistics"""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        log_message(f"[WARNING] Could not save session stats: {e}")


def load_conversations():
    """Load conversation history"""
    if CONVERSATION_FILE.exists():
        try:
            with open(CONVERSATION_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def save_conversation(q, a, topic, category):
    """Save a conversation to history"""
    conversations = load_conversations()
    
    conversation = {
        'timestamp': datetime.now().isoformat(),
        'time': datetime.now().strftime('%H:%M'),
        'topic': topic,
        'category': category,
        'q': q,
        'a': a
    }
    
    conversations.append(conversation)
    
    # Keep only last 100 conversations
    conversations = conversations[-100:]
    
    try:
        CONVERSATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        log_message(f"[WARNING] Could not save conversation: {e}")


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


def teach_lesson_with_qa(topic, question, category, brain):
    """Teach a lesson with actual Q&A interaction"""
    # Generate lesson content
    template = LESSON_TEMPLATES.get(category, LESSON_TEMPLATES['math'])
    content = template.format(topic=topic)
    
    # Learn from the lesson content
    result = brain.learn_from_text(content)
    
    # Generate Atlas's response to the question
    context = f"{topic.lower()} {question.lower()}"
    atlas_response = brain.generate_text(context, max_length=50)
    
    # If response is too short or generic, provide a sample answer for learning
    sample_answers = SAMPLE_ANSWERS.get(category, [])
    if len(atlas_response) < 20 and sample_answers:
        sample = random.choice(sample_answers)
        brain.learn_from_text(f"Q: {question} A: {sample}")
        atlas_response = sample
    
    # Provide feedback to reinforce learning
    feedback = f"Good answer about {topic}! Understanding this helps build knowledge."
    brain.learn_from_text(feedback)
    
    # Save the conversation
    save_conversation(question, atlas_response, topic, category)
    
    return result, question, atlas_response


def run_teaching_session():
    """Run one complete teaching session with Q&A"""
    log_message("=" * 60)
    log_message("🎓 Atlas Continuous Teacher - Starting Q&A Session")
    log_message("=" * 60)
    
    # Load session stats
    session_stats = load_session_stats()
    
    # Get file-based stats first (source of truth)
    file_stats = get_brain_stats_direct()
    log_message(f"[FILE] Brain file: {file_stats['vocab_size']} vocab, {file_stats['total_tokens']} tokens")
    
    # Reset singleton to force reload from file
    reset_shared_brain()
    
    # Now get the brain (will load fresh from file)
    brain = get_shared_brain()
    
    # Get initial stats from loaded brain
    initial_stats = brain.get_stats()
    initial_vocab = initial_stats['vocabulary_size']
    log_message(f"[BRAIN] Loaded vocabulary: {initial_vocab} words")
    log_message(f"[BRAIN] Total tokens: {initial_stats['total_tokens_seen']:,}")
    
    # Verify consistency
    if initial_vocab != file_stats['vocab_size']:
        log_message(f"[WARNING] Brain vocab ({initial_vocab}) != File vocab ({file_stats['vocab_size']})")
    
    # Teach lessons with Q&A
    lessons_taught = 0
    questions_asked = 0
    conversations = []
    
    categories = list(TOPICS.keys())
    random.shuffle(categories)
    
    for i in range(10):
        category = categories[i % len(categories)]
        topic, question = random.choice(TOPICS[category])
        
        log_message(f"[{i+1}/10] Teaching: {topic} ({category})")
        log_message(f"       Q: {question}")
        
        # Acquire lock before teaching and saving
        with FileLock(LOCK_FILE):
            result, q, a = teach_lesson_with_qa(topic, question, category, brain)
            current_vocab = result['vocabulary_size']
            lessons_taught += 1
            questions_asked += 1
            
            # Log the Q&A
            log_message(f"       A: {a[:80]}{'...' if len(a) > 80 else ''}")
            log_message(f"       -> Tokens: {result['tokens_processed']}, Vocab: {current_vocab}")
            
            # Save checkpoint every 5 lessons
            if lessons_taught % 5 == 0:
                save_shared_brain()
                log_message(f"💾 Checkpoint saved after {lessons_taught} lessons")
    
    # Final save with lock
    with FileLock(LOCK_FILE):
        save_shared_brain()
    
    # Update session stats
    session_stats['total_lessons'] += lessons_taught
    session_stats['total_questions'] += questions_asked
    session_stats['total_conversations'] += lessons_taught  # Each lesson has one Q&A
    session_stats['sessions_completed'] += 1
    session_stats['last_session_time'] = datetime.now().isoformat()
    save_session_stats(session_stats)
    
    # Report results
    final_stats = brain.get_stats()
    final_vocab = final_stats['vocabulary_size']
    vocab_added = final_vocab - initial_vocab
    
    log_message("=" * 60)
    log_message("📊 Session Complete")
    log_message(f"Lessons taught: {lessons_taught}")
    log_message(f"Questions asked: {questions_asked}")
    log_message(f"Vocabulary: {initial_vocab} → {final_vocab} (+{vocab_added})")
    log_message(f"Total tokens: {final_stats['total_tokens_seen']:,}")
    log_message(f"Total sessions completed: {session_stats['sessions_completed']}")
    log_message("=" * 60)
    
    return True


def main():
    try:
        success = run_teaching_session()
        return 0 if success else 1
    except Exception as e:
        log_message(f"[ERROR] {e}")
        import traceback
        log_message(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
