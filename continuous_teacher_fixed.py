#!/usr/bin/env python3
"""
Atlas Continuous Teacher - Fixed Version

Uses the ACTUAL shared brain singleton to ensure all learning
is persisted to the same brain instance.
"""

import sys
import random
import time
from pathlib import Path

# CRITICAL: Remove any paths that might cause importing the wrong shared_brain
# The workspace root has a different shared_brain.py that uses JSON (WRONG)
# We need the Atlas shared_brain.py that uses pickle and shared_brain.pkl
workspace_root = '/root/.openclaw/workspace'
if workspace_root in sys.path:
    sys.path.remove(workspace_root)

# Add Atlas paths FIRST (highest priority) - self_organizing_av_system must come before Atlas
# because shared_brain imports core.text_learning which is in self_organizing_av_system
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

# Import the CORRECT shared_brain (the one that uses pickle, not JSON)
from shared_brain import get_shared_brain, save_shared_brain

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

def teach_lesson(topic, category):
    """Teach a single lesson and update shared brain"""
    brain = get_shared_brain()
    
    # Generate lesson content
    template = LESSON_TEMPLATES.get(category, LESSON_TEMPLATES['math'])
    content = template.format(topic=topic)
    
    # Learn from the content
    brain.learn_from_text(content)
    
    # Generate a question
    question = f"What is the key concept of {topic}?"
    
    # Get Atlas's response
    context_text = topic.lower()
    response = brain.generate_text(context_text, max_length=30)
    
    # Provide feedback
    feedback = f"Good! Your response shows understanding of {topic}."
    brain.learn_from_text(feedback)
    
    stats = brain.get_stats()
    print(f"Taught: {topic} | Vocab: {stats['vocabulary_size']} words")
    
    return stats['vocabulary_size']

def main():
    print("🎓 Atlas Continuous Teacher")
    print("=" * 50)
    
    # Verify we're using the correct shared_brain module
    import shared_brain
    print(f"[DEBUG] Using shared_brain from: {shared_brain.__file__}")
    
    # Get initial stats
    brain = get_shared_brain()
    initial_stats = brain.get_stats()
    print(f"Starting vocabulary: {initial_stats['vocabulary_size']} words")
    print(f"Total tokens: {initial_stats['total_tokens_seen']:,}")
    
    # Teach 10 lessons
    lessons_taught = 0
    vocab_growth = []
    
    categories = list(TOPICS.keys())
    random.shuffle(categories)
    
    for i in range(10):
        category = categories[i % len(categories)]
        topic = random.choice(TOPICS[category])
        
        print(f"\n[{i+1}/10] Teaching: {topic} ({category})")
        vocab = teach_lesson(topic, category)
        vocab_growth.append(vocab)
        lessons_taught += 1
        
        # Save checkpoint every 5 lessons
        if lessons_taught % 5 == 0:
            save_shared_brain()
            print(f"💾 Checkpoint saved after {lessons_taught} lessons")
    
    # Final save
    save_shared_brain()
    
    # Report results
    final_stats = brain.get_stats()
    print("\n" + "=" * 50)
    print("📊 Session Complete")
    print(f"Lessons taught: {lessons_taught}")
    print(f"Vocabulary: {initial_stats['vocabulary_size']} → {final_stats['vocabulary_size']} (+{final_stats['vocabulary_size'] - initial_stats['vocabulary_size']})")
    print(f"Total tokens: {final_stats['total_tokens_seen']:,}")
    print("=" * 50)

if __name__ == "__main__":
    main()
