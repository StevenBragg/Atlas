#!/usr/bin/env python3
"""
Atlas Adaptive Teacher - Shu-Ha-Ri Methodology

Implements Japanese mastery learning:
- SHU (守): Obey - exact imitation
- HA (破): Break - understand principles  
- RI (離): Transcend - innovate and teach
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from shared_brain import get_shared_brain, save_shared_brain

class TeachingState:
    SHU = "shu"  # Foundation - exact imitation
    HA = "ha"    # Discovery - principle application
    RI = "ri"    # Mastery - innovation and teaching

class ConceptMastery:
    """Tracks mastery of a single concept"""
    def __init__(self, concept):
        self.concept = concept
        self.state = TeachingState.SHU
        self.shu_passed = False
        self.ha_passed = False
        self.ri_passed = False
        self.attempts = 0
        self.last_evaluated = time.time()
        self.misconceptions = []
        
    def to_dict(self):
        return {
            'concept': self.concept,
            'state': self.state,
            'shu_passed': self.shu_passed,
            'ha_passed': self.ha_passed,
            'ri_passed': self.ri_passed,
            'attempts': self.attempts,
            'last_evaluated': self.last_evaluated,
            'misconceptions': self.misconceptions
        }
    
    @classmethod
    def from_dict(cls, data):
        cm = cls(data['concept'])
        cm.state = data['state']
        cm.shu_passed = data['shu_passed']
        cm.ha_passed = data['ha_passed']
        cm.ri_passed = data['ri_passed']
        cm.attempts = data['attempts']
        cm.last_evaluated = data['last_evaluated']
        cm.misconceptions = data['misconceptions']
        return cm

class AdaptiveTeacher:
    """Shu-Ha-Ri adaptive teacher for Atlas"""
    
    def __init__(self):
        self.brain = get_shared_brain()
        self.concepts = {}  # concept -> ConceptMastery
        self.current_topic = None
        self.session_log = []
        self.load_mastery_data()
        
    def load_mastery_data(self):
        """Load mastery tracking from disk"""
        mastery_file = Path('/root/.openclaw/workspace/Atlas/teacher_state/concept_mastery.json')
        if mastery_file.exists():
            try:
                with open(mastery_file) as f:
                    data = json.load(f)
                for concept_data in data.get('concepts', []):
                    cm = ConceptMastery.from_dict(concept_data)
                    self.concepts[cm.concept] = cm
                print(f"[Teacher] Loaded {len(self.concepts)} concepts")
            except Exception as e:
                print(f"[Teacher] Could not load mastery data: {e}")
    
    def save_mastery_data(self):
        """Save mastery tracking to disk"""
        mastery_file = Path('/root/.openclaw/workspace/Atlas/teacher_state/concept_mastery.json')
        mastery_file.parent.mkdir(exist_ok=True)
        data = {
            'concepts': [cm.to_dict() for cm in self.concepts.values()],
            'timestamp': time.time()
        }
        with open(mastery_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_or_create_concept(self, concept_name):
        """Get existing concept or create new"""
        if concept_name not in self.concepts:
            self.concepts[concept_name] = ConceptMastery(concept_name)
        return self.concepts[concept_name]
    
    def evaluate_response_shu(self, response, expected_keywords):
        """
        SHU evaluation: Did Atlas imitate correctly?
        Binary pass/fail based on keyword presence
        """
        response_lower = response.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        match_rate = matches / len(expected_keywords) if expected_keywords else 0
        
        # Binary: pass if 70%+ keywords present
        passed = match_rate >= 0.7
        
        return {
            'passed': passed,
            'match_rate': match_rate,
            'matches': matches,
            'total_keywords': len(expected_keywords),
            'feedback': "Correct! You demonstrated understanding." if passed else 
                       f"Not quite. Expected concepts: {', '.join(expected_keywords)}"
        }
    
    def evaluate_response_ha(self, response, concept, principle):
        """
        HA evaluation: Did Atlas apply the principle?
        Check for application to new context
        """
        response_lower = response.lower()
        
        # Check if principle is mentioned
        principle_present = principle.lower() in response_lower
        
        # Check for application words (not just definition)
        application_words = ['apply', 'use', 'example', 'if', 'when', 'because', 'therefore']
        has_application = any(word in response_lower for word in application_words)
        
        # Check for novel context (not just repeating lesson)
        lesson_words = concept.lower().split()
        novel_content = len([w for w in response_lower.split() if w not in lesson_words]) > 5
        
        passed = principle_present and has_application and novel_content
        
        return {
            'passed': passed,
            'principle_present': principle_present,
            'has_application': has_application,
            'novel_content': novel_content,
            'feedback': "Excellent application of principles!" if passed else
                       "Can you explain how this principle works in a different situation?"
        }
    
    def evaluate_response_ri(self, response, concept):
        """
        RI evaluation: Did Atlas innovate or teach?
        Check for novel insights or teaching ability
        """
        response_lower = response.lower()
        
        # Check for teaching indicators
        teaching_words = ['imagine', 'think of', 'like', 'similar to', 'means that', 'in other words']
        is_teaching = any(phrase in response_lower for phrase in teaching_words)
        
        # Check for synthesis (combining with other concepts)
        other_concepts = [c for c in self.concepts.keys() if c != concept and c.lower() in response_lower]
        has_synthesis = len(other_concepts) > 0
        
        # Check for novelty (not in training data)
        # Simplified: check length and structure
        has_structure = len(response.split()) > 10 and '.' in response
        
        passed = is_teaching or (has_synthesis and has_structure)
        
        return {
            'passed': passed,
            'is_teaching': is_teaching,
            'synthesis_concepts': other_concepts,
            'has_structure': has_structure,
            'feedback': "Outstanding! You've truly mastered this concept." if passed else
                       "Can you explain this concept to someone who's never heard of it?"
        }
    
    def teach_shu(self, concept, category):
        """SHU phase: Foundation through imitation"""
        print(f"\n[SHU] Teaching: {concept}")
        
        # Generate foundation lesson
        lesson = self.generate_shu_lesson(concept, category)
        print(f"  Lesson: {lesson[:100]}...")
        
        # Present to Atlas
        self.brain.learn_from_text(lesson)
        
        # Ask recall question
        question = f"What is {concept}? Explain in your own words."
        print(f"  Q: {question}")
        
        # Get Atlas response
        response = self.brain.generate_text(concept.lower(), max_length=40)
        print(f"  A: {response}")
        
        # Evaluate
        expected_keywords = self.extract_keywords(concept, category)
        evaluation = self.evaluate_response_shu(response, expected_keywords)
        print(f"  Evaluation: {evaluation['match_rate']:.0%} match - {'PASS' if evaluation['passed'] else 'RETRY'}")
        
        # Learn from feedback
        feedback = evaluation['feedback']
        self.brain.learn_from_text(feedback)
        
        return evaluation['passed'], feedback
    
    def teach_ha(self, concept, category):
        """HA phase: Discovery through application"""
        print(f"\n[HA] Teaching: {concept}")
        
        # Present variation challenge
        challenge = self.generate_ha_challenge(concept, category)
        print(f"  Challenge: {challenge}")
        
        # Get Atlas response
        response = self.brain.generate_text(challenge.lower(), max_length=50)
        print(f"  A: {response}")
        
        # Evaluate application
        principle = self.get_principle(concept, category)
        evaluation = self.evaluate_response_ha(response, concept, principle)
        print(f"  Evaluation: {'PASS' if evaluation['passed'] else 'RETRY'}")
        
        # Learn from feedback
        feedback = evaluation['feedback']
        self.brain.learn_from_text(feedback)
        
        return evaluation['passed'], feedback
    
    def teach_ri(self, concept, category):
        """RI phase: Mastery through teaching"""
        print(f"\n[RI] Teaching: {concept}")
        
        # Ask Atlas to teach the concept
        prompt = f"Explain {concept} to a beginner who knows nothing about it."
        print(f"  Prompt: {prompt}")
        
        # Get Atlas teaching
        response = self.brain.generate_text(f"teach {concept}", max_length=60)
        print(f"  Atlas teaches: {response}")
        
        # Evaluate teaching quality
        evaluation = self.evaluate_response_ri(response, concept)
        print(f"  Evaluation: {'PASS - MASTERY ACHIEVED' if evaluation['passed'] else 'NEEDS MORE WORK'}")
        
        if evaluation['passed']:
            # Atlas has mastered this concept!
            feedback = "Excellent teaching! You've achieved mastery of this concept."
        else:
            feedback = evaluation['feedback']
        
        self.brain.learn_from_text(feedback)
        
        return evaluation['passed'], feedback
    
    def teach_concept(self, concept, category):
        """Teach a concept adapting to current mastery level"""
        cm = self.get_or_create_concept(concept)
        cm.attempts += 1
        
        print(f"\n{'='*60}")
        print(f"Concept: {concept} | Current State: {cm.state.upper()} | Attempt: {cm.attempts}")
        print(f"{'='*60}")
        
        # Check for regression (if not reviewed in 24 hours, might forget)
        hours_since_review = (time.time() - cm.last_evaluated) / 3600
        if hours_since_review > 24 and cm.state != TeachingState.SHU:
            print(f"[REGRESSION] {hours_since_review:.1f} hours since review - dropping back to check")
            # Quick check if still remembered
            check_passed, _ = self.teach_shu(concept, category)
            if not check_passed:
                cm.state = TeachingState.SHU
                cm.shu_passed = False
                print(f"[REGRESSION] Dropped back to SHU")
        
        # Teach based on current state
        if cm.state == TeachingState.SHU:
            passed, feedback = self.teach_shu(concept, category)
            if passed:
                cm.shu_passed = True
                cm.state = TeachingState.HA
                print(f"[ADVANCE] {concept} -> HA phase!")
            else:
                print(f"[RETRY] Staying in SHU phase")
                
        elif cm.state == TeachingState.HA:
            passed, feedback = self.teach_ha(concept, category)
            if passed:
                cm.ha_passed = True
                cm.state = TeachingState.RI
                print(f"[ADVANCE] {concept} -> RI phase!")
            elif cm.attempts > 5 and not passed:
                # Too many failures, drop back to SHU
                cm.state = TeachingState.SHU
                print(f"[REGRESSION] {concept} -> Back to SHU")
            else:
                print(f"[RETRY] Staying in HA phase")
                
        elif cm.state == TeachingState.RI:
            passed, feedback = self.teach_ri(concept, category)
            if passed:
                cm.ri_passed = True
                print(f"[MASTERY] {concept} FULLY MASTERED!")
            else:
                print(f"[RETRY] Working toward mastery")
        
        cm.last_evaluated = time.time()
        self.save_mastery_data()
        save_shared_brain()
        
        return cm
    
    def generate_shu_lesson(self, concept, category):
        """Generate foundation lesson for SHU phase"""
        templates = {
            'math': f"{concept} is a fundamental mathematical concept. It involves specific rules and procedures that must be followed exactly.",
            'science': f"{concept} describes how nature works. Scientists have discovered these principles through careful observation and experimentation.",
            'programming': f"{concept} is a programming technique. It provides a specific way to solve problems in code.",
            'language': f"{concept} is a rule of language. It governs how we construct meaningful sentences and communicate ideas.",
            'logic': f"{concept} is a principle of reasoning. It helps us determine what follows from what we know."
        }
        return templates.get(category, f"{concept} is an important concept to understand.")
    
    def generate_ha_challenge(self, concept, category):
        """Generate application challenge for HA phase"""
        challenges = {
            'math': f"How would you use {concept} to solve a real-world problem?",
            'science': f"Can you think of an example where {concept} applies in everyday life?",
            'programming': f"When would you choose to use {concept} in a program?",
            'language': f"How does {concept} change meaning in different contexts?",
            'logic': f"How would you apply {concept} to evaluate an argument?"
        }
        return challenges.get(category, f"How would you apply {concept} in a new situation?")
    
    def extract_keywords(self, concept, category):
        """Extract expected keywords for SHU evaluation"""
        # Simple keyword extraction
        base_keywords = concept.lower().split()
        category_keywords = {
            'math': ['calculate', 'solve', 'equation', 'number', 'formula'],
            'science': ['observe', 'experiment', 'evidence', 'theory', 'natural'],
            'programming': ['code', 'function', 'variable', 'algorithm', 'computer'],
            'language': ['word', 'sentence', 'meaning', 'grammar', 'communication'],
            'logic': ['reason', 'argument', 'valid', 'conclusion', 'premise']
        }
        return base_keywords + category_keywords.get(category, [])
    
    def get_principle(self, concept, category):
        """Get core principle for HA evaluation"""
        principles = {
            'math': 'calculation',
            'science': 'observation',
            'programming': 'automation',
            'language': 'communication',
            'logic': 'reasoning'
        }
        return principles.get(category, 'concept')
    
    def get_stats(self):
        """Get teaching statistics"""
        total = len(self.concepts)
        shu_count = sum(1 for c in self.concepts.values() if c.state == TeachingState.SHU)
        ha_count = sum(1 for c in self.concepts.values() if c.state == TeachingState.HA)
        ri_count = sum(1 for c in self.concepts.values() if c.state == TeachingState.RI)
        mastered = sum(1 for c in self.concepts.values() if c.ri_passed)
        
        return {
            'total_concepts': total,
            'shu_phase': shu_count,
            'ha_phase': ha_count,
            'ri_phase': ri_count,
            'mastered': mastered,
            'total_attempts': sum(c.attempts for c in self.concepts.values())
        }

# Topics for teaching
TOPICS = {
    'math': ['Algebra', 'Geometry', 'Calculus', 'Statistics', 'Number Theory'],
    'science': ['Photosynthesis', 'Newton\'s Laws', 'Quantum Mechanics', 'Evolution', 'Cell Biology'],
    'programming': ['Recursion', 'Object-Oriented Design', 'Functional Programming', 'Algorithms', 'Data Structures'],
    'language': ['Grammar', 'Syntax', 'Semantics', 'Etymology', 'Rhetoric'],
    'logic': ['Deductive Reasoning', 'Inductive Reasoning', 'Logical Fallacies', 'Boolean Algebra', 'Set Theory']
}

def main():
    print("🎓 Atlas Adaptive Teacher (Shu-Ha-Ri)")
    print("=" * 60)
    
    teacher = AdaptiveTeacher()
    
    # Show current stats
    stats = teacher.get_stats()
    print(f"\nCurrent Progress:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  SHU phase: {stats['shu_phase']}")
    print(f"  HA phase: {stats['ha_phase']}")
    print(f"  RI phase: {stats['ri_phase']}")
    print(f"  Mastered: {stats['mastered']}")
    
    # Teach 5 lessons
    import random
    all_topics = [(t, cat) for cat, topics in TOPICS.items() for t in topics]
    random.shuffle(all_topics)
    
    for i, (topic, category) in enumerate(all_topics[:5], 1):
        print(f"\n{'='*60}")
        print(f"Lesson {i}/5")
        teacher.teach_concept(topic, category)
        time.sleep(1)
    
    # Final stats
    print(f"\n{'='*60}")
    print("Session Complete!")
    stats = teacher.get_stats()
    print(f"  Concepts taught: {stats['total_concepts']}")
    print(f"  Mastered: {stats['mastered']}")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
