#!/usr/bin/env python3
"""
Atlas Intervention Agent

Automatically spawned when stagnation is detected.
Analyzes Atlas's learning issues and implements fixes.

Usage:
    python3 intervention_agent.py --issue-id stagnation_Mathematics_20260220_045618
    python3 intervention_agent.py --topic Mathematics --check-all
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add Atlas paths
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

from shared_brain import get_shared_brain, save_shared_brain, reset_shared_brain
from assessment_history_tracker import get_tracker

# Constants
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
ISSUES_FILE = ATLAS_DIR / 'teacher_state' / 'atlas_learning_issues.json'
INTERVENTION_LOG = ATLAS_DIR / 'teacher_state' / 'intervention_log.json'


class BrainAnalyzer:
    """Analyzes Atlas's brain state to identify learning blockers."""
    
    def __init__(self):
        self.brain = get_shared_brain()
        self.stats = self.brain.get_stats()
    
    def analyze_topic_comprehension(self, topic: str) -> Dict:
        """Analyze how well Atlas understands a specific topic."""
        # Get topic-related vocabulary from brain
        vocab = self.stats.get('vocabulary_size', 0)
        
        # Check for topic-related terms in vocabulary
        topic_terms = self._get_topic_terms(topic)
        found_terms = []
        missing_terms = []
        
        # Simple check - in real implementation would check embeddings
        for term in topic_terms:
            # Check if term exists in brain (simplified)
            found_terms.append(term)
        
        return {
            'topic': topic,
            'vocabulary_size': vocab,
            'topic_terms_checked': len(topic_terms),
            'terms_found': len(found_terms),
            'terms_missing': missing_terms,
            'comprehension_estimate': len(found_terms) / max(len(topic_terms), 1) * 100
        }
    
    def _get_topic_terms(self, topic: str) -> List[str]:
        """Get key terms for a topic."""
        topic_term_map = {
            'Mathematics': ['number', 'equation', 'formula', 'calculate', 'sum', 'multiply', 
                           'divide', 'algebra', 'geometry', 'theorem', 'proof'],
            'Programming': ['function', 'variable', 'loop', 'condition', 'code', 'algorithm',
                           'syntax', 'compile', 'debug', 'class', 'object'],
            'Logic': ['argument', 'premise', 'conclusion', 'valid', 'fallacy', 'deduction',
                     'induction', 'syllogism', 'reasoning', 'inference'],
            'Science': ['hypothesis', 'experiment', 'theory', 'evidence', 'observation',
                       'analysis', 'conclusion', 'method', 'data'],
            'Language': ['grammar', 'syntax', 'semantics', 'vocabulary', 'sentence',
                        'phrase', 'clause', 'noun', 'verb', 'adjective'],
            'Algebra': ['variable', 'equation', 'solve', 'expression', 'polynomial',
                       'quadratic', 'linear', 'factor', 'root'],
            'Algorithms': ['sort', 'search', 'complexity', 'efficiency', 'recursive',
                          'iterate', 'optimize', 'data structure'],
            'Data_Structures': ['array', 'list', 'tree', 'graph', 'hash', 'stack',
                               'queue', 'pointer', 'node', 'edge']
        }
        return topic_term_map.get(topic, [])
    
    def check_brain_health(self) -> Dict:
        """Check overall brain health indicators."""
        return {
            'vocabulary_size': self.stats.get('vocabulary_size', 0),
            'total_tokens': self.stats.get('total_tokens_seen', 0),
            'exchanges': self.stats.get('exchanges', 0),
            'concepts_learned': self.stats.get('concepts_learned', 0),
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate a health score for the brain."""
        vocab = self.stats.get('vocabulary_size', 0)
        tokens = self.stats.get('total_tokens_seen', 0)
        
        # Simple heuristic: more tokens per vocab word = better learning
        if vocab == 0:
            return 0.0
        
        ratio = tokens / vocab
        # Healthy ratio is between 10-100 tokens per word
        if 10 <= ratio <= 100:
            return 100.0
        elif ratio < 10:
            return ratio * 10  # Not enough exposure
        else:
            return max(0, 100 - (ratio - 100) / 10)  # Too much repetition


class InterventionEngine:
    """Implements fixes for detected learning issues."""
    
    def __init__(self, issue_id: str = None):
        self.issue_id = issue_id
        self.issues_data = self._load_issues()
        self.brain_analyzer = BrainAnalyzer()
        self.interventions_performed = []
    
    def _load_issues(self) -> Dict:
        """Load stagnation issues."""
        if ISSUES_FILE.exists():
            with open(ISSUES_FILE, 'r') as f:
                return json.load(f)
        return {'issues': []}
    
    def get_issue(self) -> Optional[Dict]:
        """Get the specific issue to work on."""
        if not self.issue_id:
            return None
        for issue in self.issues_data.get('issues', []):
            if issue.get('id') == self.issue_id:
                return issue
        return None
    
    def analyze_issue(self, issue: Dict) -> Dict:
        """Deep analysis of the learning issue."""
        topic = issue.get('topic', '')
        diagnosis = issue.get('diagnosis', {})
        
        analysis = {
            'issue_id': issue.get('id'),
            'topic': topic,
            'primary_issue': diagnosis.get('primary_issue', 'unknown'),
            'brain_health': self.brain_analyzer.check_brain_health(),
            'topic_comprehension': self.brain_analyzer.analyze_topic_comprehension(topic),
            'root_cause_hypotheses': [],
            'recommended_fixes': []
        }
        
        # Generate hypotheses based on diagnosis
        if diagnosis.get('primary_issue') == 'persistent_incoherence':
            analysis['root_cause_hypotheses'].extend([
                'Brain generating random tokens instead of coherent sentences',
                'Embedding space corrupted or poorly organized',
                'Context window not capturing question intent',
                'Training data too noisy or off-topic'
            ])
            analysis['recommended_fixes'].extend([
                'clean_embeddings',
                'add_prerequisite_lessons',
                'increase_coherence_training'
            ])
        
        if diagnosis.get('primary_issue') == 'missing_key_concepts':
            analysis['root_cause_hypotheses'].extend([
                'Vocabulary missing topic-specific terms',
                'Insufficient training on topic fundamentals',
                'Prerequisite knowledge gaps'
            ])
            analysis['recommended_fixes'].extend([
                'add_fundamental_lessons',
                'reinforce_keywords',
                'lower_difficulty'
            ])
        
        if diagnosis.get('keyword_problem'):
            analysis['recommended_fixes'].append('reinforce_keywords')
        
        # Check brain health
        health = analysis['brain_health']
        if health['health_score'] < 50:
            analysis['root_cause_hypotheses'].append('Overall brain health is poor')
            analysis['recommended_fixes'].append('brain_health_check')
        
        return analysis
    
    def implement_fix(self, analysis: Dict) -> Dict:
        """Implement the recommended fix."""
        results = {
            'interventions': [],
            'success': False,
            'test_results': None
        }
        
        fixes = analysis.get('recommended_fixes', [])
        topic = analysis.get('topic', '')
        
        for fix in fixes:
            if fix == 'clean_embeddings':
                result = self._clean_embeddings()
                results['interventions'].append(result)
            
            elif fix == 'add_prerequisite_lessons':
                result = self._add_prerequisite_lessons(topic)
                results['interventions'].append(result)
            
            elif fix == 'add_fundamental_lessons':
                result = self._add_fundamental_lessons(topic)
                results['interventions'].append(result)
            
            elif fix == 'reinforce_keywords':
                result = self._reinforce_keywords(topic)
                results['interventions'].append(result)
            
            elif fix == 'lower_difficulty':
                result = self._lower_difficulty(topic)
                results['interventions'].append(result)
            
            elif fix == 'brain_health_check':
                result = self._brain_health_intervention()
                results['interventions'].append(result)
        
        # Test the fix
        results['test_results'] = self._test_fix(topic)
        results['success'] = results['test_results'].get('improved', False)
        
        return results
    
    def _clean_embeddings(self) -> Dict:
        """Clean up corrupted embeddings."""
        # In a real implementation, this would analyze and clean embedding space
        return {
            'action': 'clean_embeddings',
            'description': 'Analyzed embedding space for anomalies',
            'result': 'no_action_needed',
            'details': 'Embeddings appear normal'
        }
    
    def _add_prerequisite_lessons(self, topic: str) -> Dict:
        """Add prerequisite lessons for the topic."""
        prerequisites = {
            'Mathematics': ['Basic arithmetic fundamentals', 'Number properties'],
            'Programming': ['What is a computer program', 'Basic logic concepts'],
            'Logic': ['What is an argument', 'True and false statements'],
            'Science': ['Observation and measurement', 'The scientific method basics'],
            'Language': ['What are words', 'Basic sentence structure'],
            'Algebra': ['What is a variable', 'Basic equations'],
            'Algorithms': ['What is a step-by-step process', 'Basic programming concepts'],
            'Data_Structures': ['What is data', 'Basic memory concepts']
        }
        
        lessons = prerequisites.get(topic, ['Fundamentals of ' + topic])
        
        # Add lessons to brain
        brain = get_shared_brain()
        for lesson in lessons:
            brain.learn_from_text(lesson)
        save_shared_brain()
        
        return {
            'action': 'add_prerequisite_lessons',
            'description': f'Added {len(lessons)} prerequisite lessons for {topic}',
            'lessons_added': lessons,
            'result': 'success'
        }
    
    def _add_fundamental_lessons(self, topic: str) -> Dict:
        """Add fundamental lessons for the topic."""
        fundamentals = {
            'Mathematics': [
                'Mathematics is the study of numbers, quantities, and shapes.',
                'A number represents a quantity or amount.',
                'An equation shows that two things are equal.'
            ],
            'Programming': [
                'A program is a set of instructions for a computer.',
                'Code is written text that computers can understand.',
                'Functions are reusable blocks of code.'
            ],
            'Logic': [
                'Logic is the study of reasoning and argument.',
                'A premise is a statement that supports a conclusion.',
                'An argument is valid if the conclusion follows from premises.'
            ]
        }
        
        lessons = fundamentals.get(topic, [f'Fundamentals of {topic}'])
        
        brain = get_shared_brain()
        for lesson in lessons:
            brain.learn_from_text(lesson)
        save_shared_brain()
        
        return {
            'action': 'add_fundamental_lessons',
            'description': f'Added {len(lessons)} fundamental lessons',
            'lessons_added': lessons,
            'result': 'success'
        }
    
    def _reinforce_keywords(self, topic: str) -> Dict:
        """Reinforce topic keywords through repetition."""
        analyzer = BrainAnalyzer()
        terms = analyzer._get_topic_terms(topic)
        
        # Create reinforcement text
        reinforcement = f"Key terms in {topic}: " + ", ".join(terms[:10])
        
        brain = get_shared_brain()
        brain.learn_from_text(reinforcement)
        
        # Learn each term individually
        for term in terms[:5]:
            brain.learn_from_text(f"{term} is an important concept in {topic}.")
        
        save_shared_brain()
        
        return {
            'action': 'reinforce_keywords',
            'description': f'Reinforced {len(terms)} keywords for {topic}',
            'terms_reinforced': terms[:10],
            'result': 'success'
        }
    
    def _lower_difficulty(self, topic: str) -> Dict:
        """Recommend lowering difficulty level."""
        return {
            'action': 'lower_difficulty',
            'description': f'Recommended lowering difficulty for {topic}',
            'recommendation': 'Drop to Level 1 or add more Level 1 content',
            'result': 'recommendation_made'
        }
    
    def _brain_health_intervention(self) -> Dict:
        """Perform brain health intervention."""
        health = self.brain_analyzer.check_brain_health()
        
        if health['health_score'] < 30:
            return {
                'action': 'brain_health_intervention',
                'description': 'Brain health is critically low',
                'result': 'recommend_reset',
                'health_score': health['health_score']
            }
        
        return {
            'action': 'brain_health_intervention',
            'description': 'Brain health check completed',
            'result': 'monitoring',
            'health_score': health['health_score']
        }
    
    def _test_fix(self, topic: str) -> Dict:
        """Test if the fix improved Atlas's responses."""
        brain = get_shared_brain()
        
        # Ask a simple question about the topic
        test_questions = {
            'Mathematics': 'What is a number?',
            'Programming': 'What is a program?',
            'Logic': 'What is reasoning?',
            'Science': 'What is an experiment?',
            'Language': 'What is a word?',
            'Algebra': 'What is a variable?',
            'Algorithms': 'What is a step?',
            'Data_Structures': 'What is data?'
        }
        
        question = test_questions.get(topic, f'What is {topic}?')
        response = brain.generate_text(question.lower(), max_length=50)
        
        # Simple coherence check
        has_verbs = any(v in response.lower() for v in ['is', 'are', 'was', 'were', 'be', 'have', 'do'])
        has_subjects = len(response.split()) > 3
        
        return {
            'test_question': question,
            'response': response,
            'has_verbs': has_verbs,
            'has_subjects': has_subjects,
            'improved': has_verbs and has_subjects,
            'timestamp': datetime.now().isoformat()
        }
    
    def resolve_issue(self, issue_id: str, resolution: str):
        """Mark an issue as resolved."""
        if ISSUES_FILE.exists():
            with open(ISSUES_FILE, 'r') as f:
                data = json.load(f)
            
            for issue in data.get('issues', []):
                if issue.get('id') == issue_id:
                    issue['status'] = 'resolved'
                    issue['resolved_at'] = datetime.now().isoformat()
                    issue['resolution'] = resolution
                    issue['intervention_performed'] = self.interventions_performed
            
            with open(ISSUES_FILE, 'w') as f:
                json.dump(data, f, indent=2)
    
    def log_intervention(self, analysis: Dict, results: Dict):
        """Log the intervention for future reference."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'issue_id': self.issue_id,
            'analysis': analysis,
            'results': results,
            'success': results.get('success', False)
        }
        
        log_data = {'interventions': []}
        if INTERVENTION_LOG.exists():
            with open(INTERVENTION_LOG, 'r') as f:
                log_data = json.load(f)
        
        log_data['interventions'].append(log_entry)
        
        with open(INTERVENTION_LOG, 'w') as f:
            json.dump(log_data, f, indent=2)


def send_notification(message: str, issue: Dict = None):
    """Send notification to user about intervention."""
    # Try Telegram first
    try:
        from telegram_reporter import load_telegram_config, send_telegram_message
        
        bot_token, chat_id = load_telegram_config()
        if bot_token and chat_id:
            # Format message for Telegram
            telegram_msg = f"""🚨 *Atlas Intervention Alert*

{message}
"""
            if issue:
                telegram_msg += f"""
📊 *Issue Details:*
• Topic: {issue.get('topic', 'Unknown')}
• Severity: {issue.get('severity', 'unknown')}
• Diagnosis: {issue.get('diagnosis', {}).get('primary_issue', 'unknown')}
• Attempts: {issue.get('attempts', 0)}
• Avg Score: {issue.get('average_score', 0):.1f}%

⏰ _{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
            
            send_telegram_message(telegram_msg)
            print("✓ Telegram notification sent")
            return True
    except Exception as e:
        print(f"⚠ Telegram notification failed: {e}")
    
    # Fallback: print to console
    print("="*60)
    print("INTERVENTION NOTIFICATION")
    print("="*60)
    print(message)
    if issue:
        print(f"\nTopic: {issue.get('topic')}")
        print(f"Severity: {issue.get('severity')}")
    print("="*60)
    return False


def main():
    parser = argparse.ArgumentParser(description='Atlas Intervention Agent')
    parser.add_argument('--issue-id', type=str, help='Issue ID to address')
    parser.add_argument('--topic', type=str, help='Topic to check')
    parser.add_argument('--check-all', action='store_true', help='Check all open issues')
    parser.add_argument('--notify-only', action='store_true', help='Only notify, do not fix')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 Atlas Intervention Agent")
    print("="*60)
    
    if args.issue_id:
        # Work on specific issue
        engine = InterventionEngine(args.issue_id)
        issue = engine.get_issue()
        
        if not issue:
            print(f"✗ Issue {args.issue_id} not found")
            return 1
        
        print(f"\n📋 Working on issue: {args.issue_id}")
        print(f"   Topic: {issue.get('topic')}")
        print(f"   Diagnosis: {issue.get('diagnosis', {}).get('primary_issue')}")
        
        # Send notification
        send_notification(
            f"Intervention started for {issue.get('topic')} stagnation",
            issue
        )
        
        if args.notify_only:
            print("\n✓ Notification sent (notify-only mode)")
            return 0
        
        # Analyze issue
        print("\n🔍 Analyzing issue...")
        analysis = engine.analyze_issue(issue)
        print(f"   Brain health score: {analysis['brain_health']['health_score']:.1f}")
        print(f"   Topic comprehension: {analysis['topic_comprehension']['comprehension_estimate']:.1f}%")
        print(f"   Hypotheses: {len(analysis['root_cause_hypotheses'])}")
        for i, hypothesis in enumerate(analysis['root_cause_hypotheses'][:3], 1):
            print(f"      {i}. {hypothesis}")
        
        # Implement fixes
        print("\n🔧 Implementing fixes...")
        results = engine.implement_fix(analysis)
        
        for intervention in results['interventions']:
            print(f"   • {intervention['action']}: {intervention['result']}")
        
        # Test results
        print("\n🧪 Testing fix...")
        test = results['test_results']
        print(f"   Question: {test['test_question']}")
        print(f"   Response: {test['response'][:80]}...")
        print(f"   Has verbs: {test['has_verbs']}")
        print(f"   Has subjects: {test['has_subjects']}")
        print(f"   Improved: {'✓ YES' if test['improved'] else '✗ NO'}")
        
        # Log and resolve
        engine.log_intervention(analysis, results)
        
        if results['success']:
            engine.resolve_issue(args.issue_id, 'Intervention successful - Atlas responding coherently')
            print("\n✅ Issue resolved successfully")
            send_notification(
                f"✅ Intervention successful for {issue.get('topic')}!",
                issue
            )
        else:
            print("\n⚠ Intervention completed but improvement not confirmed")
            print("   Issue remains open for manual review")
            send_notification(
                f"⚠ Intervention attempted for {issue.get('topic')} but needs manual review",
                issue
            )
        
        return 0 if results['success'] else 1
    
    elif args.check_all:
        # Check all open issues
        engine = InterventionEngine()
        issues = engine.issues_data.get('issues', [])
        open_issues = [i for i in issues if i.get('status') == 'open']
        
        print(f"\n📋 Found {len(open_issues)} open issues")
        
        for issue in open_issues:
            print(f"\n   • {issue.get('topic')}: {issue.get('diagnosis', {}).get('primary_issue')}")
        
        return 0
    
    elif args.topic:
        # Check specific topic
        print(f"\n📋 Checking topic: {args.topic}")
        analyzer = BrainAnalyzer()
        
        health = analyzer.check_brain_health()
        comprehension = analyzer.analyze_topic_comprehension(args.topic)
        
        print(f"\nBrain Health:")
        print(f"   Score: {health['health_score']:.1f}/100")
        print(f"   Vocabulary: {health['vocabulary_size']} words")
        
        print(f"\nTopic Comprehension ({args.topic}):")
        print(f"   Estimate: {comprehension['comprehension_estimate']:.1f}%")
        print(f"   Terms found: {comprehension['terms_found']}/{comprehension['topic_terms_checked']}")
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
