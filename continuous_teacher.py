#!/usr/bin/env python3
"""
Atlas Fixed Evaluator - v6 - NEVER GIVES UP

Fixed evaluation system with coherence checking and persistent adaptive learning.
- SHU: Must have 70%+ keywords AND coherence score > 0.5
- HA: Must show principle application in coherent sentences
- RI: Must generate understandable teaching explanation
- NO MAX RETRIES - keeps trying until Atlas passes
- Targeted learning between attempts
"""

import sys
import os
import random
import time
import fcntl
import json
import re
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# CRITICAL: Remove any paths that might cause importing the wrong shared_brain
workspace_root = '/root/.openclaw/workspace'
if workspace_root in sys.path:
    sys.path.remove(workspace_root)

# Add Atlas paths FIRST (highest priority)
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

# Import the CORRECT shared_brain
from shared_brain import get_shared_brain, save_shared_brain, reset_shared_brain

# Import assessment history tracker
from assessment_history_tracker import log_assessment, get_tracker as get_assessment_tracker

# Import coherence evaluator
from coherence_evaluator import evaluate_coherence, CoherenceEvaluator, CoherenceIssue

# Constants
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
PID_FILE = ATLAS_DIR / 'teacher_state' / 'continuous_teacher.pid'
LOCK_FILE = ATLAS_DIR / 'teacher_state' / 'shared_brain.lock'
LOG_FILE = ATLAS_DIR / 'logs' / 'continuous_teacher.log'
CONVERSATION_FILE = ATLAS_DIR / 'teacher_state' / 'conversations.json'
SESSION_FILE = ATLAS_DIR / 'teacher_state' / 'session_stats.json'
MASTERY_FILE = ATLAS_DIR / 'teacher_state' / 'hierarchical_mastery.json'
CONCEPT_MASTERY_FILE = ATLAS_DIR / 'teacher_state' / 'concept_mastery.json'
STAGNATION_FILE = ATLAS_DIR / 'teacher_state' / 'atlas_learning_issues.json'  # NEW: Stagnation reports

# Stagnation detection constants
STAGNATION_THRESHOLD = 5  # Number of consecutive failures to trigger stagnation
STAGNATION_SCORE_WINDOW = 5  # Number of attempts to check for score improvement
MIN_SCORE_IMPROVEMENT = 5.0  # Minimum % improvement to not be considered stagnant


class StagnationDetector:
    """
    Detects when Atlas is failing to improve on a topic and generates diagnostic reports.
    """
    
    def __init__(self, state_file: str = None):
        self.state_file = state_file or str(STAGNATION_FILE)
        self.issues = []
        self._load_state()
    
    def _load_state(self):
        """Load existing stagnation reports."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.issues = data.get('issues', [])
            except Exception as e:
                print(f"[StagnationDetector] Load error: {e}")
                self.issues = []
    
    def save_state(self):
        """Save stagnation reports."""
        data = {
            'issues': self.issues,
            'last_updated': datetime.now().isoformat(),
            'total_issues': len(self.issues)
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def check_stagnation(self, topic: str, assessments: list) -> Optional[dict]:
        """
        Check if Atlas is stagnant on a topic.
        
        Args:
            topic: Topic name
            assessments: List of assessment dicts for this topic (most recent first)
        
        Returns:
            Stagnation report dict if stagnant, None otherwise
        """
        if len(assessments) < STAGNATION_THRESHOLD:
            return None
        
        # Get recent assessments (oldest first for trend analysis)
        recent = list(reversed(assessments[-STAGNATION_THRESHOLD:]))
        
        # Check if all recent assessments failed
        all_failed = all(not a.get('passed', False) for a in recent)
        if not all_failed:
            return None
        
        # Check if scores are improving
        scores = [a.get('score', 0) for a in recent]
        score_trend = scores[-1] - scores[0]
        
        # If scores are improving significantly, not stagnant
        if score_trend >= MIN_SCORE_IMPROVEMENT:
            return None
        
        # Check if we've already reported this stagnation
        for issue in self.issues:
            if (issue.get('topic') == topic and 
                issue.get('status') == 'open' and
                issue.get('last_attempt') == recent[-1].get('timestamp')):
                return None  # Already reported
        
        # Generate stagnation report
        report = self._generate_report(topic, recent, scores, score_trend)
        self.issues.append(report)
        self.save_state()
        
        return report
    
    def _generate_report(self, topic: str, assessments: list, scores: list, trend: float) -> dict:
        """Generate a detailed stagnation report."""
        
        # Collect response samples
        response_samples = []
        for a in assessments[-3:]:  # Last 3 responses
            response_samples.append({
                'attempt': a.get('attempt_number', 0),
                'question': a.get('question', ''),
                'response': a.get('response', '')[:200] + '...' if len(a.get('response', '')) > 200 else a.get('response', ''),
                'score': a.get('score', 0),
                'feedback': a.get('feedback', '')
            })
        
        # Diagnose the issue
        diagnosis = self._diagnose(assessments, scores)
        
        report = {
            'id': f"stagnation_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'topic': topic,
            'detected_at': datetime.now().isoformat(),
            'status': 'open',
            'severity': 'high' if scores[-1] < 30 else 'medium',
            'attempts': len(assessments),
            'score_history': scores,
            'score_trend': trend,
            'average_score': sum(scores) / len(scores),
            'last_score': scores[-1],
            'response_samples': response_samples,
            'diagnosis': diagnosis,
            'recommendations': self._generate_recommendations(diagnosis),
            'flagged_for_intervention': True
        }
        
        return report
    
    def _diagnose(self, assessments: list, scores: list) -> dict:
        """Diagnose the cause of stagnation."""
        diagnosis = {
            'primary_issue': 'unknown',
            'secondary_issues': [],
            'coherence_problem': False,
            'keyword_problem': False,
            'off_topic': False
        }
        
        # Check for coherence issues
        coherence_scores = [a.get('coherence_score', 1.0) for a in assessments if 'coherence_score' in a]
        if coherence_scores and sum(coherence_scores) / len(coherence_scores) < 0.5:
            diagnosis['coherence_problem'] = True
            diagnosis['primary_issue'] = 'incoherent_responses'
        
        # Check feedback patterns
        feedback_patterns = {}
        for a in assessments:
            feedback = a.get('feedback', '')
            if 'incoherent' in feedback.lower():
                feedback_patterns['incoherent'] = feedback_patterns.get('incoherent', 0) + 1
            if 'keyword' in feedback.lower():
                feedback_patterns['keywords'] = feedback_patterns.get('keywords', 0) + 1
            if 'off topic' in feedback.lower():
                feedback_patterns['off_topic'] = feedback_patterns.get('off_topic', 0) + 1
        
        # Determine primary issue
        if feedback_patterns.get('incoherent', 0) >= 3:
            diagnosis['primary_issue'] = 'persistent_incoherence'
            diagnosis['secondary_issues'].append('responses_are_gibberish')
        elif feedback_patterns.get('off_topic', 0) >= 3:
            diagnosis['primary_issue'] = 'completely_off_topic'
            diagnosis['off_topic'] = True
        elif feedback_patterns.get('keywords', 0) >= 3:
            diagnosis['primary_issue'] = 'missing_key_concepts'
            diagnosis['keyword_problem'] = True
        elif scores[-1] < 20:
            diagnosis['primary_issue'] = 'complete_failure_to_learn'
            diagnosis['secondary_issues'].append('may_need_curriculum_reset')
        else:
            diagnosis['primary_issue'] = 'slow_progress'
        
        # Check for score plateau
        if max(scores) - min(scores) < 10:
            diagnosis['secondary_issues'].append('score_plateau_no_improvement')
        
        return diagnosis
    
    def _generate_recommendations(self, diagnosis: dict) -> list:
        """Generate recommendations based on diagnosis."""
        recommendations = []
        
        if diagnosis['primary_issue'] == 'persistent_incoherence':
            recommendations.append("URGENT: Atlas is generating incoherent responses. Check brain state.")
            recommendations.append("Consider resetting the shared brain and retraining from scratch.")
            recommendations.append("Review coherence evaluator thresholds.")
        
        if diagnosis['primary_issue'] == 'completely_off_topic':
            recommendations.append("Atlas responses are off-topic. Check topic-keyword mapping.")
            recommendations.append("Consider adding more targeted learning content.")
        
        if diagnosis['primary_issue'] == 'missing_key_concepts':
            recommendations.append("Atlas is not learning key concepts. Strengthen lesson content.")
            recommendations.append("Add more explicit keyword reinforcement in lessons.")
        
        if diagnosis['primary_issue'] == 'complete_failure_to_learn':
            recommendations.append("CRITICAL: Atlas showing complete failure to learn.")
            recommendations.append("Immediate agent team intervention required.")
            recommendations.append("Consider full system diagnostic.")
        
        if 'score_plateau_no_improvement' in diagnosis['secondary_issues']:
            recommendations.append("Scores are plateaued. Try different teaching strategies.")
            recommendations.append("Consider lowering difficulty or changing topics temporarily.")
        
        recommendations.append("Monitor Atlas responses manually for patterns.")
        recommendations.append("Check if vocabulary is being properly updated.")
        
        return recommendations
    
    def get_open_issues(self) -> list:
        """Get all open stagnation issues."""
        return [i for i in self.issues if i.get('status') == 'open']
    
    def resolve_issue(self, issue_id: str, resolution: str):
        """Mark an issue as resolved."""
        for issue in self.issues:
            if issue.get('id') == issue_id:
                issue['status'] = 'resolved'
                issue['resolved_at'] = datetime.now().isoformat()
                issue['resolution'] = resolution
                self.save_state()
                return True
        return False


class ShuHaRiPhase(Enum):
    """The three phases of mastery in Japanese martial arts tradition."""
    SHU = "shu"    # Learn - follow the rules exactly
    HA = "ha"      # Detach - break with tradition, adapt
    RI = "ri"      # Transcend - create your own way


@dataclass
class LevelProgress:
    """Tracks progress for a specific complexity level within a topic."""
    level: int
    phase: str = "shu"  # shu, ha, or ri
    mastery_percentage: float = 0.0  # 0.0 to 100.0
    lessons_completed: int = 0
    exercises_completed: int = 0
    assessments_passed: int = 0
    assessments_failed: int = 0
    retry_count: int = 0  # Track retries on same topic
    last_studied: Optional[str] = None
    phase_started: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LevelProgress':
        return cls(**data)


@dataclass
class TopicMastery:
    """Tracks mastery for a topic across all complexity levels."""
    topic_name: str
    description: str = ""
    levels: Dict[str, LevelProgress] = field(default_factory=dict)
    current_level: int = 1
    max_level: int = 3  # Default to 3 levels per topic
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'topic_name': self.topic_name,
            'description': self.description,
            'levels': {k: v.to_dict() for k, v in self.levels.items()},
            'current_level': self.current_level,
            'max_level': self.max_level,
            'prerequisites': self.prerequisites
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TopicMastery':
        topic = cls(
            topic_name=data['topic_name'],
            description=data.get('description', ''),
            current_level=data.get('current_level', 1),
            max_level=data.get('max_level', 3),
            prerequisites=data.get('prerequisites', [])
        )
        topic.levels = {
            k: LevelProgress.from_dict(v) 
            for k, v in data.get('levels', {}).items()
        }
        return topic
    
    def get_current_level_progress(self) -> Optional[LevelProgress]:
        """Get progress for the current level."""
        level_key = f"level_{self.current_level}"
        return self.levels.get(level_key)
    
    def get_or_create_level(self, level: int) -> LevelProgress:
        """Get or create progress for a specific level."""
        level_key = f"level_{level}"
        if level_key not in self.levels:
            self.levels[level_key] = LevelProgress(
                level=level,
                phase="shu",
                phase_started=datetime.now().isoformat()
            )
        return self.levels[level_key]
    
    def is_level_mastered(self, level: int) -> bool:
        """Check if a specific level is mastered (RI phase with high mastery)."""
        level_key = f"level_{level}"
        if level_key not in self.levels:
            return False
        progress = self.levels[level_key]
        return progress.phase == "ri" and progress.mastery_percentage >= 80.0
    
    def can_advance_to_level(self, level: int) -> bool:
        """Check if can advance to a specific level (prerequisites met)."""
        if level <= 1:
            return True
        # Must master all previous levels
        for prev_level in range(1, level):
            if not self.is_level_mastered(prev_level):
                return False
        return True


class HierarchicalMasterySystem:
    """
    Manages hierarchical Shu-Ha-Ri mastery tracking for Atlas.
    """
    
    # Phase transition thresholds
    SHU_TO_HA_THRESHOLD = 70.0  # 70% mastery to advance SHU → HA
    HA_TO_RI_THRESHOLD = 80.0   # 80% mastery to advance HA → RI
    RI_COMPLETION_THRESHOLD = 90.0  # 90% to consider level fully mastered
    # REMOVED: MAX_RETRIES = 3  # No longer used - persistent adaptive learning now
    
    # Progress increments
    LESSON_COMPLETION_BOOST = 15.0
    EXERCISE_COMPLETION_BOOST = 10.0
    ASSESSMENT_PASS_BOOST = 20.0
    ASSESSMENT_FAIL_PENALTY = -5.0
    
    def __init__(self, state_file: str = None):
        self.state_file = state_file or str(MASTERY_FILE)
        self.topics: Dict[str, TopicMastery] = {}
        self.learning_history: List[dict] = []
        self.total_lessons_taught = 0
        self.total_assessments_given = 0
        self.passed_assessments = 0
        self.failed_assessments = 0
        self.total_retries = 0
        self._load_state()
    
    def _load_state(self):
        """Load mastery state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.topics = {
                    k: TopicMastery.from_dict(v) 
                    for k, v in data.get('topics', {}).items()
                }
                self.learning_history = data.get('learning_history', [])
                self.total_lessons_taught = data.get('total_lessons_taught', 0)
                self.total_assessments_given = data.get('total_assessments_given', 0)
                self.passed_assessments = data.get('passed_assessments', 0)
                self.failed_assessments = data.get('failed_assessments', 0)
                self.total_retries = data.get('total_retries', 0)
                print(f"[Mastery System] Loaded {len(self.topics)} topics from {self.state_file}")
            except Exception as e:
                print(f"[Mastery System] Load error: {e}, initializing fresh")
                self._initialize_default_topics()
        else:
            print(f"[Mastery System] No state file found, initializing defaults")
            self._initialize_default_topics()
    
    def _initialize_default_topics(self):
        """Initialize default topics with hierarchical levels."""
        default_topics = [
            {
                'name': 'Mathematics',
                'description': 'Mathematical concepts from basic arithmetic to advanced calculus',
                'max_level': 3,
                'prerequisites': []
            },
            {
                'name': 'Algebra',
                'description': 'Working with variables, equations, and algebraic structures',
                'max_level': 3,
                'prerequisites': ['Mathematics']
            },
            {
                'name': 'Logic',
                'description': 'Logical reasoning, deduction, and critical thinking',
                'max_level': 3,
                'prerequisites': ['Mathematics']
            },
            {
                'name': 'Programming',
                'description': 'Computer programming concepts and software development',
                'max_level': 3,
                'prerequisites': ['Logic', 'Mathematics']
            },
            {
                'name': 'Algorithms',
                'description': 'Algorithm design, analysis, and optimization',
                'max_level': 3,
                'prerequisites': ['Programming', 'Logic']
            },
            {
                'name': 'Data_Structures',
                'description': 'Organization and storage of data for efficient access',
                'max_level': 3,
                'prerequisites': ['Programming']
            },
            {
                'name': 'Science',
                'description': 'Scientific method and natural phenomena',
                'max_level': 3,
                'prerequisites': ['Mathematics', 'Logic']
            },
            {
                'name': 'Language',
                'description': 'Linguistics, communication, and natural language',
                'max_level': 3,
                'prerequisites': ['Logic']
            }
        ]
        
        for topic_data in default_topics:
            topic = TopicMastery(
                topic_name=topic_data['name'],
                description=topic_data['description'],
                max_level=topic_data['max_level'],
                prerequisites=topic_data['prerequisites']
            )
            # Initialize level 1
            topic.get_or_create_level(1)
            self.topics[topic_data['name']] = topic
    
    def save_state(self):
        """Save mastery state to file."""
        data = {
            'topics': {k: v.to_dict() for k, v in self.topics.items()},
            'learning_history': self.learning_history[-100:],  # Keep last 100 entries
            'total_lessons_taught': self.total_lessons_taught,
            'total_assessments_given': self.total_assessments_given,
            'passed_assessments': self.passed_assessments,
            'failed_assessments': self.failed_assessments,
            'total_retries': self.total_retries,
            'last_saved': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Mastery System] Saved state to {self.state_file}")
    
    def get_topic(self, topic_name: str) -> Optional[TopicMastery]:
        """Get a topic by name."""
        return self.topics.get(topic_name)
    
    def get_current_status(self, topic_name: str) -> dict:
        """Get current learning status for a topic."""
        topic = self.topics.get(topic_name)
        if not topic:
            return {'error': f'Topic {topic_name} not found'}
        
        current_level = topic.get_current_level_progress()
        prerequisites_met = topic.can_advance_to_level(topic.current_level)
        
        return {
            'topic': topic_name,
            'current_level': topic.current_level,
            'phase': current_level.phase if current_level else 'unknown',
            'mastery_percentage': current_level.mastery_percentage if current_level else 0.0,
            'lessons_completed': current_level.lessons_completed if current_level else 0,
            'retry_count': current_level.retry_count if current_level else 0,
            'prerequisites_met': prerequisites_met,
            'prerequisites': topic.prerequisites,
            'is_mastered': topic.is_level_mastered(topic.current_level) if current_level else False
        }
    
    def record_lesson(self, topic_name: str, lesson_title: str, content_summary: str = "") -> dict:
        """Record completion of a lesson and update mastery progress."""
        topic = self.topics.get(topic_name)
        if not topic:
            return {'error': f'Topic {topic_name} not found'}
        
        # Check prerequisites
        if not topic.can_advance_to_level(topic.current_level):
            return {
                'error': f'Prerequisites not met for {topic_name} Level {topic.current_level}',
                'required': [f'{topic_name} Level {i}' for i in range(1, topic.current_level)]
            }
        
        level_progress = topic.get_or_create_level(topic.current_level)
        
        # Record the lesson
        level_progress.lessons_completed += 1
        level_progress.mastery_percentage = min(
            100.0, 
            level_progress.mastery_percentage + self.LESSON_COMPLETION_BOOST
        )
        level_progress.last_studied = datetime.now().isoformat()
        
        self.total_lessons_taught += 1
        
        # Record in history
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'topic': topic_name,
            'level': topic.current_level,
            'phase': level_progress.phase,
            'action': 'lesson_completed',
            'lesson_title': lesson_title,
            'mastery_after': level_progress.mastery_percentage
        })
        
        # Check for phase transitions
        phase_change = self._check_phase_transition(topic, level_progress)
        
        result = {
            'topic': topic_name,
            'level': topic.current_level,
            'phase': level_progress.phase,
            'mastery_percentage': level_progress.mastery_percentage,
            'lessons_completed': level_progress.lessons_completed,
            'retry_count': level_progress.retry_count,
            'phase_change': phase_change,
            'recommendation': self._get_recommendation(topic, level_progress)
        }
        
        self.save_state()
        return result
    
    def record_assessment(self, topic_name: str, passed: bool, score: float = None,
                          coherence_score: float = None) -> dict:
        """Record result of an assessment/quiz."""
        topic = self.topics.get(topic_name)
        if not topic:
            return {'error': f'Topic {topic_name} not found'}
        
        level_progress = topic.get_current_level_progress()
        if not level_progress:
            return {'error': f'No progress found for {topic_name}'}
        
        self.total_assessments_given += 1
        
        if passed:
            level_progress.assessments_passed += 1
            self.passed_assessments += 1
            level_progress.mastery_percentage = min(
                100.0,
                level_progress.mastery_percentage + self.ASSESSMENT_PASS_BOOST
            )
            # Reset retry count on pass
            level_progress.retry_count = 0
        else:
            level_progress.assessments_failed += 1
            self.failed_assessments += 1
            level_progress.mastery_percentage = max(
                0.0,
                level_progress.mastery_percentage + self.ASSESSMENT_FAIL_PENALTY
            )
            # Increment retry count on fail
            level_progress.retry_count += 1
            self.total_retries += 1
        
        level_progress.last_studied = datetime.now().isoformat()
        
        # Record in history
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'topic': topic_name,
            'level': topic.current_level,
            'phase': level_progress.phase,
            'action': 'assessment_completed',
            'passed': passed,
            'score': score,
            'coherence_score': coherence_score,
            'mastery_after': level_progress.mastery_percentage,
            'retry_count': level_progress.retry_count
        })
        
        phase_change = self._check_phase_transition(topic, level_progress)
        
        self.save_state()
        return {
            'topic': topic_name,
            'level': topic.current_level,
            'phase': level_progress.phase,
            'mastery_percentage': level_progress.mastery_percentage,
            'assessments_passed': level_progress.assessments_passed,
            'assessments_failed': level_progress.assessments_failed,
            'retry_count': level_progress.retry_count,
            'passed': passed,
            'score': score,
            'coherence_score': coherence_score,
            'phase_change': phase_change,
            'needs_retry': not passed  # NEVER give up - always retry if failed
        }
    
    def _check_phase_transition(self, topic: TopicMastery, level_progress: LevelProgress) -> Optional[dict]:
        """Check and handle phase transitions (SHU → HA → RI → next level)."""
        old_phase = level_progress.phase
        new_phase = old_phase
        advanced_level = False
        
        # Check SHU → HA transition
        if old_phase == "shu" and level_progress.mastery_percentage >= self.SHU_TO_HA_THRESHOLD:
            new_phase = "ha"
            level_progress.phase = new_phase
            level_progress.phase_started = datetime.now().isoformat()
        
        # Check HA → RI transition
        elif old_phase == "ha" and level_progress.mastery_percentage >= self.HA_TO_RI_THRESHOLD:
            new_phase = "ri"
            level_progress.phase = new_phase
            level_progress.phase_started = datetime.now().isoformat()
        
        # Check RI → Next Level transition
        elif old_phase == "ri" and level_progress.mastery_percentage >= self.RI_COMPLETION_THRESHOLD:
            if topic.current_level < topic.max_level:
                # Advance to next level
                topic.current_level += 1
                new_level_progress = topic.get_or_create_level(topic.current_level)
                new_level_progress.phase_started = datetime.now().isoformat()
                advanced_level = True
                
                return {
                    'type': 'level_advancement',
                    'from_level': topic.current_level - 1,
                    'to_level': topic.current_level,
                    'new_phase': 'shu',
                    'message': f'🎉 Advanced to {topic.topic_name} Level {topic.current_level}! Starting in SHU phase.'
                }
        
        if old_phase != new_phase:
            return {
                'type': 'phase_transition',
                'from_phase': old_phase,
                'to_phase': new_phase,
                'level': topic.current_level,
                'message': f'🔄 Phase transition: {old_phase.upper()} → {new_phase.upper()} in {topic.topic_name} Level {topic.current_level}'
            }
        
        return None
    
    def _get_recommendation(self, topic: TopicMastery, level_progress: LevelProgress) -> str:
        """Generate learning recommendation based on current state."""
        phase = level_progress.phase
        mastery = level_progress.mastery_percentage
        retries = level_progress.retry_count
        
        # If failed multiple times, suggest review - persistent learning
        if retries > 0:
            return f"🔄 Persistent learning: Retry #{retries} for {topic.topic_name}. Keep trying until mastered!"
        
        if phase == "shu":
            if mastery < 30:
                return f"📚 Focus on fundamentals of {topic.topic_name} Level {topic.current_level}. Study core concepts and examples."
            elif mastery < 60:
                return f"📝 Practice applying {topic.topic_name} rules. Complete guided exercises."
            else:
                return f"🎯 Near HA phase. Continue practicing to build confidence before adapting techniques."
        
        elif phase == "ha":
            if mastery < 85:
                return f"🔧 Experiment with different approaches in {topic.topic_name}. Try variations and adaptations."
            else:
                return f"🌟 Near RI phase. Start creating your own solutions and teaching others."
        
        elif phase == "ri":
            if mastery < 95:
                return f"💡 Innovate! Create original work in {topic.topic_name}. Mentor others."
            else:
                if topic.current_level < topic.max_level:
                    return f"🏆 Level {topic.current_level} nearly mastered! Prepare for Level {topic.current_level + 1}."
                else:
                    return f"🏆 {topic.topic_name} fully mastered across all levels!"
        
        return "Continue learning!"
    
    def get_learning_path(self, topic_name: str) -> List[dict]:
        """Get the full learning path for a topic across all levels."""
        topic = self.topics.get(topic_name)
        if not topic:
            return []
        
        path = []
        for level_num in range(1, topic.max_level + 1):
            level_key = f"level_{level_num}"
            level_data = topic.levels.get(level_key)
            
            if level_data:
                path.append({
                    'level': level_num,
                    'phase': level_data.phase,
                    'mastery': level_data.mastery_percentage,
                    'lessons': level_data.lessons_completed,
                    'retry_count': level_data.retry_count,
                    'is_current': level_num == topic.current_level,
                    'is_mastered': topic.is_level_mastered(level_num)
                })
            else:
                path.append({
                    'level': level_num,
                    'phase': 'not_started',
                    'mastery': 0.0,
                    'lessons': 0,
                    'retry_count': 0,
                    'is_current': level_num == topic.current_level,
                    'is_mastered': False
                })
        
        return path
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics for reporting."""
        total_topics = len(self.topics)
        total_assessments = self.total_assessments_given
        passed = self.passed_assessments
        failed = self.failed_assessments
        pass_rate = (passed / total_assessments * 100) if total_assessments > 0 else 0.0
        
        return {
            'total_topics': total_topics,
            'total_lessons_taught': self.total_lessons_taught,
            'total_assessments': total_assessments,
            'passed_assessments': passed,
            'failed_assessments': failed,
            'pass_rate': pass_rate,
            'total_retries': self.total_retries
        }


# Teaching topics with Q&A pairs and expected keywords for evaluation
TOPICS = {
    'Mathematics': [
        {
            'topic': 'Fibonacci Sequence',
            'question': 'What is the pattern in the Fibonacci sequence?',
            'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence', '0', '1', '1', '2', '3', '5', '8'],
            'level': 1
        },
        {
            'topic': 'Prime Numbers',
            'question': 'Why is 2 the only even prime number?',
            'keywords': ['divisible', 'factor', 'even', 'divide', 'only', 'two'],
            'level': 1
        },
        {
            'topic': 'Geometry',
            'question': 'How do you calculate the area of a circle?',
            'keywords': ['pi', 'radius', 'squared', 'π', 'r²', 'formula'],
            'level': 1
        },
        {
            'topic': 'Calculus',
            'question': 'What is the difference between a derivative and an integral?',
            'keywords': ['rate', 'change', 'accumulation', 'slope', 'area', 'curve'],
            'level': 2
        },
        {
            'topic': 'Number Theory',
            'question': 'Why are prime numbers important in cryptography?',
            'keywords': ['factorization', 'encryption', 'security', 'multiply', 'difficult', 'RSA'],
            'level': 3
        }
    ],
    'Science': [
        {
            'topic': 'Photosynthesis',
            'question': 'What do plants need for photosynthesis?',
            'keywords': ['sunlight', 'water', 'carbon dioxide', 'CO2', 'light', 'energy'],
            'level': 1
        },
        {
            'topic': 'Newton\'s Laws',
            'question': 'What is Newton\'s third law?',
            'keywords': ['action', 'reaction', 'equal', 'opposite', 'force'],
            'level': 1
        },
        {
            'topic': 'DNA',
            'question': 'What does DNA stand for and what does it do?',
            'keywords': ['deoxyribonucleic', 'acid', 'genetic', 'information', 'heredity', 'genes'],
            'level': 1
        },
        {
            'topic': 'Evolution',
            'question': 'What is natural selection?',
            'keywords': ['survival', 'fittest', 'adaptation', 'environment', 'traits', 'reproduce'],
            'level': 2
        },
        {
            'topic': 'Quantum Mechanics',
            'question': 'What is superposition?',
            'keywords': ['multiple', 'states', 'simultaneously', 'wave', 'particle', 'probability'],
            'level': 3
        }
    ],
    'Programming': [
        {
            'topic': 'Variables and Data Types',
            'question': 'What is the difference between a string and an integer?',
            'keywords': ['text', 'number', 'whole', 'characters', 'numeric', 'type'],
            'level': 1
        },
        {
            'topic': 'Functions',
            'question': 'Why should I use functions in my code?',
            'keywords': ['reuse', 'modular', 'organization', 'readability', 'maintain', 'DRY'],
            'level': 1
        },
        {
            'topic': 'Recursion',
            'question': 'What is a recursive function?',
            'keywords': ['itself', 'self-calling', 'base case', 'smaller', 'repeat'],
            'level': 2
        },
        {
            'topic': 'Object-Oriented Design',
            'question': 'What is a class in programming?',
            'keywords': ['blueprint', 'template', 'object', 'instance', 'attributes', 'methods'],
            'level': 2
        },
        {
            'topic': 'Big O Notation',
            'question': 'What does O(n) mean?',
            'keywords': ['linear', 'complexity', 'time', 'proportional', 'input', 'growth'],
            'level': 3
        }
    ],
    'Language': [
        {
            'topic': 'Etymology',
            'question': 'Where does the word "salary" come from?',
            'keywords': ['salt', 'salarium', 'Latin', 'Roman', 'payment', 'soldier'],
            'level': 1
        },
        {
            'topic': 'Grammar',
            'question': 'What is the difference between a noun and a verb?',
            'keywords': ['thing', 'action', 'person', 'place', 'doing', 'naming'],
            'level': 1
        },
        {
            'topic': 'Syntax',
            'question': 'Why does word order matter in sentences?',
            'keywords': ['meaning', 'structure', 'clarity', 'comprehension', 'grammar'],
            'level': 2
        },
        {
            'topic': 'Semantics',
            'question': 'What is the difference between syntax and semantics?',
            'keywords': ['structure', 'meaning', 'form', 'interpretation', 'rules'],
            'level': 2
        },
        {
            'topic': 'Pragmatics',
            'question': 'What does context mean in language?',
            'keywords': ['situation', 'speaker', 'listener', 'implied', 'meaning', 'environment'],
            'level': 3
        }
    ],
    'Logic': [
        {
            'topic': 'Deductive Reasoning',
            'question': 'What is a syllogism?',
            'keywords': ['premises', 'conclusion', 'argument', 'deduction', 'valid', 'form'],
            'level': 1
        },
        {
            'topic': 'Logical Fallacies',
            'question': 'What is an ad hominem attack?',
            'keywords': ['person', 'character', 'argument', 'attack', 'irrelevant', 'personal'],
            'level': 1
        },
        {
            'topic': 'Set Theory',
            'question': 'What is the difference between a union and an intersection?',
            'keywords': ['combine', 'common', 'both', 'either', 'elements', 'shared'],
            'level': 2
        },
        {
            'topic': 'Boolean Algebra',
            'question': 'What are the three basic Boolean operations?',
            'keywords': ['AND', 'OR', 'NOT', 'conjunction', 'disjunction', 'negation'],
            'level': 2
        },
        {
            'topic': 'Critical Thinking',
            'question': 'What is confirmation bias?',
            'keywords': ['seek', 'confirm', 'ignore', 'contradict', 'evidence', 'preconceptions'],
            'level': 3
        }
    ],
    'Algebra': [
        {
            'topic': 'Linear Equations',
            'question': 'What does solving for x mean?',
            'keywords': ['isolate', 'unknown', 'value', 'variable', 'equation', 'find'],
            'level': 1
        },
        {
            'topic': 'Quadratic Equations',
            'question': 'What is the quadratic formula?',
            'keywords': ['x', 'b²-4ac', 'square root', 'roots', 'solutions', 'ax²'],
            'level': 2
        },
        {
            'topic': 'Functions',
            'question': 'What is the difference between domain and range?',
            'keywords': ['input', 'output', 'possible', 'values', 'function', 'valid'],
            'level': 2
        }
    ],
    'Algorithms': [
        {
            'topic': 'Sorting',
            'question': 'What is the difference between bubble sort and merge sort?',
            'keywords': ['O(n²)', 'O(n log n)', 'efficiency', 'compare', 'divide', 'conquer'],
            'level': 2
        },
        {
            'topic': 'Binary Search',
            'question': 'How does binary search work?',
            'keywords': ['sorted', 'divide', 'middle', 'half', 'eliminate', 'O(log n)'],
            'level': 2
        }
    ],
    'Data_Structures': [
        {
            'topic': 'Arrays vs Lists',
            'question': 'What is the difference between an array and a linked list?',
            'keywords': ['contiguous', 'memory', 'pointers', 'index', 'dynamic', 'fixed'],
            'level': 1
        },
        {
            'topic': 'Hash Tables',
            'question': 'How does a hash table work?',
            'keywords': ['key', 'value', 'hash', 'function', 'bucket', 'lookup'],
            'level': 2
        }
    ]
}


LESSON_TEMPLATES = {
    'Mathematics': """{topic}:

{topic} is a fundamental concept in mathematics. Let me explain how it works.

Key principles:
1. First principle of {topic}
2. Second principle of {topic}  
3. Applications in real world

Example: When we study {topic}, we discover patterns that appear throughout nature.
The mathematical relationships help us understand the underlying structure.
""",
    'Science': """{topic}:

{topic} describes how the natural world operates. This process is essential to understanding.

Key concepts:
1. Mechanism of {topic}
2. Components involved
3. Outcomes and effects

Example: In {topic}, we observe cause and effect relationships.
The scientific method helps us verify these observations through experimentation.
""",
    'Programming': """{topic}:

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
    'Language': """{topic}:

{topic} explores how we use and understand language. This field reveals the structure of communication.

Key elements:
1. Core definition
2. Historical development
3. Modern applications

Example: Analyzing {topic} shows how meaning emerges from structure.
Language evolves through usage patterns across communities.
""",
    'Logic': """{topic}:

{topic} provides tools for clear thinking and valid argumentation. These skills apply everywhere.

Key principles:
1. Basic structure
2. Valid forms
3. Common errors to avoid

Example argument demonstrating {topic}:
Premise 1: All humans are mortal
Premise 2: Socrates is human
Conclusion: Socrates is mortal
""",
    'Algebra': """{topic}:

{topic} is fundamental to algebraic thinking and problem solving.

Key concepts:
1. Definition and properties
2. Solving techniques
3. Real-world applications

Example: Understanding {topic} allows us to model and solve complex problems.
""",
    'Algorithms': """{topic}:

{topic} is essential for efficient computation and problem solving.

Key aspects:
1. Algorithm design principles
2. Time and space complexity
3. Implementation considerations

Example: Mastering {topic} enables writing efficient, scalable code.
""",
    'Data_Structures': """{topic}:

{topic} is crucial for organizing and accessing data efficiently.

Key concepts:
1. Structure and properties
2. Operations and complexity
3. Use cases and trade-offs

Example: Choosing the right {topic} can dramatically improve program performance.
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


def save_conversation(q, a, topic, category, evaluation=None):
    """Save a conversation to history"""
    conversations = load_conversations()
    
    # Convert evaluation to serializable format if needed
    if evaluation is not None:
        if hasattr(evaluation, 'to_dict'):
            evaluation = evaluation.to_dict()
        elif hasattr(evaluation, '__dict__'):
            evaluation = evaluation.__dict__
    
    conversation = {
        'timestamp': datetime.now().isoformat(),
        'time': datetime.now().strftime('%H:%M'),
        'topic': topic,
        'category': category,
        'q': q,
        'a': a,
        'evaluation': evaluation
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


# ============================================================================
# FIXED SHU-HA-RI EVALUATION FUNCTIONS WITH COHERENCE CHECKING
# ============================================================================

def evaluate_response_shu(response: str, question_data: dict, topic: str = None) -> Tuple[bool, float, str, dict]:
    """
    FIXED SHU Phase Evaluation: Check for keyword match AND coherence.
    Student must demonstrate they know the fundamental facts/definitions
    AND express them in a coherent way.
    
    Returns: (passed, score, feedback, details)
    """
    response_lower = response.lower()
    keywords = question_data.get('keywords', [])
    
    # First check coherence
    coherence_result = evaluate_coherence(response, topic, threshold=0.5)
    
    if not keywords:
        # No keywords defined - rely on coherence only
        passed = coherence_result.is_coherent
        score = coherence_result.score * 100
        if passed:
            feedback = f"✅ SHU Passed! Response is coherent (score: {coherence_result.score:.2f})."
        else:
            feedback = f"❌ SHU Failed. {coherence_result.get_feedback()}"
        return passed, score, feedback, {'coherence': coherence_result}
    
    # Count matched keywords
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    match_ratio = matched / len(keywords)
    keyword_score = match_ratio * 100
    
    # Combined score: 70% keywords, 30% coherence
    coherence_score = coherence_result.score * 100
    combined_score = (keyword_score * 0.7) + (coherence_score * 0.3)
    
    # Pass threshold: 70% combined AND coherence > 0.5
    passed = combined_score >= 70.0 and coherence_result.is_coherent
    
    details = {
        'keyword_score': keyword_score,
        'coherence_score': coherence_score,
        'combined_score': combined_score,
        'keywords_matched': f"{matched}/{len(keywords)}",
        'coherence': coherence_result
    }
    
    if passed:
        feedback = f"✅ SHU Passed! Keywords: {matched}/{len(keywords)} ({keyword_score:.1f}%), Coherence: {coherence_score:.1f}%, Combined: {combined_score:.1f}%"
    else:
        if not coherence_result.is_coherent:
            feedback = f"❌ SHU Failed. Response is incoherent: {get_coherence_feedback(coherence_result)} Keywords: {matched}/{len(keywords)}"
        else:
            feedback = f"❌ SHU Failed. Only matched {matched}/{len(keywords)} keywords ({keyword_score:.1f}%). Need 70%+ AND coherent response."
    
    return passed, combined_score, feedback, details


def evaluate_response_ha(response: str, question_data: dict, topic: str) -> Tuple[bool, float, str, dict]:
    """
    FIXED HA Phase Evaluation: Check for principle application in coherent sentences.
    Student must apply the concept to a new situation AND be coherent.
    
    Returns: (passed, score, feedback, details)
    """
    response_lower = response.lower()
    keywords = question_data.get('keywords', [])
    
    # Check coherence first
    coherence_result = evaluate_coherence(response, topic, threshold=0.5)
    
    # Application indicators
    application_words = [
        'example', 'apply', 'using', 'therefore', 'thus', 'because', 'since',
        'if', 'then', 'when', 'would', 'could', 'can', 'will', 'should',
        'this means', 'in practice', 'for instance', 'such as', 'like when',
        'consider', 'imagine', 'suppose', 'let\'s say', 'in this case'
    ]
    
    # Check for keyword presence (basic knowledge)
    matched_keywords = sum(1 for kw in keywords if kw.lower() in response_lower)
    keyword_score = (matched_keywords / len(keywords)) * 40 if keywords else 20
    
    # Check for application language
    app_matches = sum(1 for word in application_words if word in response_lower)
    app_score = min(30, app_matches * 5)  # Max 30 points for application words
    
    # Check for novel context (response length and structure)
    word_count = len(response.split())
    length_score = min(15, word_count / 5)  # Max 15 points for length
    
    # Coherence score (max 15 points)
    coherence_score = coherence_result.score * 15
    
    total_score = keyword_score + app_score + length_score + coherence_score
    
    # Pass threshold: 75% (demonstrates both knowledge and application) AND coherent
    passed = total_score >= 75.0 and coherence_result.is_coherent
    
    details = {
        'keyword_score': keyword_score,
        'application_score': app_score,
        'length_score': length_score,
        'coherence_score': coherence_score,
        'total_score': total_score,
        'coherence': coherence_result
    }
    
    if passed:
        feedback = f"✅ HA Passed! Score: {total_score:.1f}% - Shows application of {topic} principles with coherence."
    else:
        if not coherence_result.is_coherent:
            feedback = f"❌ HA Failed. Response is incoherent: {get_coherence_feedback(coherence_result)}"
        else:
            feedback = f"❌ HA Failed. Score: {total_score:.1f}%. Need to show more application of principles in coherent sentences."
    
    return passed, total_score, feedback, details


def evaluate_response_ri(response: str, question_data: dict, topic: str) -> Tuple[bool, float, str, dict]:
    """
    FIXED RI Phase Evaluation: Check for teaching ability and synthesis.
    Student must demonstrate they can explain the concept clearly to a beginner.
    
    Returns: (passed, score, feedback, details)
    """
    response_lower = response.lower()
    
    # Check coherence first
    coherence_result = evaluate_coherence(response, topic, threshold=0.6)  # Higher threshold for RI
    
    # Teaching indicators
    teaching_words = [
        'imagine', 'think of', 'picture', 'like', 'similar to', 'analogy',
        'in other words', 'to put it simply', 'basically', 'essentially',
        'the key idea', 'what this means', 'you can think', 'remember that',
        'let me explain', 'think about it this way', 'here\'s how'
    ]
    
    # Structure indicators
    structure_words = [
        'first', 'second', 'third', 'finally', 'step', 'stage', 'process',
        'next', 'then', 'after', 'before', 'lastly'
    ]
    
    # Synthesis indicators
    synthesis_words = [
        'therefore', 'in conclusion', 'overall', 'combining', 'together',
        'this shows', 'we can see', 'the result', 'in summary'
    ]
    
    # Score components
    teaching_matches = sum(1 for word in teaching_words if word in response_lower)
    teaching_score = min(30, teaching_matches * 5)
    
    structure_matches = sum(1 for word in structure_words if word in response_lower)
    structure_score = min(25, structure_matches * 5)
    
    synthesis_matches = sum(1 for word in synthesis_words if word in response_lower)
    synthesis_score = min(25, synthesis_matches * 5)
    
    # Coherence score (max 20 points)
    coherence_score = coherence_result.score * 20
    
    total_score = teaching_score + structure_score + synthesis_score + coherence_score
    
    # Pass threshold: 80% (demonstrates mastery and teaching ability) AND coherent
    passed = total_score >= 80.0 and coherence_result.is_coherent
    
    details = {
        'teaching_score': teaching_score,
        'structure_score': structure_score,
        'synthesis_score': synthesis_score,
        'coherence_score': coherence_score,
        'total_score': total_score,
        'coherence': coherence_result
    }
    
    if passed:
        feedback = f"✅ RI Passed! Score: {total_score:.1f}% - Demonstrates teaching mastery of {topic}."
    else:
        if not coherence_result.is_coherent:
            feedback = f"❌ RI Failed. Response is incoherent: {get_coherence_feedback(coherence_result)}"
        else:
            feedback = f"❌ RI Failed. Score: {total_score:.1f}%. Need clearer explanation for beginners."
    
    return passed, total_score, feedback, details


def evaluate_response(response: str, question_data: dict, phase: str, topic: str) -> dict:
    """
    Main evaluation function that routes to the appropriate phase evaluator.
    """
    if phase == "shu":
        passed, score, feedback, details = evaluate_response_shu(response, question_data, topic)
    elif phase == "ha":
        passed, score, feedback, details = evaluate_response_ha(response, question_data, topic)
    elif phase == "ri":
        passed, score, feedback, details = evaluate_response_ri(response, question_data, topic)
    else:
        passed, score, feedback, details = False, 0.0, f"Unknown phase: {phase}", {}
    
    coherence = details.get('coherence', None)
    coherence_score = coherence.score if coherence else 0.0
    
    # Convert coherence to dict for JSON serialization
    if coherence is not None and hasattr(coherence, 'to_dict'):
        details['coherence'] = coherence.to_dict()
    
    return {
        'passed': passed,
        'score': score,
        'feedback': feedback,
        'phase': phase,
        'topic': topic,
        'question': question_data.get('question', ''),
        'coherence_score': coherence_score,
        'details': details
    }


# ============================================================================
# FIXED TEACHING FUNCTIONS WITH RETRY LOGIC
# ============================================================================

def get_coherence_feedback(coherence_result) -> str:
    """Generate human-readable feedback from coherence result."""
    from coherence_evaluator import CoherenceIssue
    
    # Handle both dict (JSON serialized) and object
    if isinstance(coherence_result, dict):
        is_coherent = coherence_result.get('is_coherent', True)
        score = coherence_result.get('score', 0.0)
        issues = coherence_result.get('issues', [])
    else:
        is_coherent = coherence_result.is_coherent
        score = coherence_result.score
        issues = coherence_result.issues
    
    if is_coherent:
        return f"Response is coherent (score: {score:.2f})."
    
    feedback_parts = []
    for issue in issues:
        # Handle issue as string (from JSON) or enum
        issue_val = issue if isinstance(issue, str) else issue.value
        
        if issue_val == CoherenceIssue.CODE_NOISE.value:
            feedback_parts.append("contains programming/code terms")
        elif issue_val == CoherenceIssue.NO_VERBS.value:
            feedback_parts.append("lacks verbs (not a complete sentence)")
        elif issue_val == CoherenceIssue.NO_SUBJECTS.value:
            feedback_parts.append("lacks clear subject")
        elif issue_val == CoherenceIssue.TOO_SHORT.value:
            feedback_parts.append("too short to evaluate")
        elif issue_val == CoherenceIssue.WORD_SALAD.value:
            feedback_parts.append("appears to be random word combinations")
        elif issue_val == CoherenceIssue.OFF_TOPIC.value:
            feedback_parts.append("not relevant to the topic")
    
    if feedback_parts:
        return f"Response is incoherent (score: {score:.2f}): {', '.join(feedback_parts)}."
    else:
        return f"Response is incoherent (score: {score:.2f})."


def generate_corrective_feedback(evaluation: dict, topic_data: dict, topic: str) -> str:
    """
    Generate corrective feedback based on what was wrong with the response.
    """
    coherence = evaluation.get('details', {}).get('coherence', None)
    phase = evaluation.get('phase', 'shu')
    
    feedback_parts = []
    
    # Add coherence-related feedback
    from coherence_evaluator import CoherenceIssue
    
    # Handle coherence as dict (JSON serialized) or object
    if coherence:
        is_coherent = coherence.get('is_coherent', True) if isinstance(coherence, dict) else coherence.is_coherent
        issues = coherence.get('issues', []) if isinstance(coherence, dict) else coherence.issues
        
        if not is_coherent:
            for issue in issues:
                # Handle issue as string (from JSON) or enum
                issue_val = issue if isinstance(issue, str) else issue.value
                
                if issue_val == CoherenceIssue.CODE_NOISE.value:
                    feedback_parts.append("Your response contains programming terms that don't belong in this answer.")
                elif issue_val == CoherenceIssue.NO_VERBS.value:
                    feedback_parts.append("Your response needs complete sentences with verbs.")
                elif issue_val == CoherenceIssue.NO_SUBJECTS.value:
                    feedback_parts.append("Make sure your sentences have clear subjects.")
                elif issue_val == CoherenceIssue.TOO_SHORT.value:
                    feedback_parts.append("Your response is too short. Please provide a more complete answer.")
                elif issue_val == CoherenceIssue.WORD_SALAD.value:
                    feedback_parts.append("Your response seems to be random words. Please form complete thoughts.")
                elif issue_val == CoherenceIssue.OFF_TOPIC.value:
                    feedback_parts.append(f"Your response doesn't address the topic of {topic}.")
    
    # Add phase-specific feedback
    if phase == 'shu':
        if not evaluation.get('passed'):
            feedback_parts.append(f"Focus on the key facts about {topic_data.get('topic', topic)}.")
            keywords = topic_data.get('keywords', [])
            if keywords:
                feedback_parts.append(f"Key terms to include: {', '.join(keywords[:5])}.")
    elif phase == 'ha':
        if not evaluation.get('passed'):
            feedback_parts.append(f"Try to apply the {topic} concept to a specific example.")
    elif phase == 'ri':
        if not evaluation.get('passed'):
            feedback_parts.append(f"Explain {topic} as if teaching a beginner. Use analogies and simple language.")
    
    if feedback_parts:
        return " ".join(feedback_parts)
    else:
        return f"Review the material on {topic_data.get('topic', topic)} and try again."


def generate_targeted_learning(evaluation: dict, topic_data: dict, topic: str) -> str:
    """
    Generate targeted learning content based on failure analysis.
    This helps Atlas learn from mistakes and improve.
    """
    coherence = evaluation.get('details', {}).get('coherence', None)
    topic_name = topic_data.get('topic', topic)
    keywords = topic_data.get('keywords', [])
    
    # Handle coherence as dict or object
    if isinstance(coherence, dict):
        is_coherent = coherence.get('is_coherent', True)
        issues = coherence.get('issues', [])
    else:
        is_coherent = coherence.is_coherent if coherence else True
        issues = coherence.issues if coherence else []
    
    lesson_parts = [f"Let's learn from the previous attempt about {topic_name}."]
    
    # Add specific guidance based on issues
    issue_values = [i.value if hasattr(i, 'value') else i for i in issues]
    
    if CoherenceIssue.CODE_NOISE.value in issue_values:
        lesson_parts.append("Use natural language, not programming terms like 'engine', 'state_norm', or 'debug'.")
    
    if CoherenceIssue.NO_VERBS.value in issue_values:
        lesson_parts.append("Make sure your sentences have action words like 'is', 'are', 'works', or 'means'.")
    
    if CoherenceIssue.NO_SUBJECTS.value in issue_values:
        lesson_parts.append("Start your sentences with clear subjects like 'The Fibonacci sequence' or 'It'.")
    
    if CoherenceIssue.TOO_SHORT.value in issue_values:
        lesson_parts.append("Write at least 2-3 complete sentences to fully answer the question.")
    
    if CoherenceIssue.WORD_SALAD.value in issue_values:
        lesson_parts.append("Connect your ideas logically. Each sentence should follow from the previous one.")
    
    if CoherenceIssue.OFF_TOPIC.value in issue_values:
        lesson_parts.append(f"Stay focused on {topic_name}. Don't wander to unrelated topics.")
    
    # Add keyword guidance if missing keywords
    if not is_coherent or evaluation.get('details', {}).get('keyword_score', 100) < 70:
        if keywords:
            keyword_text = ', '.join(keywords[:5])
            lesson_parts.append(f"Key terms to include: {keyword_text}.")
    
    # Add the correct answer structure
    lesson_parts.append(f"A good answer about {topic_name} would use clear sentences and include the key concepts.")
    
    return " ".join(lesson_parts)


def teach_lesson_with_qa(topic_data, category, brain, mastery_system, max_retries: int = 3):
    """
    FIXED: Teach a lesson with Q&A interaction and proper Shu-Ha-Ri evaluation.
    Includes retry logic - if Atlas fails assessment, reteach the topic.
    """
    topic_name = topic_data['topic']
    question = topic_data['question']
    level = topic_data.get('level', 1)
    
    # Get current phase and retry count from mastery system
    status = mastery_system.get_current_status(category)
    current_phase = status.get('phase', 'shu')
    retry_count = status.get('retry_count', 0)
    
    # Generate lesson content
    template = LESSON_TEMPLATES.get(category, LESSON_TEMPLATES['Mathematics'])
    content = template.format(topic=topic_name)
    
    # Learn from the lesson content
    result = brain.learn_from_text(content)
    
    # Generate Atlas's response to the question
    context = f"{topic_name.lower()} {question.lower()}"
    atlas_response = brain.generate_text(context, max_length=80)
    
    # Evaluate the response based on current phase
    evaluation = evaluate_response(atlas_response, topic_data, current_phase, category)
    
    # Record the lesson in mastery system
    lesson_result = mastery_system.record_lesson(category, topic_name, content[:50])
    
    # Record assessment based on evaluation
    assessment_result = mastery_system.record_assessment(
        category, 
        evaluation['passed'], 
        evaluation['score'],
        evaluation.get('coherence_score', 0.0)
    )
    
    # Log to assessment history tracker
    try:
        log_assessment(
            topic=category,
            level=level,
            phase=current_phase,
            score=evaluation['score'],
            passed=evaluation['passed'],
            question=question,
            response=atlas_response,
            feedback=evaluation.get('feedback', '')
        )
    except Exception as e:
        log_message(f"[WARNING] Could not log assessment to history: {e}")
    
    # PERSISTENT ADAPTIVE LEARNING: Never give up on Atlas
    if not evaluation['passed']:
        # Generate corrective feedback
        corrective_feedback = generate_corrective_feedback(evaluation, topic_data, category)
        
        # Provide corrective feedback to reinforce learning
        feedback_text = f"Let's review {topic_name}. {corrective_feedback}"
        brain.learn_from_text(feedback_text)
        
        # Log the retry - teacher keeps trying, NEVER gives up
        log_message(f"       🔄 Assessment failed. Persistent learning mode. Retry #{retry_count + 1}. {corrective_feedback[:80]}...")
        
        # Add targeted learning based on failure type
        failure_lesson = generate_targeted_learning(evaluation, topic_data, category)
        brain.learn_from_text(failure_lesson)
        log_message(f"       📚 Targeted lesson applied: {failure_lesson[:60]}...")
    else:
        # Success
        feedback_text = f"Excellent! Your understanding of {topic_name} shows {current_phase.upper()} level mastery. {evaluation['feedback']}"
        brain.learn_from_text(feedback_text)
    
    # Save the conversation with evaluation
    save_conversation(question, atlas_response, topic_name, category, evaluation)
    
    return {
        'brain_result': result,
        'question': question,
        'response': atlas_response,
        'evaluation': evaluation,
        'lesson_result': lesson_result,
        'assessment_result': assessment_result,
        'phase': current_phase,
        'level': level,
        'retry_count': retry_count,
        'needs_retry': not evaluation['passed']  # NEVER give up - always retry if failed
    }


def run_teaching_session(max_lessons: int = 10):
    """Run one complete teaching session with Shu-Ha-Ri evaluation."""
    log_message("=" * 70)
    log_message("🎓 Atlas Continuous Teacher v6 - NEVER GIVES UP with Coherence Checking")
    log_message("=" * 70)
    
    # Load session stats
    session_stats = load_session_stats()
    
    # Initialize mastery system
    mastery_system = HierarchicalMasterySystem()
    
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
    
    # Show current mastery status
    log_message("\n📊 Current Mastery Status:")
    for topic_name in TOPICS.keys():
        status = mastery_system.get_current_status(topic_name)
        path = mastery_system.get_learning_path(topic_name)
        if path:
            current_step = next((p for p in path if p['is_current']), None)
            if current_step:
                retry_indicator = f"(R{current_step['retry_count']})" if current_step['retry_count'] > 0 else ""
                log_message(f"   {topic_name:15} | L{current_step['level']}-{current_step['phase'].upper():4} | {current_step['mastery']:5.1f}% {retry_indicator}")
    
    # Teach lessons with Q&A
    lessons_taught = 0
    questions_asked = 0
    evaluations = []
    retries_triggered = 0
    
    # Flatten topics for selection
    all_topics = []
    for category, questions in TOPICS.items():
        for q_data in questions:
            all_topics.append((category, q_data))
    
    random.shuffle(all_topics)
    
    # Track which topics need retry - PERSISTENT: never give up
    retry_queue = []
    normal_queue = all_topics[:max_lessons * 2]  # Extra topics for persistent retries
    
    # Teach up to max_lessons lessons - persistent adaptive learning
    lesson_count = 0
    while lesson_count < max_lessons and (normal_queue or retry_queue):
        # Prioritize retries - failed topics get immediate attention
        if retry_queue:
            category, topic_data = retry_queue.pop(0)
            is_retry = True
        elif normal_queue:
            category, topic_data = normal_queue.pop(0)
            is_retry = False
        else:
            break
        
        topic_name = topic_data['topic']
        question = topic_data['question']
        
        # Get current status before teaching
        status = mastery_system.get_current_status(category)
        current_phase = status.get('phase', 'shu')
        current_level = status.get('current_level', 1)
        retry_count = status.get('retry_count', 0)
        
        retry_label = "[RETRY]" if is_retry else "[NEW]"
        log_message(f"\n[{lesson_count+1}/{max_lessons}] {retry_label} Teaching: {topic_name} ({category})")
        log_message(f"       Level: {current_level} | Phase: {current_phase.upper()} | Attempts: {retry_count + 1}")
        log_message(f"       Q: {question}")
        
        # Acquire lock before teaching and saving
        with FileLock(LOCK_FILE):
            result = teach_lesson_with_qa(topic_data, category, brain, mastery_system)
            current_vocab = result['brain_result']['vocabulary_size']
            lessons_taught += 1
            questions_asked += 1
            evaluations.append(result['evaluation'])
            
            # PERSISTENT: Always retry failed topics - NEVER give up
            if result.get('needs_retry'):
                retries_triggered += 1
                # Add back to retry queue - will keep trying until passed
                retry_queue.append((category, topic_data))
                log_message(f"       🔄 Topic will be retried - persistent learning engaged")
            else:
                log_message(f"       ✅ Topic mastered - moving on!")
            
            # Log the Q&A
            response = result['response']
            coherence = result['evaluation'].get('coherence_score', 0.0)
            log_message(f"       A: {response[:80]}{'...' if len(response) > 80 else ''}")
            log_message(f"       📊 Evaluation: {result['evaluation']['feedback']}")
            log_message(f"       📊 Coherence: {coherence:.2f}")
            
            # Check for phase/level changes
            if result['assessment_result'].get('phase_change'):
                change = result['assessment_result']['phase_change']
                log_message(f"       🔔 {change['message']}")
            
            log_message(f"       -> Tokens: {result['brain_result']['tokens_processed']}, Vocab: {current_vocab}")
            
            # Save checkpoint every 5 lessons
            if lessons_taught % 5 == 0:
                save_shared_brain()
                log_message(f"💾 Checkpoint saved after {lessons_taught} lessons")
        
        lesson_count += 1
    
    # Final save with lock
    with FileLock(LOCK_FILE):
        save_shared_brain()
    
    # Update session stats
    session_stats['total_lessons'] += lessons_taught
    session_stats['total_questions'] += questions_asked
    session_stats['total_conversations'] += lessons_taught
    session_stats['sessions_completed'] += 1
    session_stats['last_session_time'] = datetime.now().isoformat()
    save_session_stats(session_stats)
    
    # Report results
    final_stats = brain.get_stats()
    final_vocab = final_stats['vocabulary_size']
    vocab_added = final_vocab - initial_vocab
    
    # Count evaluations
    passed_count = sum(1 for e in evaluations if e['passed'])
    failed_count = len(evaluations) - passed_count
    
    # Get summary stats
    summary = mastery_system.get_summary_stats()
    
    log_message("\n" + "=" * 70)
    log_message("📊 Session Complete - FIXED Evaluation System")
    log_message("=" * 70)
    log_message(f"Lessons taught: {lessons_taught}")
    log_message(f"Questions asked: {questions_asked}")
    log_message(f"Evaluations passed: {passed_count}/{len(evaluations)} ({passed_count/len(evaluations)*100:.1f}%)")
    log_message(f"Evaluations failed: {failed_count}/{len(evaluations)}")
    log_message(f"Retries triggered: {retries_triggered}")
    log_message(f"Total retries (all time): {summary['total_retries']}")
    log_message(f"Vocabulary: {initial_vocab} → {final_vocab} (+{vocab_added})")
    log_message(f"Total tokens: {final_stats['total_tokens_seen']:,}")
    log_message(f"Total sessions completed: {session_stats['sessions_completed']}")
    
    # Show updated mastery status
    log_message("\n📊 Updated Mastery Status:")
    for topic_name in TOPICS.keys():
        status = mastery_system.get_current_status(topic_name)
        path = mastery_system.get_learning_path(topic_name)
        if path:
            current_step = next((p for p in path if p['is_current']), None)
            if current_step:
                mastered = "✅" if current_step['is_mastered'] else "⏳"
                retry_indicator = f"(R{current_step['retry_count']})" if current_step['retry_count'] > 0 else ""
                log_message(f"   {topic_name:15} | L{current_step['level']}-{current_step['phase'].upper():4} | {current_step['mastery']:5.1f}% {retry_indicator} {mastered}")
    
    log_message("=" * 70)
    
    # STAGNATION DETECTION: Check for learning issues
    log_message("\n🔍 Checking for stagnation...")
    stagnation_detector = StagnationDetector()
    
    # Get assessment tracker
    try:
        assessment_tracker = get_assessment_tracker()
        stagnation_found = False
        
        for topic_name in TOPICS.keys():
            topic_assessments = assessment_tracker.get_assessments_for_topic(topic_name)
            if len(topic_assessments) >= STAGNATION_THRESHOLD:
                # Convert to dicts for stagnation detector
                assessment_dicts = [a.to_dict() for a in topic_assessments]
                report = stagnation_detector.check_stagnation(topic_name, assessment_dicts)
                
                if report:
                    stagnation_found = True
                    log_message(f"   🚨 STAGNATION DETECTED: {topic_name}")
                    log_message(f"      Severity: {report['severity']}")
                    log_message(f"      Issue: {report['diagnosis']['primary_issue']}")
                    log_message(f"      Avg Score: {report['average_score']:.1f}%")
                    log_message(f"      Attempts: {report['attempts']}")
                    log_message(f"      Report: {STAGNATION_FILE}")
        
        if not stagnation_found:
            log_message("   ✅ No stagnation detected")
        else:
            open_issues = stagnation_detector.get_open_issues()
            log_message(f"\n   📋 Total open issues: {len(open_issues)}")
            for issue in open_issues[-3:]:  # Show last 3
                log_message(f"      • {issue['topic']}: {issue['diagnosis']['primary_issue']}")
                
    except Exception as e:
        log_message(f"   [WARNING] Stagnation check error: {e}")
    
    return {
        'success': True,
        'lessons_taught': lessons_taught,
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(evaluations) * 100 if evaluations else 0,
        'retries_triggered': retries_triggered,
        'total_retries': summary['total_retries'],
        'vocab_added': vocab_added
    }


def main():
    try:
        result = run_teaching_session(max_lessons=10)
        return 0 if result['success'] else 1
    except Exception as e:
        log_message(f"[ERROR] {e}")
        import traceback
        log_message(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
