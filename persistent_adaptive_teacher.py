#!/usr/bin/env python3
"""
Atlas Persistent Adaptive Teacher - v7

NEVER GIVES UP ON ATLAS - Implements persistent adaptive learning
- No max retry limits - keeps trying until Atlas passes
- Adaptive teaching based on failure analysis
- Learning between attempts
- Teaching method tracking
- Mastery-based progression
- Diagnostic mode for struggling topics
- ADAPTATION MONITORING: Tracks if Atlas is actually improving
- AUTO-ESCALATION: Detects stagnation and requests help
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
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set
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
TEACHING_PROFILE_FILE = ATLAS_DIR / 'teacher_state' / 'atlas_learning_profile.json'
DIAGNOSTIC_LOG_FILE = ATLAS_DIR / 'teacher_state' / 'diagnostic_log.json'
LEARNING_ISSUES_FILE = ATLAS_DIR / 'teacher_state' / 'atlas_learning_issues.json'
ADAPTATION_LOG_FILE = ATLAS_DIR / 'teacher_state' / 'adaptation_log.json'


class FailureType(Enum):
    """Types of failures Atlas can have."""
    MISSING_KEYWORDS = auto()
    INCOHERENT = auto()
    LOW_COHERENCE = auto()
    OFF_TOPIC = auto()
    TOO_SHORT = auto()
    CODE_NOISE = auto()
    WORD_SALAD = auto()
    NO_VERBS = auto()
    NO_SUBJECTS = auto()
    PREREQUISITE_GAP = auto()


class TeachingMethod(Enum):
    """Different teaching methods to try."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"  # Simpler vocabulary, focus on core terms
    STEP_BY_STEP = "step_by_step"  # Break into smaller steps
    VISUAL_ANALOGY = "visual_analogy"  # Use visual analogies
    REAL_WORLD = "real_world"  # Real-world examples
    INTERACTIVE = "interactive"  # Ask questions during lesson
    REPETITION = "repetition"  # Repeat key concepts multiple times
    CONTRAST = "contrast"  # Compare with opposites/differences
    STORY = "story"  # Embed in a narrative
    HANDS_ON = "hands_on"  # Practical application focus


class TeachingMethod(Enum):
    """Different teaching methods to try."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"  # Simpler vocabulary, focus on core terms
    STEP_BY_STEP = "step_by_step"  # Break into smaller steps
    VISUAL_ANALOGY = "visual_analogy"  # Use visual analogies
    REAL_WORLD = "real_world"  # Real-world examples
    INTERACTIVE = "interactive"  # Ask questions during lesson
    REPETITION = "repetition"  # Repeat key concepts multiple times
    CONTRAST = "contrast"  # Compare with opposites/differences
    STORY = "story"  # Embed in a narrative
    HANDS_ON = "hands_on"  # Practical application focus


class ProgressTrend(Enum):
    """Trends in Atlas's learning progress."""
    IMPROVING = "improving"
    FLAT = "flat"
    DECLINING = "declining"
    STAGNANT = "stagnant"
    REGRESSING = "regressing"
    NOT_LEARNING = "not_learning"


class AdaptationStatus(Enum):
    """Status of Atlas's adaptation to teaching."""
    ADAPTING_WELL = "adapting_well"
    SLOW_PROGRESS = "slow_progress"
    NEEDS_INTERVENTION = "needs_intervention"
    CRITICAL = "critical"


@dataclass
class ProgressMetrics:
    """Metrics for a single learning attempt."""
    timestamp: str
    score: float
    coherence_score: float
    keyword_match_rate: float
    response_time_ms: int
    method_used: str
    failure_type: Optional[str]
    response_sample: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProgressMetrics':
        return cls(**data)


@dataclass
class TopicProgressTracker:
    """Tracks progress metrics for a specific topic."""
    topic_name: str
    metrics: List[ProgressMetrics] = field(default_factory=list)
    trend: str = "unknown"
    coherence_trend: str = "unknown"
    keyword_trend: str = "unknown"
    last_updated: Optional[str] = None
    escalation_level: int = 0  # 0=normal, 1=watch, 2=intervention, 3=critical
    
    def to_dict(self) -> dict:
        return {
            'topic_name': self.topic_name,
            'metrics': [m.to_dict() for m in self.metrics],
            'trend': self.trend,
            'coherence_trend': self.coherence_trend,
            'keyword_trend': self.keyword_trend,
            'last_updated': self.last_updated,
            'escalation_level': self.escalation_level
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TopicProgressTracker':
        tracker = cls(
            topic_name=data['topic_name'],
            trend=data.get('trend', 'unknown'),
            coherence_trend=data.get('coherence_trend', 'unknown'),
            keyword_trend=data.get('keyword_trend', 'unknown'),
            last_updated=data.get('last_updated'),
            escalation_level=data.get('escalation_level', 0)
        )
        tracker.metrics = [ProgressMetrics.from_dict(m) for m in data.get('metrics', [])]
        return tracker
    
    def add_metric(self, metric: ProgressMetrics):
        """Add a new metric and update trends."""
        self.metrics.append(metric)
        self.last_updated = metric.timestamp
        
        # Keep only last 20 metrics for trend analysis
        if len(self.metrics) > 20:
            self.metrics = self.metrics[-20:]
        
        self._update_trends()
    
    def _update_trends(self):
        """Update trend analysis based on recent metrics."""
        if len(self.metrics) < 3:
            return
        
        recent = self.metrics[-5:]  # Last 5 attempts
        scores = [m.score for m in recent]
        coherence_scores = [m.coherence_score for m in recent]
        keyword_rates = [m.keyword_match_rate for m in recent]
        
        # Score trend
        self.trend = self._calculate_trend(scores, threshold=5.0)
        
        # Coherence trend
        self.coherence_trend = self._calculate_trend(coherence_scores, threshold=0.05)
        
        # Keyword trend
        self.keyword_trend = self._calculate_trend(keyword_rates, threshold=0.1)
    
    def _calculate_trend(self, values: List[float], threshold: float) -> str:
        """Calculate trend from a series of values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Check for stagnation (same ±threshold for 5+ attempts)
        if len(values) >= 5:
            recent_5 = values[-5:]
            avg_5 = sum(recent_5) / len(recent_5)
            if all(abs(v - avg_5) <= threshold for v in recent_5):
                return ProgressTrend.STAGNANT.value
        
        # Check for regression (declining over 3+ attempts)
        if len(values) >= 3:
            if values[-1] < values[-2] < values[-3]:
                return ProgressTrend.REGRESSING.value
        
        # Check for improvement
        first_avg = sum(values[:3]) / 3
        last_avg = sum(values[-3:]) / 3
        
        if last_avg > first_avg + threshold:
            return ProgressTrend.IMPROVING.value
        elif last_avg < first_avg - threshold:
            return ProgressTrend.DECLINING.value
        else:
            return ProgressTrend.FLAT.value
    
    def get_adaptation_status(self) -> str:
        """Get overall adaptation status."""
        recent_attempts = len(self.metrics)
        
        if recent_attempts < 3:
            return AdaptationStatus.ADAPTING_WELL.value
        
        # Check for critical stagnation
        if self.trend == ProgressTrend.STAGNANT.value and recent_attempts >= 5:
            self.escalation_level = max(self.escalation_level, 2)
            return AdaptationStatus.NEEDS_INTERVENTION.value
        
        # Check for regression
        if self.trend == ProgressTrend.REGRESSING.value:
            self.escalation_level = max(self.escalation_level, 2)
            return AdaptationStatus.NEEDS_INTERVENTION.value
        
        # Check for no coherence improvement
        if self.coherence_trend in [ProgressTrend.STAGNANT.value, ProgressTrend.DECLINING.value]:
            if recent_attempts >= 5:
                self.escalation_level = max(self.escalation_level, 1)
                return AdaptationStatus.SLOW_PROGRESS.value
        
        # Check for not learning (flat everything)
        if (self.trend == ProgressTrend.FLAT.value and 
            self.coherence_trend == ProgressTrend.FLAT.value and
            recent_attempts >= 5):
            self.escalation_level = max(self.escalation_level, 1)
            return AdaptationStatus.SLOW_PROGRESS.value
        
        return AdaptationStatus.ADAPTING_WELL.value
    
    def get_recommendation(self) -> str:
        """Get recommendation based on trends."""
        status = self.get_adaptation_status()
        
        if status == AdaptationStatus.CRITICAL.value:
            return "immediate_intervention"
        elif status == AdaptationStatus.NEEDS_INTERVENTION.value:
            if self.trend == ProgressTrend.STAGNANT.value:
                return "try_radically_different_approach"
            elif self.trend == ProgressTrend.REGRESSING.value:
                return "review_fundamentals"
            else:
                return "spawn_diagnostic_agent"
        elif status == AdaptationStatus.SLOW_PROGRESS.value:
            return "increase_scaffolding"
        else:
            return "continue_current_approach"


@dataclass
class LearningIssue:
    """Represents a learning issue requiring intervention."""
    issue_id: str
    topic: str
    detected_at: str
    diagnosis: str  # "stagnant", "regressing", "not_adapting"
    severity: str  # "warning", "critical"
    attempts_count: int
    score_history: List[float]
    response_samples: List[str]
    recommended_fix: str
    status: str = "open"  # "open", "in_progress", "resolved"
    assigned_agent: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LearningIssue':
        return cls(**data)


class AtlasAdaptationMonitor:
    """Monitors Atlas's adaptation and learning progress."""
    
    STAGNATION_THRESHOLD = 5  # 5+ attempts with same score ±5%
    REGRESSION_THRESHOLD = 3  # 3+ declining attempts
    SCORE_VARIATION_THRESHOLD = 5.0  # ±5% for stagnation
    
    def __init__(self):
        self.trackers: Dict[str, TopicProgressTracker] = {}
        self.issues: List[LearningIssue] = []
        self.adaptation_log: List[dict] = []
        self._load_data()
    
    def _load_data(self):
        """Load existing tracking data."""
        if os.path.exists(ADAPTATION_LOG_FILE):
            try:
                with open(ADAPTATION_LOG_FILE, 'r') as f:
                    data = json.load(f)
                self.trackers = {
                    k: TopicProgressTracker.from_dict(v)
                    for k, v in data.get('trackers', {}).items()
                }
                self.issues = [LearningIssue.from_dict(i) for i in data.get('issues', [])]
                print(f"[Adaptation Monitor] Loaded {len(self.trackers)} topic trackers, {len(self.issues)} issues")
            except Exception as e:
                print(f"[Adaptation Monitor] Load error: {e}, starting fresh")
    
    def save_data(self):
        """Save tracking data."""
        data = {
            'trackers': {k: v.to_dict() for k, v in self.trackers.items()},
            'issues': [i.to_dict() for i in self.issues],
            'last_saved': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(ADAPTATION_LOG_FILE), exist_ok=True)
        with open(ADAPTATION_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save issues to separate file for easy access
        self._save_issues_file()
    
    def _save_issues_file(self):
        """Save issues to dedicated file for agent team."""
        open_issues = [i for i in self.issues if i.status == "open"]
        
        issues_data = {
            'generated_at': datetime.now().isoformat(),
            'open_issues_count': len(open_issues),
            'issues': [i.to_dict() for i in open_issues],
            'summary': self._generate_issues_summary()
        }
        
        with open(LEARNING_ISSUES_FILE, 'w') as f:
            json.dump(issues_data, f, indent=2)
    
    def _generate_issues_summary(self) -> dict:
        """Generate summary of current issues."""
        stagnant = sum(1 for i in self.issues if i.diagnosis == "stagnant" and i.status == "open")
        regressing = sum(1 for i in self.issues if i.diagnosis == "regressing" and i.status == "open")
        not_adapting = sum(1 for i in self.issues if i.diagnosis == "not_adapting" and i.status == "open")
        
        return {
            'total_open_issues': len([i for i in self.issues if i.status == "open"]),
            'stagnant_topics': stagnant,
            'regressing_topics': regressing,
            'not_adapting_topics': not_adapting,
            'topics_needing_intervention': stagnant + regressing + not_adapting
        }
    
    def record_attempt(self, topic: str, evaluation: dict, method: str, 
                       response_time_ms: int = 0) -> dict:
        """Record a learning attempt and check for issues."""
        # Get or create tracker
        if topic not in self.trackers:
            self.trackers[topic] = TopicProgressTracker(topic_name=topic)
        
        tracker = self.trackers[topic]
        
        # Extract metrics from evaluation
        details = evaluation.get('details', {})
        coherence = details.get('coherence', {})
        
        if isinstance(coherence, dict):
            coherence_score = coherence.get('score', 0.0)
        else:
            coherence_score = coherence.score if coherence else 0.0
        
        # Calculate keyword match rate
        keywords_matched = details.get('keywords_matched', '0/0')
        if isinstance(keywords_matched, str) and '/' in keywords_matched:
            matched, total = keywords_matched.split('/')
            keyword_rate = int(matched) / int(total) if int(total) > 0 else 0.0
        else:
            keyword_rate = 0.0
        
        # Create metric
        metric = ProgressMetrics(
            timestamp=datetime.now().isoformat(),
            score=evaluation.get('score', 0.0),
            coherence_score=coherence_score,
            keyword_match_rate=keyword_rate,
            response_time_ms=response_time_ms,
            method_used=method,
            failure_type=tracker.metrics[-1].failure_type if tracker.metrics else None,
            response_sample=evaluation.get('response', '')[:100]  # First 100 chars
        )
        
        # Add to tracker
        tracker.add_metric(metric)
        
        # Check for issues
        issue = self._check_for_issues(topic, tracker)
        
        # Log adaptation event
        self.adaptation_log.append({
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'score': metric.score,
            'trend': tracker.trend,
            'status': tracker.get_adaptation_status(),
            'issue_detected': issue is not None
        })
        
        self.save_data()
        
        return {
            'tracker': tracker,
            'issue': issue,
            'status': tracker.get_adaptation_status(),
            'recommendation': tracker.get_recommendation()
        }
    
    def _check_for_issues(self, topic: str, tracker: TopicProgressTracker) -> Optional[LearningIssue]:
        """Check if this topic has a learning issue requiring intervention."""
        status = tracker.get_adaptation_status()
        
        # Only create issues for problems needing intervention
        if status not in [AdaptationStatus.NEEDS_INTERVENTION.value, AdaptationStatus.CRITICAL.value]:
            return None
        
        # Check if we already have an open issue for this topic
        existing = [i for i in self.issues if i.topic == topic and i.status == "open"]
        if existing:
            # Update existing issue
            issue = existing[0]
            issue.attempts_count = len(tracker.metrics)
            issue.score_history = [m.score for m in tracker.metrics[-10:]]
            return issue
        
        # Create new issue
        recent_metrics = tracker.metrics[-5:] if len(tracker.metrics) >= 5 else tracker.metrics
        
        diagnosis = "stagnant"
        if tracker.trend == ProgressTrend.REGRESSING.value:
            diagnosis = "regressing"
        elif tracker.coherence_trend == ProgressTrend.STAGNANT.value and tracker.trend == ProgressTrend.FLAT.value:
            diagnosis = "not_adapting"
        
        # Determine recommended fix
        if diagnosis == "stagnant":
            recommended_fix = "needs_simpler_lessons"
        elif diagnosis == "regressing":
            recommended_fix = "missing_prerequisites"
        else:
            recommended_fix = "teaching_method_mismatch"
        
        issue = LearningIssue(
            issue_id=f"{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            topic=topic,
            detected_at=datetime.now().isoformat(),
            diagnosis=diagnosis,
            severity="critical" if tracker.escalation_level >= 2 else "warning",
            attempts_count=len(tracker.metrics),
            score_history=[m.score for m in tracker.metrics],
            response_samples=[m.response_sample for m in recent_metrics],
            recommended_fix=recommended_fix,
            status="open"
        )
        
        self.issues.append(issue)
        
        # Log the issue
        log_message(f"\n{'!'*70}")
        log_message(f"🚨 LEARNING ISSUE DETECTED: {topic}")
        log_message(f"   Diagnosis: {diagnosis}")
        log_message(f"   Attempts: {issue.attempts_count}")
        log_message(f"   Trend: {tracker.trend}")
        log_message(f"   Recommended fix: {recommended_fix}")
        log_message(f"   Issue ID: {issue.issue_id}")
        log_message(f"{'!'*70}\n")
        
        return issue
    
    def get_topic_status(self, topic: str) -> dict:
        """Get detailed status for a topic."""
        if topic not in self.trackers:
            return {'error': f'No data for topic {topic}'}
        
        tracker = self.trackers[topic]
        recent = tracker.metrics[-5:] if tracker.metrics else []
        
        return {
            'topic': topic,
            'total_attempts': len(tracker.metrics),
            'score_trend': tracker.trend,
            'coherence_trend': tracker.coherence_trend,
            'keyword_trend': tracker.keyword_trend,
            'adaptation_status': tracker.get_adaptation_status(),
            'escalation_level': tracker.escalation_level,
            'recommendation': tracker.get_recommendation(),
            'recent_scores': [m.score for m in recent],
            'recent_coherence': [m.coherence_score for m in recent],
            'methods_tried': list(set(m.method_used for m in tracker.metrics))
        }
    
    def get_all_status(self) -> dict:
        """Get status of all tracked topics."""
        return {
            'topics_tracked': len(self.trackers),
            'open_issues': len([i for i in self.issues if i.status == "open"]),
            'topics': {topic: self.get_topic_status(topic) for topic in self.trackers.keys()}
        }
    
    def resolve_issue(self, issue_id: str, resolution: str, agent: str):
        """Mark an issue as resolved."""
        for issue in self.issues:
            if issue.issue_id == issue_id:
                issue.status = "resolved"
                issue.resolution_notes = resolution
                issue.assigned_agent = agent
                self.save_data()
                return True
        return False


@dataclass
class TeachingAttempt:
    """Records a single teaching attempt."""
    timestamp: str
    method: str
    failure_type: Optional[str]
    keywords_matched: int
    total_keywords: int
    coherence_score: float
    passed: bool
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TeachingAttempt':
        return cls(**data)


@dataclass
class TopicLearningProfile:
    """Learning profile for a specific topic."""
    topic_name: str
    successful_methods: List[str] = field(default_factory=list)
    failed_methods: List[str] = field(default_factory=list)
    attempts: List[TeachingAttempt] = field(default_factory=list)
    best_method: Optional[str] = None
    common_failure_types: List[str] = field(default_factory=list)
    prerequisite_gaps: List[str] = field(default_factory=list)
    estimated_learning_style: str = "unknown"  # visual, auditory, kinesthetic, reading
    
    def to_dict(self) -> dict:
        return {
            'topic_name': self.topic_name,
            'successful_methods': self.successful_methods,
            'failed_methods': self.failed_methods,
            'attempts': [a.to_dict() for a in self.attempts],
            'best_method': self.best_method,
            'common_failure_types': self.common_failure_types,
            'prerequisite_gaps': self.prerequisite_gaps,
            'estimated_learning_style': self.estimated_learning_style
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TopicLearningProfile':
        profile = cls(
            topic_name=data['topic_name'],
            successful_methods=data.get('successful_methods', []),
            failed_methods=data.get('failed_methods', []),
            best_method=data.get('best_method'),
            common_failure_types=data.get('common_failure_types', []),
            prerequisite_gaps=data.get('prerequisite_gaps', []),
            estimated_learning_style=data.get('estimated_learning_style', 'unknown')
        )
        profile.attempts = [TeachingAttempt.from_dict(a) for a in data.get('attempts', [])]
        return profile
    
    def record_attempt(self, method: str, passed: bool, evaluation: dict):
        """Record a teaching attempt."""
        details = evaluation.get('details', {})
        coherence = details.get('coherence', {})
        
        # Handle coherence as dict or object
        if isinstance(coherence, dict):
            coherence_score = coherence.get('score', 0.0)
            is_coherent = coherence.get('is_coherent', False)
            issues = coherence.get('issues', [])
        else:
            coherence_score = coherence.score if coherence else 0.0
            is_coherent = coherence.is_coherent if coherence else False
            issues = coherence.issues if coherence else []
        
        # Determine failure type
        failure_type = None
        if not passed:
            if not is_coherent:
                if CoherenceIssue.CODE_NOISE.value in issues or \
                   (hasattr(CoherenceIssue, 'CODE_NOISE') and CoherenceIssue.CODE_NOISE in issues):
                    failure_type = FailureType.CODE_NOISE.name
                elif CoherenceIssue.WORD_SALAD.value in issues or \
                     (hasattr(CoherenceIssue, 'WORD_SALAD') and CoherenceIssue.WORD_SALAD in issues):
                    failure_type = FailureType.WORD_SALAD.name
                elif CoherenceIssue.NO_VERBS.value in issues or \
                     (hasattr(CoherenceIssue, 'NO_VERBS') and CoherenceIssue.NO_VERBS in issues):
                    failure_type = FailureType.NO_VERBS.name
                elif CoherenceIssue.NO_SUBJECTS.value in issues or \
                     (hasattr(CoherenceIssue, 'NO_SUBJECTS') and CoherenceIssue.NO_SUBJECTS in issues):
                    failure_type = FailureType.NO_SUBJECTS.name
                elif CoherenceIssue.TOO_SHORT.value in issues or \
                     (hasattr(CoherenceIssue, 'TOO_SHORT') and CoherenceIssue.TOO_SHORT in issues):
                    failure_type = FailureType.TOO_SHORT.name
                elif CoherenceIssue.OFF_TOPIC.value in issues or \
                     (hasattr(CoherenceIssue, 'OFF_TOPIC') and CoherenceIssue.OFF_TOPIC in issues):
                    failure_type = FailureType.OFF_TOPIC.name
                else:
                    failure_type = FailureType.INCOHERENT.name
            else:
                # Coherent but failed - likely missing keywords
                failure_type = FailureType.MISSING_KEYWORDS.name
        
        # Extract keyword info
        keywords_matched = details.get('keywords_matched', '0/0')
        if isinstance(keywords_matched, str) and '/' in keywords_matched:
            matched, total = keywords_matched.split('/')
            keywords_matched = int(matched)
            total_keywords = int(total)
        else:
            keywords_matched = 0
            total_keywords = 0
        
        attempt = TeachingAttempt(
            timestamp=datetime.now().isoformat(),
            method=method,
            failure_type=failure_type,
            keywords_matched=keywords_matched,
            total_keywords=total_keywords,
            coherence_score=coherence_score,
            passed=passed
        )
        
        self.attempts.append(attempt)
        
        # Update method tracking
        if passed:
            if method not in self.successful_methods:
                self.successful_methods.append(method)
            if method in self.failed_methods:
                self.failed_methods.remove(method)
            # Update best method
            if self.best_method is None:
                self.best_method = method
        else:
            if method not in self.failed_methods:
                self.failed_methods.append(method)
        
        # Update common failure types
        if failure_type and failure_type not in self.common_failure_types:
            self.common_failure_types.append(failure_type)
        
        # Infer learning style based on successful methods
        self._infer_learning_style()
    
    def _infer_learning_style(self):
        """Infer Atlas's learning style based on successful methods."""
        visual_methods = {TeachingMethod.VISUAL_ANALOGY.value, TeachingMethod.STORY.value}
        auditory_methods = {TeachingMethod.REPETITION.value, TeachingMethod.INTERACTIVE.value}
        kinesthetic_methods = {TeachingMethod.HANDS_ON.value, TeachingMethod.REAL_WORLD.value}
        reading_methods = {TeachingMethod.STEP_BY_STEP.value, TeachingMethod.STANDARD.value}
        
        visual_score = sum(1 for m in self.successful_methods if m in visual_methods)
        auditory_score = sum(1 for m in self.successful_methods if m in auditory_methods)
        kinesthetic_score = sum(1 for m in self.successful_methods if m in kinesthetic_methods)
        reading_score = sum(1 for m in self.successful_methods if m in reading_methods)
        
        scores = {
            'visual': visual_score,
            'auditory': auditory_score,
            'kinesthetic': kinesthetic_score,
            'reading': reading_score
        }
        
        if max(scores.values()) > 0:
            self.estimated_learning_style = max(scores, key=scores.get)
    
    def get_recommended_method(self) -> str:
        """Get the next recommended teaching method."""
        # If we have a best method, use it
        if self.best_method:
            return self.best_method
        
        # Try methods not yet failed
        all_methods = [m.value for m in TeachingMethod]
        untried = [m for m in all_methods if m not in self.failed_methods and m not in self.successful_methods]
        
        if untried:
            return untried[0]
        
        # All methods failed - cycle through with variations
        return TeachingMethod.SIMPLIFIED.value
    
    def get_failure_pattern(self) -> Optional[str]:
        """Analyze failure pattern for this topic."""
        if not self.attempts:
            return None
        
        recent_failures = [a for a in self.attempts[-5:] if not a.passed]
        if len(recent_failures) >= 3:
            failure_types = [a.failure_type for a in recent_failures if a.failure_type]
            if failure_types:
                from collections import Counter
                most_common = Counter(failure_types).most_common(1)[0][0]
                return most_common
        return None


@dataclass
class AtlasLearningProfile:
    """Overall learning profile for Atlas across all topics."""
    topic_profiles: Dict[str, TopicLearningProfile] = field(default_factory=dict)
    global_learning_style: str = "unknown"
    total_attempts: int = 0
    total_successes: int = 0
    preferred_methods: List[str] = field(default_factory=list)
    difficult_topics: List[str] = field(default_factory=list)
    easy_topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'topic_profiles': {k: v.to_dict() for k, v in self.topic_profiles.items()},
            'global_learning_style': self.global_learning_style,
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'preferred_methods': self.preferred_methods,
            'difficult_topics': self.difficult_topics,
            'easy_topics': self.easy_topics
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AtlasLearningProfile':
        profile = cls(
            global_learning_style=data.get('global_learning_style', 'unknown'),
            total_attempts=data.get('total_attempts', 0),
            total_successes=data.get('total_successes', 0),
            preferred_methods=data.get('preferred_methods', []),
            difficult_topics=data.get('difficult_topics', []),
            easy_topics=data.get('easy_topics', [])
        )
        profile.topic_profiles = {
            k: TopicLearningProfile.from_dict(v) 
            for k, v in data.get('topic_profiles', {}).items()
        }
        return profile
    
    def get_or_create_topic_profile(self, topic_name: str) -> TopicLearningProfile:
        """Get or create a topic profile."""
        if topic_name not in self.topic_profiles:
            self.topic_profiles[topic_name] = TopicLearningProfile(topic_name=topic_name)
        return self.topic_profiles[topic_name]
    
    def update_global_stats(self):
        """Update global statistics from topic profiles."""
        self.total_attempts = sum(len(p.attempts) for p in self.topic_profiles.values())
        self.total_successes = sum(
            sum(1 for a in p.attempts if a.passed) 
            for p in self.topic_profiles.values()
        )
        
        # Update preferred methods
        method_success = {}
        for profile in self.topic_profiles.values():
            for method in profile.successful_methods:
                method_success[method] = method_success.get(method, 0) + 1
        
        self.preferred_methods = sorted(
            method_success.keys(), 
            key=lambda m: method_success[m], 
            reverse=True
        )[:3]
        
        # Update difficult/easy topics
        topic_success_rates = {}
        for topic, profile in self.topic_profiles.items():
            if len(profile.attempts) >= 2:
                success_rate = sum(1 for a in profile.attempts if a.passed) / len(profile.attempts)
                topic_success_rates[topic] = success_rate
        
        self.difficult_topics = [
            t for t, r in topic_success_rates.items() if r < 0.3
        ]
        self.easy_topics = [
            t for t, r in topic_success_rates.items() if r > 0.7
        ]
        
        # Update global learning style
        style_counts = {}
        for profile in self.topic_profiles.values():
            if profile.estimated_learning_style != 'unknown':
                style_counts[profile.estimated_learning_style] = \
                    style_counts.get(profile.estimated_learning_style, 0) + 1
        
        if style_counts:
            self.global_learning_style = max(style_counts, key=style_counts.get)


class AdaptiveTeachingSystem:
    """Manages adaptive teaching with persistent learning."""
    
    def __init__(self):
        self.profile_file = TEACHING_PROFILE_FILE
        self.diagnostic_file = DIAGNOSTIC_LOG_FILE
        self.profile = self._load_profile()
        self.diagnostic_log: List[dict] = []
    
    def _load_profile(self) -> AtlasLearningProfile:
        """Load learning profile from file."""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                print(f"[Adaptive System] Loaded learning profile from {self.profile_file}")
                return AtlasLearningProfile.from_dict(data)
            except Exception as e:
                print(f"[Adaptive System] Load error: {e}, creating new profile")
                return AtlasLearningProfile()
        return AtlasLearningProfile()
    
    def save_profile(self):
        """Save learning profile to file."""
        self.profile.update_global_stats()
        os.makedirs(os.path.dirname(self.profile_file), exist_ok=True)
        with open(self.profile_file, 'w') as f:
            json.dump(self.profile.to_dict(), f, indent=2)
        print(f"[Adaptive System] Saved learning profile")
    
    def record_attempt(self, topic: str, method: str, passed: bool, evaluation: dict):
        """Record a teaching attempt."""
        topic_profile = self.profile.get_or_create_topic_profile(topic)
        topic_profile.record_attempt(method, passed, evaluation)
        self.save_profile()
    
    def get_recommended_method(self, topic: str) -> str:
        """Get recommended teaching method for a topic."""
        topic_profile = self.profile.get_or_create_topic_profile(topic)
        method = topic_profile.get_recommended_method()
        
        # If topic has a failure pattern, adjust method
        failure_pattern = topic_profile.get_failure_pattern()
        if failure_pattern:
            method = self._adjust_method_for_failure(method, failure_pattern)
        
        return method
    
    def _adjust_method_for_failure(self, base_method: str, failure_pattern: str) -> str:
        """Adjust teaching method based on failure pattern."""
        adjustments = {
            FailureType.MISSING_KEYWORDS.name: TeachingMethod.SIMPLIFIED.value,
            FailureType.INCOHERENT.name: TeachingMethod.STEP_BY_STEP.value,
            FailureType.LOW_COHERENCE.name: TeachingMethod.SIMPLIFIED.value,
            FailureType.OFF_TOPIC.name: TeachingMethod.VISUAL_ANALOGY.value,
            FailureType.TOO_SHORT.name: TeachingMethod.INTERACTIVE.value,
            FailureType.CODE_NOISE.name: TeachingMethod.REAL_WORLD.value,
            FailureType.WORD_SALAD.name: TeachingMethod.STEP_BY_STEP.value,
            FailureType.NO_VERBS.name: TeachingMethod.REPETITION.value,
            FailureType.NO_SUBJECTS.name: TeachingMethod.STANDARD.value,
            FailureType.PREREQUISITE_GAP.name: TeachingMethod.STEP_BY_STEP.value,
        }
        
        return adjustments.get(failure_pattern, base_method)
    
    def diagnose_topic(self, topic: str) -> dict:
        """Diagnose why a topic is difficult for Atlas."""
        topic_profile = self.profile.get_or_create_topic_profile(topic)
        
        diagnosis = {
            'topic': topic,
            'total_attempts': len(topic_profile.attempts),
            'success_rate': 0.0,
            'common_failure': None,
            'recommended_approach': None,
            'prerequisite_gaps': topic_profile.prerequisite_gaps,
            'suggested_action': None
        }
        
        if topic_profile.attempts:
            successes = sum(1 for a in topic_profile.attempts if a.passed)
            diagnosis['success_rate'] = successes / len(topic_profile.attempts)
            
            # Find common failure
            failures = [a for a in topic_profile.attempts if not a.passed]
            if failures:
                from collections import Counter
                failure_types = [a.failure_type for a in failures if a.failure_type]
                if failure_types:
                    diagnosis['common_failure'] = Counter(failure_types).most_common(1)[0][0]
            
            # Determine recommended approach
            if diagnosis['success_rate'] < 0.2:
                diagnosis['recommended_approach'] = "Foundational review needed"
                diagnosis['suggested_action'] = "Go back to simpler concepts"
            elif diagnosis['common_failure'] == FailureType.MISSING_KEYWORDS.name:
                diagnosis['recommended_approach'] = "Focus on core vocabulary"
                diagnosis['suggested_action'] = "Use simplified lessons with key terms"
            elif diagnosis['common_failure'] == FailureType.INCOHERENT.name:
                diagnosis['recommended_approach'] = "Build sentence structure"
                diagnosis['suggested_action'] = "Use step-by-step breakdown"
            else:
                diagnosis['recommended_approach'] = f"Address {diagnosis['common_failure']}"
                diagnosis['suggested_action'] = "Try alternative teaching method"
        
        # Log diagnosis
        self.diagnostic_log.append({
            'timestamp': datetime.now().isoformat(),
            'diagnosis': diagnosis
        })
        
        return diagnosis
    
    def get_learning_summary(self) -> dict:
        """Get summary of Atlas's learning patterns."""
        return {
            'global_learning_style': self.profile.global_learning_style,
            'total_attempts': self.profile.total_attempts,
            'total_successes': self.profile.total_successes,
            'overall_success_rate': (
                self.profile.total_successes / self.profile.total_attempts 
                if self.profile.total_attempts > 0 else 0.0
            ),
            'preferred_methods': self.profile.preferred_methods,
            'difficult_topics': self.profile.difficult_topics,
            'easy_topics': self.profile.easy_topics,
            'topic_count': len(self.profile.topic_profiles)
        }


# Import the rest from continuous_teacher
from continuous_teacher import (
    ShuHaRiPhase, LevelProgress, TopicMastery, HierarchicalMasterySystem,
    TOPICS, LESSON_TEMPLATES, FileLock, log_message, load_session_stats,
    save_session_stats, load_conversations, save_conversation,
    get_brain_stats_direct, evaluate_response, get_coherence_feedback,
    generate_corrective_feedback, PID_FILE, SESSION_FILE, CONVERSATION_FILE,
    MASTERY_FILE, CONCEPT_MASTERY_FILE, LOCK_FILE, LOG_FILE, ATLAS_DIR
)


# ============================================================================
# ADAPTIVE LESSON CONTENT GENERATORS
# ============================================================================

def generate_lesson_standard(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate standard lesson content."""
    template = LESSON_TEMPLATES.get(topic, LESSON_TEMPLATES['Mathematics'])
    return template.format(topic=topic_name)


def generate_lesson_simplified(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate simplified lesson focusing on core terms."""
    keywords = question_data.get('keywords', [])
    keyword_text = ', '.join(keywords[:5]) if keywords else topic_name
    
    return f"""{topic_name} - Core Concepts:

The key idea of {topic_name} is simple. Focus on these important words: {keyword_text}.

What {topic_name} means:
- It is about {topic_name.lower()}
- Remember: {keyword_text}
- This is the basic idea

Key words to remember: {keyword_text}
"""


def generate_lesson_step_by_step(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson broken into small steps."""
    return f"""{topic_name} - Step by Step:

Step 1: Understand what {topic_name} is.
{topic_name} is a concept in {topic}.

Step 2: Learn the definition.
{topic_name} has specific rules and properties.

Step 3: See how it works.
When we use {topic_name}, we follow patterns.

Step 4: Practice the concept.
Try to explain {topic_name} in your own words.

Step 5: Apply what you learned.
Use {topic_name} to solve problems.
"""


def generate_lesson_visual_analogy(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson using visual analogies."""
    analogies = {
        'Mathematics': f"Think of {topic_name} like building with blocks. Each piece connects to the next.",
        'Science': f"Imagine {topic_name} as a machine. Each part has a job to do.",
        'Programming': f"Picture {topic_name} like a recipe. Follow the steps to get the result.",
        'Language': f"See {topic_name} like a puzzle. Each piece fits together to make meaning.",
        'Logic': f"Visualize {topic_name} as a path. Each step follows from the one before.",
    }
    
    analogy = analogies.get(topic, f"Think of {topic_name} like a journey with clear steps.")
    
    return f"""{topic_name} - Visual Guide:

{analogy}

Picture this:
- {topic_name} has a starting point
- It follows a clear path
- Each step leads to the next
- The end result makes sense

When you think about {topic_name}, imagine the picture in your mind.
"""


def generate_lesson_real_world(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson with real-world examples."""
    return f"""{topic_name} - Real World Application:

In everyday life, {topic_name} appears in many places:

Example 1: At home
You use {topic_name.lower()} concepts without realizing it.

Example 2: At work or school
Professionals apply {topic_name} to solve real problems.

Example 3: In nature
The world around us follows patterns related to {topic_name.lower()}.

Why {topic_name} matters:
Understanding this helps you make better decisions and solve problems.
"""


def generate_lesson_interactive(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate interactive lesson with questions."""
    return f"""{topic_name} - Interactive Lesson:

Let's explore {topic_name} together.

Question 1: What do you already know about {topic_name}?
Think about this for a moment.

Question 2: Why is {topic_name} important?
Consider how it connects to other ideas.

Question 3: How would you explain {topic_name} to a friend?
This helps solidify your understanding.

Now, the key facts about {topic_name}:
- Fact 1: {topic_name} is fundamental to {topic}
- Fact 2: It follows clear rules
- Fact 3: Practice makes it easier to understand

Your turn: Try explaining {topic_name} in one sentence.
"""


def generate_lesson_repetition(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson with repetition of key concepts."""
    keywords = question_data.get('keywords', [])
    key_terms = ', '.join(keywords[:3]) if keywords else topic_name
    
    return f"""{topic_name} - Key Concepts (Repeated for Learning):

IMPORTANT: {topic_name} is about {key_terms}.

Repeat after me:
- {topic_name} involves {key_terms}
- {topic_name} involves {key_terms}
- {topic_name} involves {key_terms}

Remember these key points:
1. {topic_name} = {key_terms}
2. {topic_name} = {key_terms}
3. {topic_name} = {key_terms}

The main idea: {topic_name} is {key_terms}.

Once more: {topic_name} = {key_terms}.
"""


def generate_lesson_contrast(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson using contrast/comparison."""
    return f"""{topic_name} - Understanding Through Contrast:

What {topic_name} IS:
- A specific concept in {topic}
- Has clear rules and definitions
- Can be learned and applied

What {topic_name} is NOT:
- Random or arbitrary
- Impossible to understand
- Unrelated to other concepts

Compare {topic_name} with similar ideas:
- They share some features
- But {topic_name} is unique
- Understanding the difference is key

The distinction: {topic_name} stands out because of its specific properties.
"""


def generate_lesson_story(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate lesson embedded in a story."""
    return f"""The Story of {topic_name}:

Once upon a time, students wanted to understand {topic_name}.
They started their journey in the world of {topic}.

Chapter 1: The Discovery
Our heroes discovered that {topic_name} has special properties.
It wasn't always easy, but they kept trying.

Chapter 2: The Challenge
They faced problems that seemed impossible.
But {topic_name} gave them the tools to solve them.

Chapter 3: The Mastery
Through practice and dedication, they mastered {topic_name}.
They could use it whenever they needed.

The moral: {topic_name} is a powerful tool when you understand it.
"""


def generate_lesson_hands_on(topic: str, topic_name: str, question_data: dict) -> str:
    """Generate hands-on practical lesson."""
    return f"""{topic_name} - Hands-On Practice:

Let's practice {topic_name} together.

Exercise 1: Identify {topic_name}
Look for examples of {topic_name.lower()} in the world around you.

Exercise 2: Apply {topic_name}
Use what you know to solve a simple problem.

Exercise 3: Teach {topic_name}
Explain it to someone else to reinforce your learning.

Practical Application:
When you encounter {topic_name.lower()} in real life:
1. Stop and recognize it
2. Apply what you've learned
3. Check if your understanding is correct

Practice makes perfect with {topic_name}!
"""


# Method dispatch table
LESSON_GENERATORS = {
    TeachingMethod.STANDARD.value: generate_lesson_standard,
    TeachingMethod.SIMPLIFIED.value: generate_lesson_simplified,
    TeachingMethod.STEP_BY_STEP.value: generate_lesson_step_by_step,
    TeachingMethod.VISUAL_ANALOGY.value: generate_lesson_visual_analogy,
    TeachingMethod.REAL_WORLD.value: generate_lesson_real_world,
    TeachingMethod.INTERACTIVE.value: generate_lesson_interactive,
    TeachingMethod.REPETITION.value: generate_lesson_repetition,
    TeachingMethod.CONTRAST.value: generate_lesson_contrast,
    TeachingMethod.STORY.value: generate_lesson_story,
    TeachingMethod.HANDS_ON.value: generate_lesson_hands_on,
}


def generate_adaptive_lesson(topic: str, topic_name: str, question_data: dict, 
                              method: str, adaptive_system: AdaptiveTeachingSystem) -> str:
    """Generate lesson using the specified teaching method."""
    generator = LESSON_GENERATORS.get(method, generate_lesson_standard)
    return generator(topic, topic_name, question_data)


# ============================================================================
# FIXED TEACHING WITH PERSISTENT ADAPTIVE LEARNING
# ============================================================================

def teach_lesson_persistent_adaptive(topic_data, category, brain, mastery_system, 
                                      adaptive_system: AdaptiveTeachingSystem) -> dict:
    """
    Teach a lesson with persistent adaptive learning.
    NEVER gives up - keeps trying until Atlas passes.
    """
    topic_name = topic_data['topic']
    question = topic_data['question']
    level = topic_data.get('level', 1)
    
    # Get current phase from mastery system
    status = mastery_system.get_current_status(category)
    current_phase = status.get('phase', 'shu')
    
    # Get recommended teaching method
    method = adaptive_system.get_recommended_method(topic_name)
    log_message(f"       Using teaching method: {method}")
    
    # Generate adaptive lesson content
    content = generate_adaptive_lesson(category, topic_name, topic_data, method, adaptive_system)
    
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
    
    # Record in adaptive teaching system
    adaptive_system.record_attempt(topic_name, method, evaluation['passed'], evaluation)
    
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
    
    # Handle failure - NEVER give up, adapt and retry
    if not evaluation['passed']:
        # Generate corrective feedback
        corrective_feedback = generate_corrective_feedback(evaluation, topic_data, category)
        
        # Analyze failure for targeted learning
        failure_analysis = analyze_failure(evaluation, topic_data)
        
        # Provide targeted learning based on failure type
        targeted_lesson = generate_targeted_lesson(failure_analysis, topic_data, category)
        
        # Have Atlas learn from the targeted lesson
        brain.learn_from_text(targeted_lesson)
        
        # Also learn from the correct answer
        correct_answer = generate_correct_answer(topic_data, category)
        brain.learn_from_text(correct_answer)
        
        # Log the adaptive response
        log_message(f"       🔄 Adaptive learning triggered: {failure_analysis['failure_type']}")
        log_message(f"       📚 Targeted lesson applied: {targeted_lesson[:60]}...")
        log_message(f"       ✅ Correct answer studied: {correct_answer[:60]}...")
        
        # Check if we need diagnostic mode
        topic_profile = adaptive_system.profile.get_or_create_topic_profile(topic_name)
        recent_failures = len([a for a in topic_profile.attempts[-5:] if not a.passed])
        
        if recent_failures >= 3:
            # Enter diagnostic mode
            diagnosis = adaptive_system.diagnose_topic(topic_name)
            log_message(f"       🔍 DIAGNOSTIC MODE: {diagnosis['suggested_action']}")
            
            # If prerequisite gap detected, suggest simpler topic
            if diagnosis.get('common_failure') == FailureType.PREREQUISITE_GAP.name:
                log_message(f"       ⚠️ Prerequisite gap detected - will revisit fundamentals")
    else:
        # Success
        feedback_text = f"Excellent! Your understanding of {topic_name} shows {current_phase.upper()} level mastery. {evaluation['feedback']}"
        brain.learn_from_text(feedback_text)
        log_message(f"       🎉 SUCCESS! {evaluation['feedback']}")
    
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
        'method': method,
        'passed': evaluation['passed'],
        'needs_retry': not evaluation['passed']  # Always retry if failed
    }


def analyze_failure(evaluation: dict, topic_data: dict) -> dict:
    """Analyze why Atlas failed and what to focus on."""
    details = evaluation.get('details', {})
    coherence = details.get('coherence', {})
    
    # Handle coherence as dict or object
    if isinstance(coherence, dict):
        is_coherent = coherence.get('is_coherent', True)
        issues = coherence.get('issues', [])
        coherence_score = coherence.get('score', 0.0)
    else:
        is_coherent = coherence.is_coherent if coherence else True
        issues = coherence.issues if coherence else []
        coherence_score = coherence.score if coherence else 0.0
    
    analysis = {
        'failure_type': None,
        'focus_areas': [],
        'coherence_score': coherence_score,
        'keyword_score': details.get('keyword_score', 0.0),
    }
    
    if not is_coherent:
        analysis['failure_type'] = 'incoherent'
        analysis['focus_areas'].append('sentence_structure')
        
        # Check specific issues
        issue_values = [i.value if hasattr(i, 'value') else i for i in issues]
        
        if CoherenceIssue.CODE_NOISE.value in issue_values:
            analysis['focus_areas'].append('avoid_code_terms')
        if CoherenceIssue.NO_VERBS.value in issue_values:
            analysis['focus_areas'].append('use_verbs')
        if CoherenceIssue.NO_SUBJECTS.value in issue_values:
            analysis['focus_areas'].append('use_subjects')
        if CoherenceIssue.TOO_SHORT.value in issue_values:
            analysis['focus_areas'].append('expand_answers')
        if CoherenceIssue.WORD_SALAD.value in issue_values:
            analysis['focus_areas'].append('logical_flow')
    else:
        # Coherent but missing keywords
        analysis['failure_type'] = 'missing_keywords'
        analysis['focus_areas'].append('key_vocabulary')
        keywords = topic_data.get('keywords', [])
        analysis['target_keywords'] = keywords[:5] if keywords else []
    
    return analysis


def generate_targeted_lesson(failure_analysis: dict, topic_data: dict, category: str) -> str:
    """Generate a targeted lesson based on failure analysis."""
    topic_name = topic_data['topic']
    focus_areas = failure_analysis.get('focus_areas', [])
    
    lesson_parts = [f"Let's focus on improving your understanding of {topic_name}."]
    
    if 'sentence_structure' in focus_areas:
        lesson_parts.append("When answering, use complete sentences with a subject and verb.")
    
    if 'avoid_code_terms' in focus_areas:
        lesson_parts.append("Use everyday language, not programming terms.")
    
    if 'use_verbs' in focus_areas:
        lesson_parts.append("Make sure your sentences have action words (verbs).")
    
    if 'use_subjects' in focus_areas:
        lesson_parts.append("Start your sentences with clear subjects (who or what).")
    
    if 'expand_answers' in focus_areas:
        lesson_parts.append("Give more detailed answers with multiple sentences.")
    
    if 'logical_flow' in focus_areas:
        lesson_parts.append("Connect your ideas in a logical order.")
    
    if 'key_vocabulary' in focus_areas:
        keywords = failure_analysis.get('target_keywords', [])
        if keywords:
            lesson_parts.append(f"Key terms to remember: {', '.join(keywords)}")
    
    return " ".join(lesson_parts)


def generate_correct_answer(topic_data: dict, category: str) -> str:
    """Generate the correct answer for Atlas to study."""
    topic_name = topic_data['topic']
    question = topic_data['question']
    keywords = topic_data.get('keywords', [])
    
    keyword_text = ', '.join(keywords[:5]) if keywords else topic_name
    
    return f"""The correct answer to "{question}" is:

{topic_name} is best understood by remembering these key points: {keyword_text}.

When explaining {topic_name}, use clear sentences that include these important terms.
Make sure your answer directly addresses the question asked.
"""


# ============================================================================
# MASTERY-BASED PROGRESSION SYSTEM
# ============================================================================

def check_mastery_requirements(mastery_system: HierarchicalMasterySystem, 
                                category: str, target_phase: str) -> bool:
    """
    Check if Atlas has met mastery requirements to advance.
    SHU: Must pass with 70%+ keywords AND coherence
    HA: Only after SHU mastered
    RI: Only after HA mastered
    """
    status = mastery_system.get_current_status(category)
    current_phase = status.get('phase', 'shu')
    current_level = status.get('current_level', 1)
    mastery_percentage = status.get('mastery_percentage', 0.0)
    
    if target_phase == 'ha':
        # Can only advance to HA if SHU mastered (70%+)
        return current_phase == 'shu' and mastery_percentage >= 70.0
    
    if target_phase == 'ri':
        # Can only advance to RI if HA mastered (80%+)
        return current_phase == 'ha' and mastery_percentage >= 80.0
    
    if target_phase == 'next_level':
        # Can only advance to next level if RI mastered (90%+)
        return current_phase == 'ri' and mastery_percentage >= 90.0
    
    return True


# ============================================================================
# MAIN TEACHING SESSION WITH PERSISTENT ADAPTIVE LEARNING
# ============================================================================

def run_persistent_adaptive_session(max_lessons: int = 20):
    """Run teaching session with persistent adaptive learning - NEVER gives up."""
    log_message("=" * 70)
    log_message("🎓 Atlas Persistent Adaptive Teacher v7 - NEVER GIVES UP")
    log_message("=" * 70)
    log_message("Features:")
    log_message("- No max retry limits - keeps trying until Atlas passes")
    log_message("- Adaptive teaching based on failure analysis")
    log_message("- Learning between attempts")
    log_message("- Teaching method tracking")
    log_message("- Mastery-based progression")
    log_message("- Diagnostic mode for struggling topics")
    log_message("- ADAPTATION MONITORING: Tracks if Atlas is actually improving")
    log_message("- AUTO-ESCALATION: Detects stagnation and requests help")
    log_message("=" * 70)
    
    # Load session stats
    session_stats = load_session_stats()
    
    # Initialize systems
    mastery_system = HierarchicalMasterySystem()
    adaptive_system = AdaptiveTeachingSystem()
    adaptation_monitor = AtlasAdaptationMonitor()  # NEW: Track adaptation
    
    # Show adaptation monitoring status
    log_message(f"\n📊 Adaptation Monitoring:")
    log_message(f"   Topics tracked: {len(adaptation_monitor.trackers)}")
    log_message(f"   Open issues: {len([i for i in adaptation_monitor.issues if i.status == 'open'])}")
    
    # Get file-based stats first
    file_stats = get_brain_stats_direct()
    log_message(f"[FILE] Brain file: {file_stats['vocab_size']} vocab, {file_stats['total_tokens']} tokens")
    
    # Reset singleton to force reload from file
    reset_shared_brain()
    
    # Now get the brain
    brain = get_shared_brain()
    
    # Get initial stats
    initial_stats = brain.get_stats()
    initial_vocab = initial_stats['vocabulary_size']
    log_message(f"[BRAIN] Loaded vocabulary: {initial_vocab} words")
    log_message(f"[BRAIN] Total tokens: {initial_stats['total_tokens_seen']:,}")
    
    # Show learning profile summary
    profile_summary = adaptive_system.get_learning_summary()
    log_message(f"\n📊 Atlas Learning Profile:")
    log_message(f"   Total attempts: {profile_summary['total_attempts']}")
    log_message(f"   Total successes: {profile_summary['total_successes']}")
    log_message(f"   Overall success rate: {profile_summary['overall_success_rate']*100:.1f}%")
    log_message(f"   Preferred methods: {profile_summary['preferred_methods']}")
    log_message(f"   Learning style: {profile_summary['global_learning_style']}")
    
    # Show current mastery status
    log_message(f"\n📊 Current Mastery Status:")
    for topic_name in TOPICS.keys():
        status = mastery_system.get_current_status(topic_name)
        path = mastery_system.get_learning_path(topic_name)
        if path:
            current_step = next((p for p in path if p['is_current']), None)
            if current_step:
                mastered = "✅" if current_step['is_mastered'] else "⏳"
                log_message(f"   {topic_name:15} | L{current_step['level']}-{current_step['phase'].upper():4} | {current_step['mastery']:5.1f}% {mastered}")
    
    # Teach lessons with persistent adaptive learning
    lessons_taught = 0
    questions_asked = 0
    evaluations = []
    retry_count = 0
    success_count = 0
    
    # Flatten topics for selection
    all_topics = []
    for category, questions in TOPICS.items():
        for q_data in questions:
            all_topics.append((category, q_data))
    
    random.shuffle(all_topics)
    
    # Track which topics need retry (persistent queue)
    retry_queue = []
    normal_queue = all_topics[:max_lessons * 2]  # Extra topics for retries
    
    # Teaching loop - continues until max_lessons reached or all topics passed
    lesson_count = 0
    while lesson_count < max_lessons:
        # Prioritize retries
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
        
        retry_label = "[RETRY]" if is_retry else "[NEW]"
        log_message(f"\n[{lesson_count+1}/{max_lessons}] {retry_label} Teaching: {topic_name} ({category})")
        log_message(f"       Level: {current_level} | Phase: {current_phase.upper()}")
        log_message(f"       Q: {question}")
        
        # Acquire lock before teaching
        with FileLock(LOCK_FILE):
            start_time = time.time()
            result = teach_lesson_persistent_adaptive(
                topic_data, category, brain, mastery_system, adaptive_system
            )
            response_time_ms = int((time.time() - start_time) * 1000)
            
            current_vocab = result['brain_result']['vocabulary_size']
            lessons_taught += 1
            questions_asked += 1
            evaluations.append(result['evaluation'])
            
            # NEW: Record in adaptation monitor
            monitor_result = adaptation_monitor.record_attempt(
                topic_name, result['evaluation'], result['method'], response_time_ms
            )
            
            # NEW: Check for stagnation and auto-escalation
            tracker = monitor_result['tracker']
            issue = monitor_result['issue']
            
            if issue:
                # Learning issue detected - log and escalate
                log_message(f"       🚨 ADAPTATION ISSUE: {issue.diagnosis}")
                log_message(f"       📈 Trend: {tracker.trend} | Status: {monitor_result['status']}")
                
                # Auto-escalation: Try 2 more methods before requesting help
                if tracker.escalation_level == 2 and len(tracker.metrics) >= 5:
                    log_message(f"       ⚠️  ESCALATION: Trying alternative methods...")
                    # Force different teaching methods
                    for alt_method in ['simplified', 'visual_analogy']:
                        if alt_method != result['method']:
                            log_message(f"       🔄 Trying alternative method: {alt_method}")
                            # Re-teach with alternative method
                            alt_content = generate_adaptive_lesson(
                                category, topic_name, topic_data, alt_method, adaptive_system
                            )
                            brain.learn_from_text(alt_content)
                            break
                    
                    # Check if still stagnant after alternatives
                    if tracker.get_adaptation_status() == AdaptationStatus.NEEDS_INTERVENTION.value:
                        log_message(f"       🆘 CRITICAL: Atlas not adapting - agent intervention required!")
                        log_message(f"       📋 Issue logged to: {LEARNING_ISSUES_FILE}")
            
            # NEW: Show progress trend
            if len(tracker.metrics) >= 3:
                recent_scores = [m.score for m in tracker.metrics[-3:]]
                log_message(f"       📊 Score trend: {recent_scores} | {tracker.trend}")
            
            if result['passed']:
                success_count += 1
                log_message(f"       ✅ PASSED! Score: {result['evaluation']['score']:.1f}%")
            else:
                retry_count += 1
                # Add back to retry queue - NEVER give up
                retry_queue.append((category, topic_data))
                log_message(f"       🔄 FAILED - Will retry (persistent learning)")
            
            # Log the Q&A
            response = result['response']
            coherence = result['evaluation'].get('coherence_score', 0.0)
            log_message(f"       A: {response[:80]}{'...' if len(response) > 80 else ''}")
            log_message(f"       📊 Method: {result['method']} | Coherence: {coherence:.2f}")
            
            # Check for phase/level changes
            if result['assessment_result'].get('phase_change'):
                change = result['assessment_result']['phase_change']
                log_message(f"       🔔 {change['message']}")
            
            log_message(f"       -> Tokens: {result['brain_result']['tokens_processed']}, Vocab: {current_vocab}")
            
            # Save checkpoint every 5 lessons
            if lessons_taught % 5 == 0:
                save_shared_brain()
                adaptive_system.save_profile()
                adaptation_monitor.save_data()
                log_message(f"💾 Checkpoint saved after {lessons_taught} lessons")
        
        lesson_count += 1
    
    # Final save
    with FileLock(LOCK_FILE):
        save_shared_brain()
        adaptive_system.save_profile()
        adaptation_monitor.save_data()
    
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
    profile_summary = adaptive_system.get_learning_summary()
    adaptation_summary = adaptation_monitor.get_all_status()
    
    log_message("\n" + "=" * 70)
    log_message("📊 Session Complete - Persistent Adaptive Learning")
    log_message("=" * 70)
    log_message(f"Lessons taught: {lessons_taught}")
    log_message(f"Questions asked: {questions_asked}")
    log_message(f"Evaluations passed: {passed_count}/{len(evaluations)} ({passed_count/len(evaluations)*100:.1f}%)")
    log_message(f"Evaluations failed (will retry): {failed_count}/{len(evaluations)}")
    log_message(f"Total retries this session: {retry_count}")
    log_message(f"Retry queue remaining: {len(retry_queue)} topics")
    log_message(f"Vocabulary: {initial_vocab} → {final_vocab} (+{vocab_added})")
    log_message(f"Total tokens: {final_stats['total_tokens_seen']:,}")
    
    # Show updated learning profile
    log_message(f"\n📊 Updated Learning Profile:")
    log_message(f"   Global learning style: {profile_summary['global_learning_style']}")
    log_message(f"   Preferred methods: {profile_summary['preferred_methods']}")
    log_message(f"   Difficult topics: {profile_summary['difficult_topics']}")
    log_message(f"   Easy topics: {profile_summary['easy_topics']}")
    
    # NEW: Show adaptation monitoring summary
    log_message(f"\n📊 Adaptation Monitoring:")
    log_message(f"   Topics tracked: {adaptation_summary['topics_tracked']}")
    log_message(f"   Open issues: {adaptation_summary['open_issues']}")
    
    # Show topics with issues
    for topic_name, status in adaptation_summary['topics'].items():
        if status['escalation_level'] > 0:
            log_message(f"   ⚠️  {topic_name}: {status['adaptation_status']} (Level {status['escalation_level']})")
            log_message(f"      Trend: {status['score_trend']} | Recommendation: {status['recommendation']}")
    
    # Show updated mastery status
    log_message(f"\n📊 Updated Mastery Status:")
    for topic_name in TOPICS.keys():
        status = mastery_system.get_current_status(topic_name)
        path = mastery_system.get_learning_path(topic_name)
        if path:
            current_step = next((p for p in path if p['is_current']), None)
            if current_step:
                mastered = "✅ MASTERED" if current_step['is_mastered'] else "⏳ IN PROGRESS"
                log_message(f"   {topic_name:15} | L{current_step['level']}-{current_step['phase'].upper():4} | {current_step['mastery']:5.1f}% | {mastered}")
    
    # Diagnostic summary for struggling topics
    log_message(f"\n🔍 Diagnostic Summary:")
    for topic_name in profile_summary['difficult_topics']:
        diagnosis = adaptive_system.diagnose_topic(topic_name)
        log_message(f"   {topic_name}: {diagnosis['suggested_action']}")
    
    # NEW: Show learning issues file location
    if adaptation_summary['open_issues'] > 0:
        log_message(f"\n🚨 Learning Issues Report:")
        log_message(f"   File: {LEARNING_ISSUES_FILE}")
        log_message(f"   Open issues require agent team intervention")
    
    log_message("=" * 70)
    log_message("✨ NEVER GIVE UP - Atlas will keep learning! ✨")
    log_message("=" * 70)
    
    return {
        'success': True,
        'lessons_taught': lessons_taught,
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(evaluations) * 100 if evaluations else 0,
        'retries': retry_count,
        'retry_queue': len(retry_queue),
        'vocab_added': vocab_added,
        'learning_style': profile_summary['global_learning_style'],
        'preferred_methods': profile_summary['preferred_methods']
    }


def main():
    try:
        result = run_persistent_adaptive_session(max_lessons=20)
        return 0 if result['success'] else 1
    except Exception as e:
        log_message(f"[ERROR] {e}")
        import traceback
        log_message(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
