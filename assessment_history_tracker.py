#!/usr/bin/env python3
"""
Atlas Assessment History Tracker

Tracks all assessment attempts with timestamps, grades, and pass/fail status.
Provides historical report card generation.

Storage: teacher_state/assessment_history.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Constants
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
TEACHER_STATE_DIR = ATLAS_DIR / 'teacher_state'
ASSESSMENT_HISTORY_FILE = TEACHER_STATE_DIR / 'assessment_history.json'


@dataclass
class AssessmentEntry:
    """Single assessment attempt record."""
    topic: str
    level: int
    phase: str  # shu, ha, or ri
    attempt_number: int
    score: float  # 0-100
    passed: bool
    timestamp: str
    question: str
    response: str
    feedback: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AssessmentEntry':
        return cls(**data)


class AssessmentHistoryTracker:
    """
    Manages historical tracking of all assessments.
    """
    
    def __init__(self, history_file: str = None):
        self.history_file = history_file or str(ASSESSMENT_HISTORY_FILE)
        self.assessments: List[AssessmentEntry] = []
        self._load_history()
    
    def _load_history(self):
        """Load assessment history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                self.assessments = [
                    AssessmentEntry.from_dict(entry) 
                    for entry in data.get('assessments', [])
                ]
                print(f"[Assessment History] Loaded {len(self.assessments)} assessments from {self.history_file}")
            except Exception as e:
                print(f"[Assessment History] Load error: {e}, initializing fresh")
                self.assessments = []
        else:
            print(f"[Assessment History] No history file found, initializing fresh")
            self.assessments = []
    
    def save_history(self):
        """Save assessment history to file."""
        data = {
            'assessments': [entry.to_dict() for entry in self.assessments],
            'version': '1.0',
            'created_at': self.assessments[0].timestamp if self.assessments else datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_assessments': len(self.assessments)
        }
        
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Assessment History] Saved {len(self.assessments)} assessments to {self.history_file}")
    
    def log_assessment(self, topic: str, level: int, phase: str, 
                       score: float, passed: bool, question: str, 
                       response: str, feedback: str = "") -> AssessmentEntry:
        """
        Log a new assessment attempt.
        
        Args:
            topic: The topic/category being assessed
            level: The complexity level (1, 2, or 3)
            phase: The Shu-Ha-Ri phase (shu, ha, or ri)
            score: The assessment score (0-100)
            passed: Whether the assessment was passed
            question: The question asked
            response: The response given
            feedback: Optional feedback text
            
        Returns:
            The created AssessmentEntry
        """
        # Calculate attempt number for this topic/level/phase combination
        attempt_number = self._get_next_attempt_number(topic, level, phase)
        
        entry = AssessmentEntry(
            topic=topic,
            level=level,
            phase=phase.lower(),
            attempt_number=attempt_number,
            score=score,
            passed=passed,
            timestamp=datetime.now().isoformat(),
            question=question,
            response=response,
            feedback=feedback
        )
        
        self.assessments.append(entry)
        self.save_history()
        
        status = "PASS" if passed else "FAIL"
        print(f"[Assessment History] Logged: {topic} L{level}-{phase.upper()} Attempt {attempt_number}: {score:.1f}% ({status})")
        
        return entry
    
    def _get_next_attempt_number(self, topic: str, level: int, phase: str) -> int:
        """Get the next attempt number for a specific topic/level/phase."""
        matching = [
            a for a in self.assessments 
            if a.topic == topic and a.level == level and a.phase == phase.lower()
        ]
        return len(matching) + 1
    
    def get_assessments_for_topic(self, topic: str) -> List[AssessmentEntry]:
        """Get all assessments for a specific topic."""
        return [a for a in self.assessments if a.topic == topic]
    
    def get_assessments_for_level(self, topic: str, level: int) -> List[AssessmentEntry]:
        """Get all assessments for a specific topic and level."""
        return [a for a in self.assessments if a.topic == topic and a.level == level]
    
    def get_assessments_for_phase(self, topic: str, level: int, phase: str) -> List[AssessmentEntry]:
        """Get all assessments for a specific topic, level, and phase."""
        return [
            a for a in self.assessments 
            if a.topic == topic and a.level == level and a.phase == phase.lower()
        ]
    
    def get_recent_assessments(self, count: int = 5) -> List[AssessmentEntry]:
        """Get the most recent assessments."""
        return sorted(self.assessments, key=lambda a: a.timestamp, reverse=True)[:count]
    
    def get_current_phase_for_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get the current phase and level for a topic based on latest assessment."""
        topic_assessments = self.get_assessments_for_topic(topic)
        if not topic_assessments:
            return None
        
        latest = max(topic_assessments, key=lambda a: a.timestamp)
        return {
            'topic': topic,
            'level': latest.level,
            'phase': latest.phase,
            'last_assessment': latest.timestamp,
            'last_score': latest.score,
            'last_passed': latest.passed
        }
    
    def get_all_topics(self) -> List[str]:
        """Get all unique topics that have assessments."""
        return sorted(list(set(a.topic for a in self.assessments)))
    
    def generate_report_card(self, topic: str = None) -> str:
        """
        Generate a formatted report card.
        
        Args:
            topic: Optional topic to filter by. If None, generates report for all topics.
            
        Returns:
            Formatted report card string
        """
        if topic:
            topics = [topic] if topic in self.get_all_topics() else []
        else:
            topics = self.get_all_topics()
        
        if not topics:
            return "No assessment history found."
        
        lines = []
        lines.append("=" * 70)
        lines.append("📊 ATLAS HISTORICAL REPORT CARD")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Assessments: {len(self.assessments)}")
        lines.append("")
        
        for t in topics:
            lines.append(self._generate_topic_report(t))
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _generate_topic_report(self, topic: str) -> str:
        """Generate report card section for a single topic."""
        topic_assessments = self.get_assessments_for_topic(topic)
        
        if not topic_assessments:
            return f"Topic: {topic}\n  No assessments recorded."
        
        lines = []
        lines.append(f"Topic: {topic}")
        
        # Group by level
        levels = sorted(list(set(a.level for a in topic_assessments)))
        
        for level in levels:
            lines.append(f"  Level {level}:")
            
            # Group by phase within level
            level_assessments = [a for a in topic_assessments if a.level == level]
            phases = ['shu', 'ha', 'ri']
            
            for phase in phases:
                phase_assessments = [a for a in level_assessments if a.phase == phase]
                if phase_assessments:
                    lines.append(f"    {phase.upper()} Phase:")
                    
                    # Sort by attempt number
                    for assessment in sorted(phase_assessments, key=lambda a: a.attempt_number):
                        status = "PASS" if assessment.passed else "FAIL"
                        ts = datetime.fromisoformat(assessment.timestamp).strftime('%Y-%m-%d %H:%M')
                        lines.append(f"      - Attempt {assessment.attempt_number}: {assessment.score:.0f}% ({status}) - {ts}")
        
        return "\n".join(lines)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all assessments."""
        if not self.assessments:
            return {
                'total_assessments': 0,
                'total_passed': 0,
                'total_failed': 0,
                'pass_rate': 0.0,
                'average_score': 0.0,
                'topics': []
            }
        
        total = len(self.assessments)
        passed = sum(1 for a in self.assessments if a.passed)
        failed = total - passed
        pass_rate = (passed / total) * 100 if total > 0 else 0.0
        avg_score = sum(a.score for a in self.assessments) / total if total > 0 else 0.0
        
        return {
            'total_assessments': total,
            'total_passed': passed,
            'total_failed': failed,
            'pass_rate': pass_rate,
            'average_score': avg_score,
            'topics': self.get_all_topics()
        }


# Global tracker instance
the_tracker: Optional[AssessmentHistoryTracker] = None


def get_tracker() -> AssessmentHistoryTracker:
    """Get the global assessment history tracker instance."""
    global the_tracker
    if the_tracker is None:
        the_tracker = AssessmentHistoryTracker()
    return the_tracker


def log_assessment(topic: str, level: int, phase: str, 
                   score: float, passed: bool, question: str, 
                   response: str, feedback: str = "") -> AssessmentEntry:
    """
    Convenience function to log an assessment.
    
    Args:
        topic: The topic/category being assessed
        level: The complexity level (1, 2, or 3)
        phase: The Shu-Ha-Ri phase (shu, ha, or ri)
        score: The assessment score (0-100)
        passed: Whether the assessment was passed
        question: The question asked
        response: The response given
        feedback: Optional feedback text
        
    Returns:
        The created AssessmentEntry
    """
    tracker = get_tracker()
    return tracker.log_assessment(topic, level, phase, score, passed, question, response, feedback)


def get_recent_assessments(count: int = 5) -> List[AssessmentEntry]:
    """Get the most recent assessments."""
    tracker = get_tracker()
    return tracker.get_recent_assessments(count)


def generate_report_card(topic: str = None) -> str:
    """Generate a formatted report card."""
    tracker = get_tracker()
    return tracker.generate_report_card(topic)


if __name__ == "__main__":
    # Test the tracker
    tracker = AssessmentHistoryTracker()
    
    # Print current stats
    stats = tracker.get_summary_stats()
    print(f"\nTotal Assessments: {stats['total_assessments']}")
    print(f"Pass Rate: {stats['pass_rate']:.1f}%")
    print(f"Average Score: {stats['average_score']:.1f}%")
    
    # Generate and print report card
    print("\n" + tracker.generate_report_card())
