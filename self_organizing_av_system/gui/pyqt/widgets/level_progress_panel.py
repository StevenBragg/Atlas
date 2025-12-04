"""
Level Progress Panel - Shows Atlas's progression through curriculum levels.

Displays visual progress through all 5 levels with unlock status.
"""

import logging
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QProgressBar
    )
    from PyQt5.QtCore import Qt
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QProgressBar
    )
    from PyQt6.QtCore import Qt
    PYQT_VERSION = 6

from self_organizing_av_system.core.curriculum_system import CurriculumLevel

logger = logging.getLogger(__name__)


class LevelBadge(QFrame):
    """Individual level badge showing progress."""

    def __init__(self, level: CurriculumLevel, name: str, parent=None):
        super().__init__(parent)
        self.level = level
        self.name = name
        self.unlocked = False
        self.completed = False
        self.progress = 0.0
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedSize(120, 100)
        self.setStyleSheet(self._get_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)

        # Level number
        self.level_label = QLabel(f"Level {self.level.value}")
        self.level_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.level_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.level_label)

        # Level name
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("font-size: 10px;")
        self.name_label.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(0,0,0,0.3);
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

    def _get_style(self) -> str:
        """Get stylesheet based on state."""
        if self.completed:
            return """
                LevelBadge {
                    background-color: #1b5e20;
                    border: 2px solid #4caf50;
                    border-radius: 10px;
                    color: white;
                }
            """
        elif self.unlocked:
            return """
                LevelBadge {
                    background-color: #1565c0;
                    border: 2px solid #42a5f5;
                    border-radius: 10px;
                    color: white;
                }
            """
        else:
            return """
                LevelBadge {
                    background-color: #424242;
                    border: 2px solid #616161;
                    border-radius: 10px;
                    color: #888;
                }
            """

    def set_state(self, unlocked: bool, completed: bool, progress: float):
        """Update the badge state."""
        self.unlocked = unlocked
        self.completed = completed
        self.progress = progress

        self.setStyleSheet(self._get_style())
        self.progress_bar.setValue(int(progress * 100))

        if completed:
            self.level_label.setText(f"✓ Level {self.level.value}")
        elif not unlocked:
            self.level_label.setText(f"🔒 Level {self.level.value}")
        else:
            self.level_label.setText(f"Level {self.level.value}")


class LevelProgressPanel(QWidget):
    """
    Level Progress Panel - Shows progression through all curriculum levels.

    Displays 5 level badges with:
    - Locked/Unlocked/Completed states
    - Progress percentage within each level
    - Visual path connecting levels
    """

    def __init__(self, controller, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self.level_badges: dict[CurriculumLevel, LevelBadge] = {}
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Curriculum Progress")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffb74d;")
        title.setAlignment(Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Level badges row
        badges_row = QHBoxLayout()
        badges_row.setSpacing(10)

        for level in CurriculumLevel:
            level_info = self.controller.curriculum.CURRICULUM.get(level)
            name = level_info.name if level_info else f"Level {level.value}"

            badge = LevelBadge(level, name)
            badges_row.addWidget(badge)
            self.level_badges[level] = badge

            # Add arrow between badges (except last)
            if level.value < 5:
                arrow = QLabel("→")
                arrow.setStyleSheet("font-size: 20px; color: #666;")
                badges_row.addWidget(arrow)

        layout.addLayout(badges_row)

        # Overall stats
        stats_row = QHBoxLayout()

        self.total_challenges_label = QLabel("Challenges: 0/0")
        self.total_challenges_label.setStyleSheet("font-size: 12px; color: #888;")
        stats_row.addWidget(self.total_challenges_label)

        stats_row.addStretch()

        self.overall_accuracy_label = QLabel("Overall Accuracy: --")
        self.overall_accuracy_label.setStyleSheet("font-size: 12px; color: #81c784;")
        stats_row.addWidget(self.overall_accuracy_label)

        layout.addLayout(stats_row)

    def refresh(self):
        """Refresh the progress display."""
        curriculum = self.controller.curriculum

        total_challenges = 0
        completed_challenges = 0

        for level in CurriculumLevel:
            level_info = curriculum.CURRICULUM.get(level)
            if not level_info:
                continue

            unlocked = curriculum.is_level_unlocked(level)
            progress = curriculum.get_level_progress(level)

            # Count challenges
            num_challenges = len(level_info.challenges)
            total_challenges += num_challenges

            # Check completed (progress >= unlock_threshold)
            completed = progress >= level_info.unlock_threshold
            if completed:
                completed_challenges += num_challenges
            else:
                completed_challenges += int(progress * num_challenges)

            # Update badge
            badge = self.level_badges.get(level)
            if badge:
                badge.set_state(unlocked, completed, progress)

        # Update totals
        self.total_challenges_label.setText(
            f"Challenges: {completed_challenges}/{total_challenges}"
        )

        # Get overall accuracy from controller
        stats = self.controller.get_stats()
        if "current_accuracy" in stats:
            self.overall_accuracy_label.setText(
                f"Overall Accuracy: {stats['current_accuracy']:.1%}"
            )
