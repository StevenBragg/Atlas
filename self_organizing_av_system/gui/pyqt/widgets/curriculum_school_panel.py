"""
Curriculum School Panel - Structured Challenge Interface

Displays curriculum challenges organized by level (like grades in school).
Atlas progresses through levels by mastering challenges.
"""

import logging
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QPushButton, QComboBox, QProgressBar, QScrollArea,
        QGridLayout, QGroupBox, QTextEdit
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QPushButton, QComboBox, QProgressBar, QScrollArea,
        QGridLayout, QGroupBox, QTextEdit
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    PYQT_VERSION = 6

from self_organizing_av_system.core.curriculum_system import CurriculumLevel

logger = logging.getLogger(__name__)


class ChallengeCard(QFrame):
    """Individual challenge display card."""

    startClicked = pyqtSignal(int)  # Challenge index

    def __init__(self, index: int, challenge: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.challenge = challenge
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box if PYQT_VERSION == 5 else QFrame.Shape.Box)
        self.setStyleSheet("""
            ChallengeCard {
                background-color: #2d2d30;
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                padding: 10px;
            }
            ChallengeCard:hover {
                border-color: #4fc3f7;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Challenge name
        name_label = QLabel(self.challenge["name"])
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

        # Modalities
        modalities = self.challenge.get("modalities", [])
        mod_text = ", ".join([m.name for m in modalities])
        mod_label = QLabel(f"Modalities: {mod_text}")
        mod_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(mod_label)

        # Target accuracy
        target = self.challenge.get("target_accuracy", 0)
        target_label = QLabel(f"Target: {target:.0%} accuracy")
        target_label.setStyleSheet("font-size: 11px; color: #81c784;")
        layout.addWidget(target_label)

        # Progress bar (current accuracy)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Start button
        self.start_btn = QPushButton("Start Challenge")
        self.start_btn.clicked.connect(lambda: self.startClicked.emit(self.index))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        layout.addWidget(self.start_btn)

    def set_accuracy(self, accuracy: float):
        """Update the current accuracy display."""
        self.progress_bar.setValue(int(accuracy * 100))

        # Update color based on target
        target = self.challenge.get("target_accuracy", 0)
        if accuracy >= target:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #1a1a1a;
                    border: 1px solid #444;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #4caf50;
                    border-radius: 2px;
                }
            """)
            self.start_btn.setText("Completed ✓")
        else:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #1a1a1a;
                    border: 1px solid #444;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #0e639c;
                    border-radius: 2px;
                }
            """)

    def set_locked(self, locked: bool):
        """Set whether this challenge is locked."""
        self.start_btn.setEnabled(not locked)
        if locked:
            self.start_btn.setText("Locked")
            self.setStyleSheet("""
                ChallengeCard {
                    background-color: #1a1a1a;
                    border: 1px solid #333;
                    border-radius: 5px;
                    padding: 10px;
                }
            """)


class CurriculumSchoolPanel(QWidget):
    """
    Curriculum School Panel - Main interface for structured learning.

    Shows:
    - Level selector (1-5)
    - Challenge cards for current level
    - Progress through curriculum
    - Learning metrics display
    """

    challengeStarted = pyqtSignal(int, int)  # level, challenge_index

    def __init__(self, controller, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self.challenge_cards: list[ChallengeCard] = []
        self._setup_ui()
        self._load_level(CurriculumLevel.LEVEL_1_BASIC)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Level selector
        level_row = self._create_level_selector()
        layout.addLayout(level_row)

        # Challenge grid (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.challenges_container = QWidget()
        self.challenges_layout = QGridLayout(self.challenges_container)
        self.challenges_layout.setSpacing(10)
        scroll.setWidget(self.challenges_container)

        layout.addWidget(scroll, stretch=1)

        # Current training section
        training_section = self._create_training_section()
        layout.addWidget(training_section)

    def _create_header(self) -> QWidget:
        """Create the header section."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1e3a5f;
                border-radius: 5px;
                padding: 10px;
            }
        """)

        layout = QHBoxLayout(header)

        title = QLabel("Atlas Curriculum Learning")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(title)

        layout.addStretch()

        # Overall progress
        self.overall_progress = QLabel("Progress: Level 1")
        self.overall_progress.setStyleSheet("font-size: 14px; color: #81c784;")
        layout.addWidget(self.overall_progress)

        return header

    def _create_level_selector(self) -> QHBoxLayout:
        """Create the level selector row."""
        row = QHBoxLayout()

        label = QLabel("Select Level:")
        label.setStyleSheet("font-size: 14px; color: #888;")
        row.addWidget(label)

        self.level_combo = QComboBox()
        self.level_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d30;
                color: white;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #4fc3f7;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)

        # Add levels
        for level in CurriculumLevel:
            level_info = self.controller.curriculum.CURRICULUM.get(level)
            if level_info:
                self.level_combo.addItem(f"Level {level.value}: {level_info.name}", level)

        self.level_combo.currentIndexChanged.connect(self._on_level_changed)
        row.addWidget(self.level_combo)

        row.addStretch()

        # Level status
        self.level_status = QLabel("Status: In Progress")
        self.level_status.setStyleSheet("font-size: 12px; color: #ffb74d;")
        row.addWidget(self.level_status)

        return row

    def _create_training_section(self) -> QWidget:
        """Create the current training display section."""
        section = QGroupBox("Current Training")
        section.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #ce93d8;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(section)

        # Training info row
        info_row = QHBoxLayout()

        self.current_challenge_label = QLabel("No challenge active")
        self.current_challenge_label.setStyleSheet("font-size: 13px; color: #ffffff;")
        info_row.addWidget(self.current_challenge_label)

        info_row.addStretch()

        self.epoch_label = QLabel("Epoch: --")
        self.epoch_label.setStyleSheet("font-size: 12px; color: #888;")
        info_row.addWidget(self.epoch_label)

        self.accuracy_label = QLabel("Accuracy: --")
        self.accuracy_label.setStyleSheet("font-size: 12px; color: #81c784;")
        info_row.addWidget(self.accuracy_label)

        layout.addLayout(info_row)

        # Training progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(True)
        self.training_progress.setFormat("Training: %p%")
        self.training_progress.setFixedHeight(25)
        self.training_progress.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1565c0, stop:1 #4fc3f7
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.training_progress)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #aaa;
                border: 1px solid #333;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_output)

        return section

    def _on_level_changed(self, index: int):
        """Handle level selection change."""
        level = self.level_combo.itemData(index)
        if level:
            self._load_level(level)

    def _load_level(self, level: CurriculumLevel):
        """Load challenges for a level."""
        # Clear existing cards
        for card in self.challenge_cards:
            card.deleteLater()
        self.challenge_cards.clear()

        # Get level info
        level_info = self.controller.curriculum.CURRICULUM.get(level)
        if not level_info:
            return

        # Update status
        if self.controller.curriculum.is_level_unlocked(level):
            self.level_status.setText("Status: Unlocked")
            self.level_status.setStyleSheet("font-size: 12px; color: #81c784;")
        else:
            self.level_status.setText("Status: Locked")
            self.level_status.setStyleSheet("font-size: 12px; color: #f44336;")

        # Create challenge cards
        for i, challenge in enumerate(level_info.challenges):
            card = ChallengeCard(i, challenge)
            card.startClicked.connect(lambda idx, lv=level: self._start_challenge(lv, idx))

            # Check if locked
            if not self.controller.curriculum.is_level_unlocked(level):
                card.set_locked(True)

            # Add to grid (2 columns)
            row = i // 2
            col = i % 2
            self.challenges_layout.addWidget(card, row, col)
            self.challenge_cards.append(card)

    def _start_challenge(self, level: CurriculumLevel, challenge_index: int):
        """Start a curriculum challenge."""
        logger.info(f"Starting challenge {challenge_index} at level {level}")

        # Update UI
        level_info = self.controller.curriculum.CURRICULUM.get(level)
        if level_info and challenge_index < len(level_info.challenges):
            challenge = level_info.challenges[challenge_index]
            self.current_challenge_label.setText(f"Training: {challenge['name']}")
            self.training_progress.setValue(0)
            self.log_output.append(f"Starting: {challenge['name']}")

        # Start via controller
        self.controller.start_curriculum_challenge(level, challenge_index)

        # Emit signal
        self.challengeStarted.emit(level.value, challenge_index)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update display with training metrics."""
        # Update challenge name if provided (for auto-learning)
        if "challenge" in metrics:
            self.current_challenge_label.setText(f"Training: {metrics['challenge']}")

        if "epoch" in metrics:
            self.epoch_label.setText(f"Epoch: {metrics['epoch']}")

        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            self.accuracy_label.setText(f"Accuracy: {acc:.1%}")
            self.training_progress.setValue(int(acc * 100))

            # Also update the current challenge card's progress if we can find it
            challenge_name = metrics.get("challenge", "")
            for card in self.challenge_cards:
                if card.challenge.get("name") == challenge_name:
                    card.set_accuracy(acc)
                    break

        if "message" in metrics:
            self.log_output.append(metrics["message"])

        if "level_up" in metrics:
            self.log_output.append(f"🎉 {metrics.get('message', 'Level up!')}")
            # Refresh level display
            level = self.level_combo.currentData()
            if level:
                self._load_level(level)

    def on_challenge_complete(self, result: Dict[str, Any]):
        """Handle challenge completion."""
        accuracy = result.get("accuracy", 0)
        passed = result.get("passed", False)

        if passed:
            self.log_output.append(f"✓ Challenge PASSED with {accuracy:.1%} accuracy!")
            self.current_challenge_label.setText("Challenge Complete!")
        else:
            self.log_output.append(f"✗ Challenge not passed ({accuracy:.1%})")
            self.current_challenge_label.setText("Try again...")

        # Refresh the level display
        level = self.level_combo.currentData()
        if level:
            self._load_level(level)

        # Update overall progress
        current_level = self.controller.curriculum.current_level
        self.overall_progress.setText(f"Progress: Level {current_level.value}")
