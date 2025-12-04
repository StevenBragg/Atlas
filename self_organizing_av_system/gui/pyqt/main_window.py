"""
Atlas Main Window - Tabbed Layout with Curriculum and Free Play

Main application window with:
- Curriculum Learning Tab: Structured school-like challenges
- Free Play Tab: Webcam, mic, chat, creative canvas
- Shared widgets: Neural network visualization, knowledge base display
"""

import logging
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QSplitter, QStatusBar, QMenuBar, QAction,
        QLabel, QFrame, QProgressBar
    )
    from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
    from PyQt5.QtGui import QFont
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QSplitter, QStatusBar, QMenuBar,
        QLabel, QFrame, QProgressBar
    )
    from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
    from PyQt6.QtGui import QFont, QAction
    PYQT_VERSION = 6

from .controllers.atlas_controller import AtlasController
from .widgets.curriculum_school_panel import CurriculumSchoolPanel
from .widgets.level_progress_panel import LevelProgressPanel
from .widgets.chat_panel import ChatPanel
from .widgets.creative_canvas import CreativeCanvasWidget
from .widgets.input_panel import InputPanel
from .widgets.network_viz_panel import NetworkVizPanel
from .widgets.knowledge_base_panel import KnowledgeBasePanel
from .threads.capture_thread import CaptureThread
from .threads.canvas_thread import CanvasThread
from .threads.processing_thread import ProcessingThread

logger = logging.getLogger(__name__)


class SignalBridge(QObject):
    """
    Thread-safe signal bridge for communicating between
    background threads and the Qt UI thread.
    """
    progressUpdated = pyqtSignal(dict)
    learningComplete = pyqtSignal(dict)
    networkUpdated = pyqtSignal(dict)


class AtlasMainWindow(QMainWindow):
    """
    Main application window for Atlas.

    Layout:
    - Tab Widget with two tabs:
      1. Curriculum Learning: School panel + Level progress
      2. Free Play: Chat + Canvas + Webcam/Mic
    - Bottom: Knowledge base panel (shared)
    - Status bar: GPU info, FPS, processing status
    """

    def __init__(self, controller: AtlasController, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.controller = controller

        # Create signal bridge for thread-safe UI updates
        self.signal_bridge = SignalBridge()

        # Set window properties
        self.setWindowTitle("Atlas Challenge-Based Learning")
        self.setMinimumSize(1400, 900)

        # Initialize threads
        self._init_threads()

        # Build UI
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Connect signals
        self._connect_signals()

        # Initial network visualization update
        initial_network_state = self.controller.get_network_state()
        self.network_viz_curriculum.update_network(initial_network_state)
        self.network_viz_free.update_network(initial_network_state)

        # Start update timer
        self._start_update_timer()

        # Start auto curriculum learning after a brief delay
        self._start_auto_learning_timer()

        logger.info("AtlasMainWindow initialized")

    def _init_threads(self):
        """Initialize background threads."""
        # AV Capture thread
        self.capture_thread = CaptureThread()
        self.capture_thread.start()

        # Canvas generation thread
        self.canvas_thread = CanvasThread(self.controller.canvas_controller)
        self.canvas_thread.start()

        # Processing thread
        self.processing_thread = ProcessingThread(self.controller)
        self.processing_thread.start()

    def _setup_menu_bar(self):
        """Setup the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        save_action = QAction("&Save Checkpoint", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_checkpoint)
        file_menu.addAction(save_action)

        load_action = QAction("&Load Checkpoint", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_checkpoint)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        reset_action = QAction("&Reset Learning (Fresh Start)", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self._on_reset_learning)
        file_menu.addAction(reset_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_central_widget(self):
        """Setup the central widget with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Create main splitter (tabs + bottom panel)
        main_splitter = QSplitter(Qt.Vertical if PYQT_VERSION == 5 else Qt.Orientation.Vertical)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        # Tab 1: Curriculum Learning
        curriculum_tab = self._create_curriculum_tab()
        self.tab_widget.addTab(curriculum_tab, "Curriculum Learning")

        # Tab 2: Free Play
        free_play_tab = self._create_free_play_tab()
        self.tab_widget.addTab(free_play_tab, "Free Play")

        main_splitter.addWidget(self.tab_widget)

        # Bottom panel: Knowledge base (shared between tabs)
        bottom_panel = self._create_bottom_panel()
        main_splitter.addWidget(bottom_panel)

        # Set splitter proportions
        main_splitter.setSizes([700, 150])

        main_layout.addWidget(main_splitter)

    def _create_curriculum_tab(self) -> QWidget:
        """Create the Curriculum Learning tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left: School panel
        self.curriculum_panel = CurriculumSchoolPanel(self.controller)
        layout.addWidget(self.curriculum_panel, stretch=2)

        # Right: Level progress + Network viz
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Level progress
        self.level_progress = LevelProgressPanel(self.controller)
        right_layout.addWidget(self.level_progress)

        # Network visualization
        self.network_viz_curriculum = NetworkVizPanel()
        right_layout.addWidget(self.network_viz_curriculum)

        layout.addWidget(right_panel, stretch=1)

        return tab

    def _create_free_play_tab(self) -> QWidget:
        """Create the Free Play tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left: Chat panel
        self.chat_panel = ChatPanel(self.controller)
        layout.addWidget(self.chat_panel, stretch=1)

        # Center: Creative canvas + Network viz
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Creative canvas (512x512)
        self.creative_canvas = CreativeCanvasWidget()
        center_layout.addWidget(self.creative_canvas)

        # Network visualization
        self.network_viz_free = NetworkVizPanel()
        center_layout.addWidget(self.network_viz_free)

        layout.addWidget(center_panel, stretch=2)

        # Right: Input panel (webcam + mic)
        self.input_panel = InputPanel()
        layout.addWidget(self.input_panel, stretch=1)

        return tab

    def _create_bottom_panel(self) -> QWidget:
        """Create the bottom panel with knowledge base display."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel if PYQT_VERSION == 5 else QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # Knowledge base panel
        self.knowledge_panel = KnowledgeBasePanel(self.controller)
        layout.addWidget(self.knowledge_panel)

        return panel

    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # GPU status
        self.gpu_label = QLabel("GPU: Checking...")
        self.status_bar.addWidget(self.gpu_label)

        # Memory status
        self.memory_label = QLabel("Memory: --")
        self.status_bar.addWidget(self.memory_label)

        # FPS indicator
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addWidget(self.fps_label)

        # Learning status
        self.learning_label = QLabel("Learning: Idle")
        self.status_bar.addWidget(self.learning_label)

        # AV Processing status
        self.av_label = QLabel("AV: --")
        self.status_bar.addPermanentWidget(self.av_label)

        # Initial status update
        self._update_status_bar()

    def _connect_signals(self):
        """Connect signals between components."""
        # Capture thread -> Input panel
        self.capture_thread.frameReady.connect(self.input_panel.update_webcam)
        self.capture_thread.audioLevelReady.connect(self.input_panel.update_audio_level)

        # Capture thread -> Controller (for AV processing)
        self.capture_thread.frameReady.connect(self._on_frame_ready)
        self.capture_thread.audioReady.connect(self._on_audio_ready)

        # Canvas thread -> Creative canvas
        self.canvas_thread.canvasReady.connect(self.creative_canvas.update_canvas)

        # Processing thread -> UI updates
        self.processing_thread.metricsUpdated.connect(self._on_metrics_updated)
        self.processing_thread.learningComplete.connect(self._on_learning_complete)
        self.processing_thread.networkUpdated.connect(self._on_network_update_safe)

        # Connect signal bridge signals to UI slots (thread-safe)
        self.signal_bridge.progressUpdated.connect(self._on_progress_safe)
        self.signal_bridge.learningComplete.connect(self._on_auto_learning_complete_safe)
        self.signal_bridge.networkUpdated.connect(self._on_network_update_safe)

        # Controller callbacks emit signals (called from background thread)
        self.controller.register_progress_callback(
            lambda metrics: self.signal_bridge.progressUpdated.emit(metrics)
        )
        self.controller.register_network_callback(
            lambda state: self.signal_bridge.networkUpdated.emit(state)
        )
        self.controller.register_learning_complete_callback(
            lambda result: self.signal_bridge.learningComplete.emit(result)
        )

    def _start_update_timer(self):
        """Start the periodic update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._on_update_tick)
        self.update_timer.start(500)  # 2 FPS for UI updates (reduced to prevent blocking)

    def _start_auto_learning_timer(self):
        """Start auto curriculum learning after UI is ready."""
        self.auto_learning_timer = QTimer()
        self.auto_learning_timer.setSingleShot(True)
        self.auto_learning_timer.timeout.connect(self._on_start_auto_learning)
        self.auto_learning_timer.start(2000)  # 2 second delay for UI to settle

    def _on_start_auto_learning(self):
        """Start automatic curriculum learning."""
        logger.info("Starting automatic curriculum learning...")
        self.curriculum_panel.log_output.append("Starting automatic curriculum learning...")
        self.curriculum_panel.log_output.append("Atlas will retry challenges until they pass.")
        self.controller.start_auto_curriculum_learning()

    def _on_update_tick(self):
        """Periodic UI update."""
        self._update_status_bar()
        self._update_knowledge_panel()

    def _update_status_bar(self):
        """Update status bar information."""
        stats = self.controller.get_stats()

        # GPU info
        if stats["gpu_available"]:
            gpu_name = stats.get("gpu_name", "Unknown")
            self.gpu_label.setText(f"GPU: {gpu_name}")
        else:
            self.gpu_label.setText("GPU: CPU Mode")

        # Memory
        mem_gb = stats.get("gpu_memory_gb") or 0
        if mem_gb > 0:
            self.memory_label.setText(f"Memory: {mem_gb:.1f}GB")

        # Learning status
        if self.controller.is_auto_learning_active():
            if stats["is_learning"]:
                self.learning_label.setText(f"AUTO: {stats['current_strategy']} ({stats['current_accuracy']:.1%})")
                self.learning_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            else:
                self.learning_label.setText("AUTO: Preparing next...")
                self.learning_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        elif stats["is_learning"]:
            self.learning_label.setText(f"Learning: {stats['current_strategy']} ({stats['current_accuracy']:.1%})")
            self.learning_label.setStyleSheet("color: #4caf50;")
        else:
            self.learning_label.setText("Learning: Idle")
            self.learning_label.setStyleSheet("")

        # AV status
        if stats["av_processing"]:
            self.av_label.setText("AV: Processing")
            self.av_label.setStyleSheet("color: #2196f3;")
        else:
            self.av_label.setText("AV: Inactive")
            self.av_label.setStyleSheet("")

    def _update_knowledge_panel(self):
        """Update knowledge base panel."""
        self.knowledge_panel.update_stats()

    def _on_frame_ready(self, frame):
        """Handle new webcam frame."""
        self.controller.process_av_frame(frame, self.capture_thread.last_audio)

    def _on_audio_ready(self, audio):
        """Handle new audio data."""
        # Audio is processed together with frames
        pass

    def _on_metrics_updated(self, metrics: dict):
        """Handle learning metrics update."""
        # Update curriculum panel if in curriculum tab
        if self.tab_widget.currentIndex() == 0:
            self.curriculum_panel.update_metrics(metrics)

        # DON'T update network viz here - it's too expensive and causes UI freeze
        # Network viz is updated by the periodic timer instead

    def _on_learning_complete(self, result: dict):
        """Handle learning completion."""
        # Refresh level progress
        self.level_progress.refresh()

        # Update curriculum panel
        self.curriculum_panel.on_challenge_complete(result)

    def _on_progress(self, metrics: dict):
        """Handle progress callback from controller (legacy, may be called from any thread)."""
        self._on_metrics_updated(metrics)

    def _on_progress_safe(self, metrics: dict):
        """Handle progress callback - thread-safe slot called on UI thread."""
        try:
            self._on_metrics_updated(metrics)
        except Exception as e:
            logger.error(f"Error in progress update: {e}")

    def _on_network_update(self, network_state: dict):
        """Handle network structure update (legacy)."""
        self.network_viz_curriculum.update_network(network_state)
        self.network_viz_free.update_network(network_state)

    def _on_network_update_safe(self, network_state: dict):
        """Handle network structure update - thread-safe slot called on UI thread."""
        try:
            self.network_viz_curriculum.update_network(network_state)
            self.network_viz_free.update_network(network_state)
        except Exception as e:
            logger.error(f"Error in network update: {e}")

    def _on_auto_learning_complete(self, result: dict):
        """Handle auto-learning challenge completion (legacy)."""
        self.curriculum_panel.on_challenge_complete(result)
        self.level_progress.refresh()

    def _on_auto_learning_complete_safe(self, result: dict):
        """Handle auto-learning challenge completion - thread-safe slot called on UI thread."""
        try:
            # Update curriculum panel
            self.curriculum_panel.on_challenge_complete(result)

            # Refresh level progress
            self.level_progress.refresh()
        except Exception as e:
            logger.error(f"Error in learning complete update: {e}")

    def _on_save_checkpoint(self):
        """Save system checkpoint."""
        logger.info("Saving checkpoint...")
        # TODO: Implement checkpoint saving

    def _on_load_checkpoint(self):
        """Load system checkpoint."""
        logger.info("Loading checkpoint...")
        # TODO: Implement checkpoint loading

    def _on_reset_learning(self):
        """Reset all learning to start fresh."""
        from PyQt5.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Reset Learning",
            "This will delete all learned weights and reset curriculum progress.\n\n"
            "Atlas will start learning from scratch with random weights.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            logger.info("User requested learning reset")

            # Stop auto-learning if active
            self.controller.stop_auto_curriculum_learning()

            # Reset all learning
            self.controller.reset_learning()

            # Update UI
            self.curriculum_panel.log_output.clear()
            self.curriculum_panel.log_output.append("=== LEARNING RESET ===")
            self.curriculum_panel.log_output.append("All weights reset to random initialization.")
            self.curriculum_panel.log_output.append("Curriculum progress cleared.")
            self.curriculum_panel.log_output.append("Ready to start fresh learning!")
            self.curriculum_panel.log_output.append("")

            # Refresh level progress
            self.level_progress.refresh()

            # Update network visualization
            network_state = self.controller.get_network_state()
            self.network_viz_curriculum.update_network(network_state)
            self.network_viz_free.update_network(network_state)

            # Restart auto-learning
            self.curriculum_panel.log_output.append("Starting automatic curriculum learning...")
            self.controller.start_auto_curriculum_learning()

            QMessageBox.information(
                self,
                "Reset Complete",
                "Learning has been reset. Atlas is now learning from scratch!"
            )

    def _on_about(self):
        """Show about dialog."""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About Atlas",
            "Atlas Challenge-Based Learning System\n\n"
            "A biology-inspired learning system using:\n"
            "- Hebbian learning\n"
            "- STDP (Spike-Timing Dependent Plasticity)\n"
            "- BCM learning rule\n\n"
            "No backpropagation!"
        )

    def closeEvent(self, event):
        """Handle window close."""
        logger.info("Closing Atlas application...")

        # Stop auto-learning timer if still pending
        if hasattr(self, 'auto_learning_timer'):
            self.auto_learning_timer.stop()

        # Stop threads
        self.capture_thread.stop()
        self.canvas_thread.stop()
        self.processing_thread.stop()

        # Wait for threads
        self.capture_thread.wait(2000)
        self.canvas_thread.wait(2000)
        self.processing_thread.wait(2000)

        # Close controller - saves to database and closes connections
        self.controller.close()

        event.accept()
