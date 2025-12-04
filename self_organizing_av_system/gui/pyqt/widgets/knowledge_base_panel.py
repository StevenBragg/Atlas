"""
Knowledge Base Panel - Displays Atlas's memory contents.

Shows episodic memories and semantic concepts from the unified knowledge base.
"""

import logging
from typing import Optional, Dict, Any, List

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
        QProgressBar
    )
    from PyQt5.QtCore import Qt
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
        QProgressBar
    )
    from PyQt6.QtCore import Qt
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class KnowledgeBasePanel(QWidget):
    """
    Knowledge Base Panel - Shows Atlas's memory system.

    Displays:
    - Episodic memories (recent experiences)
    - Semantic concepts (learned knowledge)
    - Memory statistics
    """

    def __init__(self, controller, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with stats
        header = self._create_header()
        layout.addWidget(header)

        # Tabs for different memory views
        self.memory_tabs = QTabWidget()
        self.memory_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d30;
                color: #888;
                padding: 8px 20px;
                border: 1px solid #3c3c3c;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: white;
            }
        """)

        # Episodic memory tab
        episodic_tab = self._create_episodic_tab()
        self.memory_tabs.addTab(episodic_tab, "Episodic Memory")

        # Semantic memory tab
        semantic_tab = self._create_semantic_tab()
        self.memory_tabs.addTab(semantic_tab, "Semantic Concepts")

        layout.addWidget(self.memory_tabs)

    def _create_header(self) -> QWidget:
        """Create the header with memory stats."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-radius: 3px;
                padding: 5px;
            }
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title
        title = QLabel("Knowledge Base")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #dcdcaa;")
        layout.addWidget(title)

        layout.addStretch()

        # Episodic count
        self.episodic_count = QLabel("Episodes: 0")
        self.episodic_count.setStyleSheet("font-size: 11px; color: #4fc3f7;")
        layout.addWidget(self.episodic_count)

        # Semantic count
        self.semantic_count = QLabel("Concepts: 0")
        self.semantic_count.setStyleSheet("font-size: 11px; color: #ce93d8;")
        layout.addWidget(self.semantic_count)

        # Memory usage bar
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setValue(0)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%v%")
        self.memory_bar.setFixedWidth(100)
        self.memory_bar.setFixedHeight(15)
        self.memory_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.memory_bar)

        return header

    def _create_episodic_tab(self) -> QWidget:
        """Create the episodic memory display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Recent episodes table
        self.episodes_table = QTableWidget()
        self.episodes_table.setColumnCount(4)
        self.episodes_table.setHorizontalHeaderLabels([
            "Time", "Source", "Context", "Valence"
        ])
        self.episodes_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                border: none;
                gridline-color: #3c3c3c;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #888;
                padding: 5px;
                border: 1px solid #3c3c3c;
            }
        """)

        header = self.episodes_table.horizontalHeader()
        if PYQT_VERSION == 5:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.episodes_table)

        return tab

    def _create_semantic_tab(self) -> QWidget:
        """Create the semantic concepts display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Concepts table
        self.concepts_table = QTableWidget()
        self.concepts_table.setColumnCount(4)
        self.concepts_table.setHorizontalHeaderLabels([
            "Concept", "Type", "Strength", "Connections"
        ])
        self.concepts_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                border: none;
                gridline-color: #3c3c3c;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #888;
                padding: 5px;
                border: 1px solid #3c3c3c;
            }
        """)

        header = self.concepts_table.horizontalHeader()
        if PYQT_VERSION == 5:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.concepts_table)

        return tab

    def update_stats(self):
        """Update the knowledge base statistics."""
        kb = self.controller.knowledge_base
        stats = kb.get_stats()

        # Update counts
        ep_count = stats.get("total_episodes", 0)
        sem_count = stats.get("total_concepts", 0)

        self.episodic_count.setText(f"Episodes: {ep_count:,}")
        self.semantic_count.setText(f"Concepts: {sem_count:,}")

        # Update memory bar
        total_capacity = 100000
        used = ep_count + sem_count
        usage_pct = min(100, int((used / total_capacity) * 100))
        self.memory_bar.setValue(usage_pct)

        # Update tables periodically (not every tick)
        if stats.get("recent_events_count", 0) > 0:
            self._update_episodes_table()
            self._update_concepts_table()

    def _update_episodes_table(self):
        """Update the episodic memory table."""
        kb = self.controller.knowledge_base

        # Get recent events (returns KnowledgeEvent objects)
        recent = kb.get_recent_events(n=20)

        self.episodes_table.setRowCount(len(recent))

        for i, event in enumerate(recent):
            # Time - format timestamp
            import time
            time_str = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            time_item = QTableWidgetItem(time_str)
            self.episodes_table.setItem(i, 0, time_item)

            # Source
            source_item = QTableWidgetItem(event.source)
            self.episodes_table.setItem(i, 1, source_item)

            # Description (truncated)
            desc = event.description[:50] + "..." if len(event.description) > 50 else event.description
            desc_item = QTableWidgetItem(desc)
            self.episodes_table.setItem(i, 2, desc_item)

            # Consolidation strength as valence proxy
            strength = event.consolidation_strength
            strength_item = QTableWidgetItem(f"{strength:.2f}")
            if strength > 0.5:
                strength_item.setForeground(Qt.green if PYQT_VERSION == 5 else Qt.GlobalColor.green)
            self.episodes_table.setItem(i, 3, strength_item)

    def _update_concepts_table(self):
        """Update the semantic concepts table."""
        kb = self.controller.knowledge_base
        stats = kb.get_stats()

        # Show semantic memory stats as concepts
        concepts_data = [
            {"name": "Total Concepts", "type": "count", "strength": stats.get("total_concepts", 0), "connections": 0},
            {"name": "Total Relations", "type": "count", "strength": stats.get("total_relations", 0), "connections": 0},
            {"name": "Inferences Made", "type": "count", "strength": stats.get("total_inferences", 0), "connections": 0},
            {"name": "Generalizations", "type": "count", "strength": stats.get("total_generalizations", 0), "connections": 0},
        ]

        self.concepts_table.setRowCount(len(concepts_data))

        for i, concept in enumerate(concepts_data):
            # Name
            name_item = QTableWidgetItem(concept["name"])
            self.concepts_table.setItem(i, 0, name_item)

            # Type
            type_item = QTableWidgetItem(concept["type"])
            self.concepts_table.setItem(i, 1, type_item)

            # Strength/Value
            strength = concept["strength"]
            strength_item = QTableWidgetItem(str(int(strength)))
            self.concepts_table.setItem(i, 2, strength_item)

            # Connections
            conn_item = QTableWidgetItem("--")
            self.concepts_table.setItem(i, 3, conn_item)

    def refresh(self):
        """Force refresh all displays."""
        self.update_stats()
        self._update_episodes_table()
        self._update_concepts_table()
