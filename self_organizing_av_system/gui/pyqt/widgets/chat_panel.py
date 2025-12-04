"""
Chat Panel Widget for Free Play Tab

Provides chat interface for natural language interaction with Atlas.
Atlas learns to respond through biology-inspired rules (not templates).
"""

import logging
import threading
from typing import Optional

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
        QLineEdit, QPushButton, QLabel
    )
    from PyQt5.QtCore import pyqtSignal, Qt, QObject
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
        QLineEdit, QPushButton, QLabel
    )
    from PyQt6.QtCore import pyqtSignal, Qt, QObject
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


class ChatSignalBridge(QObject):
    """Thread-safe signal bridge for chat responses."""
    responseReady = pyqtSignal(str, dict)
    errorOccurred = pyqtSignal(str)


class ChatPanel(QWidget):
    """
    Chat interface for natural language interaction with Atlas.

    Features:
    - Message history with HTML formatting
    - Input field for typing messages
    - Atlas responds using learned text generation
    """

    messageSubmitted = pyqtSignal(str)

    def __init__(self, controller, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller

        # Signal bridge for thread-safe UI updates
        self._signal_bridge = ChatSignalBridge()
        self._signal_bridge.responseReady.connect(self._on_response_ready)
        self._signal_bridge.errorOccurred.connect(self._on_error)

        # Track if we're waiting for a response
        self._waiting_for_response = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Chat with Atlas")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(title)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.chat_history, stretch=1)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message or challenge...")
        self.input_field.returnPressed.connect(self._on_send)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d30;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4fc3f7;
            }
        """)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5085;
            }
        """)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Add welcome message
        self._add_system_message("Welcome to Atlas Free Play! Type a message or learning challenge.")

    def _on_send(self):
        """Handle send button click."""
        text = self.input_field.text().strip()
        if not text:
            return

        # Don't allow multiple concurrent requests
        if self._waiting_for_response:
            return

        # Add user message
        self._add_user_message(text)

        # Clear input and disable send
        self.input_field.clear()
        self._waiting_for_response = True
        self.send_button.setEnabled(False)
        self.send_button.setText("...")

        # Add thinking indicator
        self._add_system_message("Atlas is thinking...")

        # Process in background thread to keep UI responsive
        def process_message():
            try:
                response = self.controller.process_chat_message(text)
                self._signal_bridge.responseReady.emit(response.text, {
                    "confidence": response.confidence,
                })
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                self._signal_bridge.errorOccurred.emit(str(e))

        thread = threading.Thread(target=process_message, daemon=True)
        thread.start()

        # Emit signal
        self.messageSubmitted.emit(text)

    def _on_response_ready(self, text: str, metrics: dict):
        """Handle response from background thread (called on UI thread)."""
        self._waiting_for_response = False
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self._add_atlas_response(text, metrics)

    def _on_error(self, error_message: str):
        """Handle error from background thread (called on UI thread)."""
        self._waiting_for_response = False
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self._add_system_message(f"Error: {error_message}")

    def _add_user_message(self, text: str):
        """Add a user message to the chat."""
        html = f'''
        <div style="margin: 8px 0; padding: 8px; background-color: #1a3a5c; border-radius: 5px;">
            <span style="color: #4fc3f7; font-weight: bold;">You:</span>
            <span style="color: #ffffff;"> {text}</span>
        </div>
        '''
        self.chat_history.append(html)

    def _add_atlas_response(self, text: str, metrics: Optional[dict] = None):
        """Add Atlas's response to the chat."""
        metrics_html = ""
        if metrics:
            confidence = metrics.get("confidence", 0)
            metrics_html = f'<br/><small style="color: #888;">Confidence: {confidence:.1%}</small>'

        html = f'''
        <div style="margin: 8px 0; padding: 8px; background-color: #1a3c1a; border-radius: 5px;">
            <span style="color: #81c784; font-weight: bold;">Atlas:</span>
            <span style="color: #ffffff;"> {text}</span>
            {metrics_html}
        </div>
        '''
        self.chat_history.append(html)

    def _add_system_message(self, text: str):
        """Add a system message to the chat."""
        html = f'''
        <div style="margin: 8px 0; padding: 5px; color: #888; font-style: italic;">
            {text}
        </div>
        '''
        self.chat_history.append(html)
