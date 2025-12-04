"""
Network Visualization Panel - Shows Atlas's REAL neural network structure.

Visualizes the actual self-organizing network with:
- Real layer structure from SelfOrganizingNetwork
- Dynamic neuron counts that change during learning
- Animated growth (green glow) and pruning (red fade)
- Connection weight visualization
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QRadialGradient
    from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QPointF
    PYQT_VERSION = 5
except ImportError:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QRadialGradient
    from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QPointF
    PYQT_VERSION = 6

logger = logging.getLogger(__name__)


@dataclass
class AnimatedNeuron:
    """Represents a neuron with animation state."""
    layer_idx: int
    neuron_idx: int
    x: float
    y: float
    animation_type: str  # 'grow', 'prune', 'pulse', 'none'
    animation_start: float
    animation_duration: float = 1.0


class NetworkVizPanel(QWidget):
    """
    Network Visualization Panel - Shows REAL neural network structure.

    Visualizes the actual self-organizing network including:
    - Real layer sizes from SelfOrganizingNetwork
    - Neuron-level activation display
    - Animated growth/pruning events
    - Connection weight visualization
    """

    VIZ_WIDTH = 400
    VIZ_HEIGHT = 200

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._network_state: Optional[Dict[str, Any]] = None
        self._animated_neurons: List[AnimatedNeuron] = []
        self._animation_timer: Optional[QTimer] = None
        self._last_update_time = 0.0
        self._pending_events: List[Dict[str, Any]] = []
        self._setup_ui()
        self._start_animation_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Title
        title = QLabel("Neural Network Structure")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #ff8a65;")
        layout.addWidget(title)

        # Visualization canvas
        self.viz_frame = QFrame()
        self.viz_frame.setFixedSize(self.VIZ_WIDTH + 4, self.VIZ_HEIGHT + 4)
        self.viz_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #ff8a65;
                border-radius: 3px;
            }
        """)

        self.viz_label = QLabel(self.viz_frame)
        self.viz_label.setFixedSize(self.VIZ_WIDTH, self.VIZ_HEIGHT)
        self.viz_label.move(2, 2)

        # Initialize with placeholder
        self._draw_placeholder()

        layout.addWidget(self.viz_frame)

        # Stats row
        stats_row = QHBoxLayout()

        self.neurons_label = QLabel("Neurons: --")
        self.neurons_label.setStyleSheet("font-size: 10px; color: #888;")
        stats_row.addWidget(self.neurons_label)

        self.connections_label = QLabel("Connections: --")
        self.connections_label.setStyleSheet("font-size: 10px; color: #888;")
        stats_row.addWidget(self.connections_label)

        self.activity_label = QLabel("Activity: --")
        self.activity_label.setStyleSheet("font-size: 10px; color: #4caf50;")
        stats_row.addWidget(self.activity_label)

        layout.addLayout(stats_row)

        # Second row for structural changes
        changes_row = QHBoxLayout()

        self.added_label = QLabel("+0 added")
        self.added_label.setStyleSheet("font-size: 10px; color: #4caf50;")
        changes_row.addWidget(self.added_label)

        self.pruned_label = QLabel("-0 pruned")
        self.pruned_label.setStyleSheet("font-size: 10px; color: #f44336;")
        changes_row.addWidget(self.pruned_label)

        self.layers_label = QLabel("Layers: --")
        self.layers_label.setStyleSheet("font-size: 10px; color: #2196f3;")
        changes_row.addWidget(self.layers_label)

        layout.addLayout(changes_row)

    def _start_animation_timer(self):
        """Start the animation update timer."""
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._animation_timer.start(50)  # 20 FPS for animations

    def _on_animation_tick(self):
        """Update animations."""
        if not self._animated_neurons and not self._pending_events:
            return

        current_time = time.time()

        # Process pending events
        for event in self._pending_events:
            self._create_animation_from_event(event)
        self._pending_events.clear()

        # Remove expired animations
        self._animated_neurons = [
            n for n in self._animated_neurons
            if current_time - n.animation_start < n.animation_duration
        ]

        # Re-render if there are active animations
        if self._animated_neurons:
            self._render_network()

    def _create_animation_from_event(self, event: Dict[str, Any]):
        """Create animation from a structural event."""
        event_type = event.get('type', '')
        layer_idx = event.get('layer', 0)
        neuron_idx = event.get('neuron')

        if event_type == 'neuron_added' and neuron_idx is not None:
            self._animated_neurons.append(AnimatedNeuron(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                x=0, y=0,  # Will be calculated during render
                animation_type='grow',
                animation_start=time.time(),
                animation_duration=1.5,
            ))
        elif event_type == 'neuron_pruned' and neuron_idx is not None:
            self._animated_neurons.append(AnimatedNeuron(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                x=0, y=0,
                animation_type='prune',
                animation_start=time.time(),
                animation_duration=1.0,
            ))

    def _draw_placeholder(self):
        """Draw a placeholder visualization."""
        pixmap = QPixmap(self.VIZ_WIDTH, self.VIZ_HEIGHT)
        pixmap.fill(QColor(26, 26, 26))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing if PYQT_VERSION == 5 else QPainter.RenderHint.Antialiasing)

        # Draw placeholder text
        painter.setPen(QColor(100, 100, 100))
        painter.drawText(
            pixmap.rect(),
            Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter,
            "Waiting for network data...\nStructure will appear during learning"
        )

        painter.end()
        self.viz_label.setPixmap(pixmap)

    @pyqtSlot(object)
    def update_network(self, network_state: Dict[str, Any]):
        """
        Update the network visualization with REAL network state.

        Args:
            network_state: Dict from AtlasController.get_network_state() containing:
                - layers: List of layer info dicts with 'size', 'neurons', etc.
                - total_neurons: Total neuron count
                - total_connections: Total connection count
                - recent_events: List of structural change events
                - neurons_added: Count of recently added neurons
                - neurons_pruned: Count of recently pruned neurons
        """
        if not network_state:
            return

        self._network_state = network_state
        self._last_update_time = time.time()

        # Process recent events for animation
        recent_events = network_state.get('recent_events', [])
        for event in recent_events:
            # Only process very recent events (within last 2 seconds)
            event_time = event.get('timestamp', 0)
            if time.time() - event_time < 2.0:
                self._pending_events.append(event)

        self._render_network()

    def _render_network(self):
        """Render the neural network visualization with real data."""
        if not self._network_state:
            self._draw_placeholder()
            return

        pixmap = QPixmap(self.VIZ_WIDTH, self.VIZ_HEIGHT)
        pixmap.fill(QColor(26, 26, 26))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing if PYQT_VERSION == 5 else QPainter.RenderHint.Antialiasing)

        # Get real layer data
        layers_data = self._network_state.get('layers', [])

        if not layers_data:
            # Fall back to legacy format if new format not available
            visual_layers = self._network_state.get('visual_layers', [64, 32, 16])
            layers_data = [{'size': s, 'neurons': []} for s in visual_layers]

        num_layers = len(layers_data)
        if num_layers == 0:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                pixmap.rect(),
                Qt.AlignCenter if PYQT_VERSION == 5 else Qt.AlignmentFlag.AlignCenter,
                "No layers yet"
            )
            painter.end()
            self.viz_label.setPixmap(pixmap)
            return

        # Layout parameters
        margin_x = 50
        margin_y = 30
        layer_spacing = (self.VIZ_WIDTH - 2 * margin_x) / max(1, num_layers - 1) if num_layers > 1 else 0

        # Calculate neuron positions for each layer
        all_positions: List[List[Tuple[float, float]]] = []
        layer_sizes = []

        for layer_idx, layer_info in enumerate(layers_data):
            layer_size = layer_info.get('size', 0) if isinstance(layer_info, dict) else layer_info
            layer_sizes.append(layer_size)

            # Limit displayed neurons for clarity
            display_size = min(layer_size, 12)
            layer_positions = []

            x = margin_x + layer_idx * layer_spacing if num_layers > 1 else self.VIZ_WIDTH / 2
            neuron_spacing = (self.VIZ_HEIGHT - 2 * margin_y) / max(1, display_size - 1) if display_size > 1 else 0

            for i in range(display_size):
                y = margin_y + i * neuron_spacing if display_size > 1 else self.VIZ_HEIGHT / 2
                layer_positions.append((x, y))

            all_positions.append(layer_positions)

        # Draw connections between layers
        self._draw_connections(painter, all_positions, layers_data)

        # Draw neurons with animations
        current_time = time.time()
        global_neuron_idx = 0

        for layer_idx, layer_positions in enumerate(all_positions):
            layer_info = layers_data[layer_idx] if layer_idx < len(layers_data) else {}
            neurons_info = layer_info.get('neurons', []) if isinstance(layer_info, dict) else []

            for pos_idx, (x, y) in enumerate(layer_positions):
                # Check for active animations at this position
                animation = self._find_animation(layer_idx, pos_idx)

                # Get neuron info if available
                neuron_info = neurons_info[pos_idx] if pos_idx < len(neurons_info) else {}
                activation = neuron_info.get('activation', 0.0) if neuron_info else 0.0
                is_winner = neuron_info.get('is_winner', False) if neuron_info else False

                # Draw neuron with appropriate style
                self._draw_neuron(painter, x, y, activation, is_winner, animation, current_time)

                global_neuron_idx += 1

        # Draw layer labels
        self._draw_layer_labels(painter, all_positions, layer_sizes, layers_data)

        painter.end()
        self.viz_label.setPixmap(pixmap)

        # Update stats labels
        self._update_stats_labels(layer_sizes)

    def _draw_connections(self, painter: QPainter, all_positions: List[List[Tuple[float, float]]], layers_data: List):
        """Draw connections between layers."""
        for layer_idx in range(len(all_positions) - 1):
            current_layer = all_positions[layer_idx]
            next_layer = all_positions[layer_idx + 1]

            # Get weight stats if available
            layer_info = layers_data[layer_idx] if layer_idx < len(layers_data) else {}
            weight_stats = layer_info.get('weight_stats', {}) if isinstance(layer_info, dict) else {}

            for i, (x1, y1) in enumerate(current_layer):
                for j, (x2, y2) in enumerate(next_layer):
                    # Only draw some connections for clarity
                    if (i + j) % 2 == 0:
                        # Vary color based on connection pattern
                        alpha = 40 + (i * j) % 30
                        pen = QPen(QColor(80, 80, 120, alpha))
                        pen.setWidth(1)
                        painter.setPen(pen)
                        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _find_animation(self, layer_idx: int, neuron_idx: int) -> Optional[AnimatedNeuron]:
        """Find active animation for a neuron."""
        for anim in self._animated_neurons:
            if anim.layer_idx == layer_idx and anim.neuron_idx == neuron_idx:
                return anim
        return None

    def _draw_neuron(
        self,
        painter: QPainter,
        x: float,
        y: float,
        activation: float,
        is_winner: bool,
        animation: Optional[AnimatedNeuron],
        current_time: float
    ):
        """Draw a single neuron with optional animation."""
        base_radius = 5

        if animation:
            # Calculate animation progress (0.0 to 1.0)
            progress = (current_time - animation.animation_start) / animation.animation_duration
            progress = min(1.0, max(0.0, progress))

            if animation.animation_type == 'grow':
                # Growth animation: expand and glow green
                scale = 0.5 + progress * 0.5  # Start small, grow to full
                radius = int(base_radius * (1 + (1 - progress) * 0.5))  # Slightly larger during animation
                glow_alpha = int(255 * (1 - progress))

                # Draw glow
                gradient = QRadialGradient(QPointF(x, y), radius * 3)
                gradient.setColorAt(0, QColor(76, 175, 80, glow_alpha))
                gradient.setColorAt(1, QColor(76, 175, 80, 0))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen if PYQT_VERSION == 5 else Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x - radius * 2), int(y - radius * 2), radius * 4, radius * 4)

                # Draw neuron
                color = QColor(76, 175, 80)  # Green

            elif animation.animation_type == 'prune':
                # Prune animation: shrink and fade red
                scale = 1 - progress * 0.8  # Shrink to 20%
                radius = int(base_radius * scale)
                alpha = int(255 * (1 - progress))

                # Draw fading red glow
                gradient = QRadialGradient(QPointF(x, y), radius * 2)
                gradient.setColorAt(0, QColor(244, 67, 54, alpha))
                gradient.setColorAt(1, QColor(244, 67, 54, 0))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen if PYQT_VERSION == 5 else Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)

                color = QColor(244, 67, 54, alpha)  # Fading red

            else:
                radius = base_radius
                color = QColor(100, 149, 237)
        else:
            radius = base_radius

            # Color based on activation
            if is_winner:
                color = QColor(255, 215, 0)  # Gold for winners
            elif activation > 0.5:
                intensity = int(100 + 155 * activation)
                color = QColor(intensity, intensity, 255)  # Bright blue
            elif activation > 0.1:
                intensity = int(80 + 100 * activation)
                color = QColor(intensity, intensity, 200)  # Lighter blue
            else:
                color = QColor(60, 80, 120)  # Dim blue

        # Draw the neuron
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)

    def _draw_layer_labels(
        self,
        painter: QPainter,
        all_positions: List[List[Tuple[float, float]]],
        layer_sizes: List[int],
        layers_data: List
    ):
        """Draw layer labels and sizes."""
        painter.setPen(QColor(150, 150, 150))

        for layer_idx, layer_positions in enumerate(all_positions):
            if layer_positions:
                x = layer_positions[0][0]

                # Get layer name
                layer_info = layers_data[layer_idx] if layer_idx < len(layers_data) else {}
                name = layer_info.get('name', f'L{layer_idx}') if isinstance(layer_info, dict) else f'L{layer_idx}'

                # Shorten name if needed
                if len(name) > 8:
                    name = name[:6] + '..'

                # Draw name at top
                painter.drawText(int(x - 20), 12, name)

                # Draw size at bottom
                size = layer_sizes[layer_idx] if layer_idx < len(layer_sizes) else 0
                painter.drawText(int(x - 12), self.VIZ_HEIGHT - 3, f"({size})")

    def _update_stats_labels(self, layer_sizes: List[int]):
        """Update the stats labels with real data."""
        if not self._network_state:
            return

        total_neurons = self._network_state.get('total_neurons', sum(layer_sizes))
        total_connections = self._network_state.get('total_connections', 0)
        neurons_added = self._network_state.get('neurons_added', 0)
        neurons_pruned = self._network_state.get('neurons_pruned', 0)
        num_layers = self._network_state.get('num_layers', len(layer_sizes))

        self.neurons_label.setText(f"Neurons: {total_neurons}")
        self.connections_label.setText(f"Connections: {total_connections:,}")
        self.layers_label.setText(f"Layers: {num_layers}")

        # Show recent changes
        self.added_label.setText(f"+{neurons_added}")
        if neurons_added > 0:
            self.added_label.setStyleSheet("font-size: 10px; color: #4caf50; font-weight: bold;")
        else:
            self.added_label.setStyleSheet("font-size: 10px; color: #666;")

        self.pruned_label.setText(f"-{neurons_pruned}")
        if neurons_pruned > 0:
            self.pruned_label.setStyleSheet("font-size: 10px; color: #f44336; font-weight: bold;")
        else:
            self.pruned_label.setStyleSheet("font-size: 10px; color: #666;")

        # Activity indicator
        stats = self._network_state.get('stats', {})
        learn_count = stats.get('learn_count', 0)
        if learn_count > 0:
            self.activity_label.setText(f"Learning: {learn_count}")
            self.activity_label.setStyleSheet("font-size: 10px; color: #4caf50;")
        else:
            self.activity_label.setText("Ready")
            self.activity_label.setStyleSheet("font-size: 10px; color: #888;")
