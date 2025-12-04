"""
Dark Theme Stylesheet for Atlas GUI

VS Code-inspired dark theme with accent colors for different components.
"""

DARK_STYLESHEET = """
/* ============================================
   GLOBAL STYLES
   ============================================ */

QMainWindow {
    background-color: #1e1e1e;
    color: #d4d4d4;
}

QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}

/* ============================================
   MENU BAR
   ============================================ */

QMenuBar {
    background-color: #252526;
    color: #cccccc;
    border-bottom: 1px solid #3c3c3c;
    padding: 2px;
}

QMenuBar::item {
    padding: 5px 10px;
    background: transparent;
}

QMenuBar::item:selected {
    background-color: #094771;
}

QMenu {
    background-color: #252526;
    color: #cccccc;
    border: 1px solid #3c3c3c;
}

QMenu::item {
    padding: 5px 30px 5px 20px;
}

QMenu::item:selected {
    background-color: #094771;
}

QMenu::separator {
    height: 1px;
    background-color: #3c3c3c;
    margin: 5px 0;
}

/* ============================================
   TAB WIDGET
   ============================================ */

QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background-color: #1e1e1e;
    top: -1px;
}

QTabBar::tab {
    background-color: #2d2d30;
    color: #808080;
    padding: 10px 20px;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #4fc3f7;
}

QTabBar::tab:hover:!selected {
    background-color: #353535;
    color: #cccccc;
}

/* ============================================
   STATUS BAR
   ============================================ */

QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    border: none;
    padding: 3px;
}

QStatusBar::item {
    border: none;
}

QStatusBar QLabel {
    color: #ffffff;
    padding: 0 10px;
}

/* ============================================
   BUTTONS
   ============================================ */

QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1177bb;
}

QPushButton:pressed {
    background-color: #0d5085;
}

QPushButton:disabled {
    background-color: #3c3c3c;
    color: #6c6c6c;
}

/* Secondary button style */
QPushButton[secondary="true"] {
    background-color: #3c3c3c;
    color: #cccccc;
}

QPushButton[secondary="true"]:hover {
    background-color: #4c4c4c;
}

/* ============================================
   INPUT FIELDS
   ============================================ */

QLineEdit {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 8px;
    selection-background-color: #264f78;
}

QLineEdit:focus {
    border-color: #007acc;
}

QLineEdit:disabled {
    background-color: #2d2d30;
    color: #6c6c6c;
}

QTextEdit {
    background-color: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    selection-background-color: #264f78;
}

QTextEdit:focus {
    border-color: #007acc;
}

/* ============================================
   COMBO BOX
   ============================================ */

QComboBox {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 100px;
}

QComboBox:hover {
    border-color: #007acc;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    selection-background-color: #094771;
}

/* ============================================
   PROGRESS BAR
   ============================================ */

QProgressBar {
    background-color: #3c3c3c;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #0e639c;
    border-radius: 4px;
}

/* ============================================
   SCROLL BARS
   ============================================ */

QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 14px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #5a5a5a;
    min-height: 30px;
    border-radius: 7px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #7a7a7a;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 14px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #5a5a5a;
    min-width: 30px;
    border-radius: 7px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #7a7a7a;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ============================================
   FRAMES & GROUPS
   ============================================ */

QFrame {
    background-color: transparent;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #3c3c3c;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #cccccc;
}

/* ============================================
   SPLITTERS
   ============================================ */

QSplitter::handle {
    background-color: #3c3c3c;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QSplitter::handle:hover {
    background-color: #007acc;
}

/* ============================================
   TABLES
   ============================================ */

QTableWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    border: none;
    gridline-color: #3c3c3c;
    selection-background-color: #094771;
}

QTableWidget::item {
    padding: 5px;
}

QTableWidget::item:selected {
    background-color: #094771;
}

QHeaderView::section {
    background-color: #252526;
    color: #cccccc;
    padding: 8px;
    border: none;
    border-right: 1px solid #3c3c3c;
    border-bottom: 1px solid #3c3c3c;
}

/* ============================================
   LABELS
   ============================================ */

QLabel {
    color: #d4d4d4;
    background: transparent;
}

/* ============================================
   TOOLTIPS
   ============================================ */

QToolTip {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    padding: 5px;
}

/* ============================================
   MESSAGE BOX
   ============================================ */

QMessageBox {
    background-color: #252526;
}

QMessageBox QLabel {
    color: #d4d4d4;
}

QMessageBox QPushButton {
    min-width: 80px;
}

/* ============================================
   SCROLL AREA
   ============================================ */

QScrollArea {
    border: none;
    background-color: transparent;
}

/* ============================================
   CUSTOM ACCENT COLORS
   ============================================ */

/* Input Panel (Webcam/Mic) - Blue accent */
#input_panel QLabel {
    color: #64b5f6;
}

/* Creative Canvas - Purple accent */
#creative_canvas QLabel {
    color: #ce93d8;
}

/* Curriculum - Orange accent */
#curriculum_panel QLabel {
    color: #ffb74d;
}

/* Chat - Cyan accent */
#chat_panel QLabel {
    color: #4fc3f7;
}

/* Network Viz - Deep Orange accent */
#network_viz QLabel {
    color: #ff8a65;
}

/* Knowledge Base - Yellow accent */
#knowledge_base QLabel {
    color: #dcdcaa;
}
"""

# Color palette reference
COLORS = {
    "background": "#1e1e1e",
    "background_secondary": "#252526",
    "background_tertiary": "#2d2d30",
    "foreground": "#d4d4d4",
    "foreground_dim": "#808080",
    "border": "#3c3c3c",
    "accent_blue": "#007acc",
    "accent_light_blue": "#4fc3f7",
    "accent_green": "#4caf50",
    "accent_red": "#f44336",
    "accent_orange": "#ffb74d",
    "accent_purple": "#ce93d8",
    "accent_yellow": "#dcdcaa",
    "selection": "#094771",
    "button": "#0e639c",
    "button_hover": "#1177bb",
}
