"""
Atlas Desktop Application - Main Entry Point

PyQt5 desktop application for ATLAS challenge-based learning system.
Provides two areas:
1. Curriculum Learning (School) - Structured multimodal challenges
2. Free Play - Exploratory learning with webcam, mic, and chat

Usage:
    python -m self_organizing_av_system.gui.pyqt.app
"""

import sys
import logging
from pathlib import Path

# Set up logging before importing PyQt
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QPixmap, QFont
    PYQT_VERSION = 5
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QPixmap, QFont
        PYQT_VERSION = 6
    except ImportError:
        logger.error("PyQt5 or PyQt6 is required. Install with: pip install PyQt5")
        sys.exit(1)

from .main_window import AtlasMainWindow
from .controllers.atlas_controller import AtlasController
from .styles.dark_theme import DARK_STYLESHEET


class AtlasApplication:
    """
    Main application class for Atlas GUI.

    Handles:
    - Application initialization
    - Atlas controller setup
    - Main window creation
    - Application lifecycle
    """

    def __init__(self):
        """Initialize the Atlas application."""
        self.app: QApplication = None
        self.controller: AtlasController = None
        self.main_window: AtlasMainWindow = None

    def run(self) -> int:
        """
        Run the Atlas application.

        Returns:
            Exit code (0 for success)
        """
        # Create Qt application
        if PYQT_VERSION == 5:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Atlas Challenge-Based Learning")
        self.app.setOrganizationName("Atlas AI")
        self.app.setApplicationVersion("1.0.0")

        # Apply dark theme
        self.app.setStyleSheet(DARK_STYLESHEET)

        # Show splash screen while loading
        splash = self._create_splash_screen()
        splash.show()
        self.app.processEvents()

        try:
            # Initialize Atlas controller
            splash.showMessage("Initializing Atlas brain...",
                             Qt.AlignBottom | Qt.AlignHCenter if PYQT_VERSION == 5
                             else Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                             Qt.white if PYQT_VERSION == 5 else Qt.GlobalColor.white)
            self.app.processEvents()

            logger.info("Initializing Atlas controller...")
            self.controller = AtlasController(state_dim=128)

            # Create main window
            splash.showMessage("Creating interface...",
                             Qt.AlignBottom | Qt.AlignHCenter if PYQT_VERSION == 5
                             else Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                             Qt.white if PYQT_VERSION == 5 else Qt.GlobalColor.white)
            self.app.processEvents()

            logger.info("Creating main window...")
            self.main_window = AtlasMainWindow(self.controller)

            # Close splash and show main window
            splash.finish(self.main_window)
            self.main_window.show()

            logger.info("Atlas application started successfully")

            # Run event loop
            return self.app.exec_() if PYQT_VERSION == 5 else self.app.exec()

        except Exception as e:
            logger.error(f"Error starting application: {e}")
            splash.close()
            raise

        finally:
            # Cleanup - close() saves to database and closes connections
            if self.controller:
                self.controller.close()

    def _create_splash_screen(self) -> QSplashScreen:
        """Create a splash screen for loading."""
        # Create a simple colored pixmap for splash
        pixmap = QPixmap(400, 200)
        pixmap.fill(Qt.darkGray if PYQT_VERSION == 5 else Qt.GlobalColor.darkGray)

        splash = QSplashScreen(pixmap)
        splash.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint if PYQT_VERSION == 5
            else Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )

        # Add title text
        font = QFont("Arial", 20, QFont.Bold if PYQT_VERSION == 5 else QFont.Weight.Bold)

        return splash


def main():
    """Main entry point for the Atlas GUI application."""
    try:
        app = AtlasApplication()
        sys.exit(app.run())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
