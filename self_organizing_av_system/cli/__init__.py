"""
CLI module for Atlas System

Provides command-line interfaces for managing the Atlas system.
"""

from .checkpoint import main as checkpoint_main

__all__ = ['checkpoint_main']
