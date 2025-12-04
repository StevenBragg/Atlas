"""
PyQt5 GUI for ATLAS Challenge-Based Learning System

Provides a desktop application with two main areas:
1. Curriculum Learning (School) - Structured multimodal challenges
2. Free Play - Exploratory learning with webcam, mic, and chat

Atlas exists in both areas with shared knowledge base.
"""

from .app import main, AtlasApplication

__all__ = ['main', 'AtlasApplication']
