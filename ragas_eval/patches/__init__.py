"""Патчи для исправления проблем в библиотеке RAGAS."""

from .themes_patch import apply_themes_patch
from .question_potential_patch import apply_question_potential_patch

__all__ = [
    'apply_themes_patch',
    'apply_question_potential_patch'
]
