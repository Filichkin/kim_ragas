"""
Утилиты для проекта RAGAS.

Этот модуль содержит вспомогательные функции для работы с проектом.
"""

from datetime import datetime


def generate_timestamp() -> str:
    """
    Генерирует timestamp в формате YYYYMMDD_HHMMSS.

    Returns:
        str: Строка с текущей датой и временем в формате YYYYMMDD_HHMMSS

    Example:
        >>> generate_timestamp()
        '20250917_155313'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
