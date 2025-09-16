"""
Конфигурация проекта RAGAS.

Этот модуль содержит только необходимые настройки для генерации
тестовых датасетов с использованием библиотеки Ragas.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Конфигурация для проекта RAGAS."""

    # API configuration
    OPENAI_API_KEY: str

    # Model configuration
    LLM_MODEL: str = 'gpt-3.5-turbo'

    # Data configuration
    DATA_DIR: str = './data'
    TESTSET_SIZE: int = 10
    OUTPUT_DIR: str = './output'
    OUTPUT_FILENAME: str = 'ragas_testset_singlehop_ru.csv'

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '.env'
        ),
        extra='ignore'  # Игнорируем дополнительные поля из .env
    )


settings = Config()
