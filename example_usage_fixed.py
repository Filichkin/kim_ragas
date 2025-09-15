"""
Пример использования RAGAS Data Generator

Этот скрипт демонстрирует различные способы использования
генератора данных RAGAS для создания датасетов оценки.
"""

import logging
from pathlib import Path
from ragas_data_generator import (
    RAGASDataGenerator,
    ChunkingStrategy
)
from ragas_config import (
    ChunkingConfig,
    EmbeddingConfig,
    ScenarioConfig,
    get_config,
    create_custom_config
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Базовый пример использования генератора."""
    logger.info('=== Базовый пример использования ===')

    # Пути к документам
    data_dir = Path('./data')
    document_paths = list(data_dir.glob('*.pdf'))

    if not document_paths:
        logger.error('Документы не найдены в папке data/')
        return

    # Создание генератора с настройками по умолчанию
    generator = RAGASDataGenerator(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_model='all-MiniLM-L6-v2'
    )

    # Генерация датасета
    output_path = data_dir / 'basic_dataset.json'
    generator.generate_dataset(
        document_paths=[str(p) for p in document_paths],
        output_path=str(output_path),
        num_scenarios=20
    )

    logger.info(f'Базовый датасет сохранен в {output_path}')


def example_custom_config():
    """Пример использования с пользовательской конфигурацией."""
    logger.info('=== Пример с пользовательской конфигурацией ===')

    # Создание пользовательской конфигурации
    custom_config = create_custom_config(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=800,
            chunk_overlap=100,
            similarity_threshold=0.8
        ),
        embedding=EmbeddingConfig(
            model_name='all-mpnet-base-v2',
            similarity_threshold=0.8
        ),
        scenario=ScenarioConfig(
            num_scenarios=30,
            scenario_type_distribution={
                'simple': 0.3,
                'medium': 0.5,
                'complex': 0.2
            }
        )
    )

    # Применение конфигурации (в реальном коде это будет интегрировано)
    logger.info('Пользовательская конфигурация создана')
    logger.info(f'Стратегия разбиения: {custom_config.chunking.strategy.value}')
    logger.info(f'Модель эмбедингов: {custom_config.embedding.model_name}')
    logger.info(f'Количество сценариев: {custom_config.scenario.num_scenarios}')


def example_preset_configs():
    """Пример использования предустановленных конфигураций."""
    logger.info('=== Пример с предустановленными конфигурациями ===')

    # Доступные предустановки
    presets = ['default', 'fast', 'comprehensive', 'technical']

    for preset_name in presets:
        config = get_config(preset_name)
        logger.info(f'Конфигурация "{preset_name}":')
        logger.info(f'  - Стратегия разбиения: {config.chunking.strategy.value}')
        logger.info(f'  - Размер чанка: {config.chunking.chunk_size}')
        logger.info(f'  - Модель эмбедингов: {config.embedding.model_name}')
        logger.info(f'  - Количество сценариев: {config.scenario.num_scenarios}')


def example_different_chunking_strategies():
    """Пример использования различных стратегий разбиения."""
    logger.info('=== Пример с различными стратегиями разбиения ===')

    strategies = [
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.TOKEN_BASED,
        ChunkingStrategy.NLTK_BASED,
        ChunkingStrategy.SEMANTIC
    ]

    data_dir = Path('./data')
    document_paths = list(data_dir.glob('*.pdf'))

    if not document_paths:
        logger.error('Документы не найдены в папке data/')
        return

    for strategy in strategies:
        logger.info(f'Тестирование стратегии: {strategy.value}')

        try:
            generator = RAGASDataGenerator(
                chunking_strategy=strategy,
                embedding_model='all-MiniLM-L6-v2'
            )

            output_path = data_dir / f'dataset_{strategy.value}.json'
            generator.generate_dataset(
                document_paths=[str(p) for p in document_paths],
                output_path=str(output_path),
                num_scenarios=10
            )

            logger.info(f'Датасет для {strategy.value} сохранен в {output_path}')

        except Exception as e:
            logger.error(f'Ошибка при использовании стратегии {strategy.value}: {e}')


def example_question_analysis():
    """Пример анализа сгенерированных вопросов."""
    logger.info('=== Анализ сгенерированных вопросов ===')

    # В реальном коде здесь будет загрузка и анализ сгенерированного датасета
    logger.info('Анализ типов вопросов:')
    logger.info('  - Абстрактные: вопросы о общих принципах и концепциях')
    logger.info('  - Конкретные: вопросы о точных данных и фактах')
    logger.info('  - Single-hop: вопросы, требующие информацию из одного источника')
    logger.info('  - Multi-hop: вопросы, требующие информацию из нескольких источников')

    logger.info('Анализ стилей вопросов:')
    logger.info('  - Формальный: академический стиль')
    logger.info('  - Неформальный: повседневный стиль')
    logger.info('  - Технический: специализированная терминология')
    logger.info('  - Разговорный: диалоговый стиль')


def example_scenario_distribution():
    """Пример настройки распределения сценариев."""
    logger.info('=== Настройка распределения сценариев ===')

    # Различные распределения сценариев
    distributions = {
        'balanced': {
            'simple': 0.33,
            'medium': 0.34,
            'complex': 0.33
        },
        'beginner_friendly': {
            'simple': 0.6,
            'medium': 0.3,
            'complex': 0.1
        },
        'expert_level': {
            'simple': 0.1,
            'medium': 0.3,
            'complex': 0.6
        },
        'mixed': {
            'simple': 0.4,
            'medium': 0.4,
            'complex': 0.2
        }
    }

    for dist_name, distribution in distributions.items():
        logger.info(f'Распределение "{dist_name}":')
        for scenario_type, ratio in distribution.items():
            logger.info(f'  - {scenario_type}: {ratio:.1%}')


def example_batch_processing():
    """Пример пакетной обработки документов."""
    logger.info('=== Пакетная обработка документов ===')

    # Список папок с документами
    document_folders = [
        './data/legal_docs',
        './data/technical_docs',
        './data/business_docs'
    ]

    # Создание генератора
    generator = RAGASDataGenerator(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_model='all-MiniLM-L6-v2'
    )

    for folder_path in document_folders:
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f'Папка {folder_path} не существует')
            continue

        # Поиск документов в папке
        document_paths = list(folder.glob('*.pdf'))
        if not document_paths:
            logger.warning(f'PDF документы не найдены в {folder_path}')
            continue

        logger.info(f'Обработка {len(document_paths)} документов в {folder_path}')

        # Генерация датасета для папки
        output_path = folder / f'{folder.name}_dataset.json'
        generator.generate_dataset(
            document_paths=[str(p) for p in document_paths],
            output_path=str(output_path),
            num_scenarios=15
        )

        logger.info(f'Датасет для {folder.name} сохранен в {output_path}')


def main():
    """Основная функция с примерами использования."""
    logger.info('Запуск примеров использования RAGAS Data Generator')

    try:
        # Базовый пример
        example_basic_usage()

        # Пользовательская конфигурация
        example_custom_config()

        # Предустановленные конфигурации
        example_preset_configs()

        # Различные стратегии разбиения
        example_different_chunking_strategies()

        # Анализ вопросов
        example_question_analysis()

        # Распределение сценариев
        example_scenario_distribution()

        # Пакетная обработка
        example_batch_processing()

        logger.info('Все примеры выполнены успешно!')

    except Exception as e:
        logger.error(f'Ошибка при выполнении примеров: {e}')


if __name__ == '__main__':
    main()
