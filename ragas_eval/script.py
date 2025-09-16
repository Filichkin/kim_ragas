"""
Генератор тестового датасета на русском языке с использованием Ragas.

Этот скрипт генерирует вопросы и ответы на русском языке из PDF документов
с использованием библиотеки Ragas и адаптированных промптов.
"""

import asyncio
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PDFPlumberLoader,
)
from langchain_openai import ChatOpenAI
import openai
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from config import settings
from ragas_eval.logger_config import setup_simple_logger, get_logger

# Настраиваем логирование
setup_simple_logger()
logger = get_logger()

# Применяем патчи к проблемным классам (опционально)
try:
    from ragas_eval.patches import (
        apply_themes_patch,
        apply_question_potential_patch,
    )
    apply_themes_patch()
    apply_question_potential_patch()
    logger.info('Патчи Ragas успешно применены')
except ImportError:
    # Патчи опциональны; если модуль отсутствует — просто продолжаем
    logger.warning('Патчи Ragas не найдены, продолжаем без них')


async def main():
    """
    Основная функция для генерации тестового датасета на русском языке.

    Загружает PDF документы, создает синтезатор с русскими промптами,
    генерирует вопросы и ответы, сохраняет результат в CSV файл.
    """
    logger.info('Начинаем генерацию тестового датасета на русском языке')

    # 1) Загружаем PDF документы
    data_dir = Path(settings.DATA_DIR)
    if not data_dir.exists():
        error_msg = f'Папка с данными не найдена: {data_dir}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f'Загружаем PDF документы из {data_dir}')
    loader = DirectoryLoader(
        data_dir,
        glob='**/*.pdf',
        loader_cls=PDFPlumberLoader,
    )
    docs = loader.load()

    if not docs:
        error_msg = f'В {data_dir} не найдено PDF-файлов.'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f'Загружено {len(docs)} документов')

    # 2) Инициализируем LLM и эмбеддинги
    if not settings.OPENAI_API_KEY:
        error_msg = 'OPENAI_API_KEY не задан в config.settings или env.'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info('Инициализируем LLM и эмбеддинги')
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    )
    openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    generator_embeddings = OpenAIEmbeddings(client=openai_client)

    # 3) Создаем Single-hop синтезатор
    logger.info('Создаем Single-hop синтезатор')
    specific = SingleHopSpecificQuerySynthesizer(llm=generator_llm)

    # 4) Адаптируем промпты для русского языка
    logger.info('Адаптируем промпты для русского языка')
    if hasattr(specific, 'adapt_prompts'):
        prompts = await specific.adapt_prompts('russian', llm=generator_llm)
        specific.set_prompts(**prompts)
        logger.info('Промпты успешно адаптированы для русского языка')
    else:
        logger.warning('Метод adapt_prompts не найден в синтезаторе')

    # 5) Настраиваем распределение (только один синтезатор)
    query_distribution = [(specific, 1.0)]

    # 6) Генерируем тестовый набор из документов
    logger.info('Генерируем тестовый набор из документов')
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=settings.TESTSET_SIZE,
        query_distribution=query_distribution,
    )

    # 7) Сохраняем результат
    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / settings.OUTPUT_FILENAME

    logger.info(f'Сохраняем результат в {output_file}')
    df = dataset.to_pandas()
    df.to_csv(output_file, index=False)

    logger.success(f'Готово! Записей: {len(df)}. Файл: {output_file}')


if __name__ == '__main__':
    asyncio.run(main())
