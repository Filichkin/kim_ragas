import asyncio
from datetime import datetime
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
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import apply_transforms

from config import settings
from ragas_eval.logger_config import setup_simple_logger, get_logger
from ragas_eval.russian_transforms import russian_transforms


setup_simple_logger()
logger = get_logger()


try:
    from ragas_eval.patches import (
        apply_themes_patch,
        apply_question_potential_patch,
    )
    apply_themes_patch()
    apply_question_potential_patch()
    logger.info('Патчи Ragas успешно применены')
except ImportError:
    logger.warning('Патчи Ragas не найдены, продолжаем без них')


async def main():
    """
    Основная функция для генерации тестового датасета на русском языке.

    Загружает PDF документы, создает синтезаторы для разных типов вопросов
    (single-hop и multi-hop) с русскими промптами, генерирует вопросы и ответы,
    сохраняет результат в CSV файл.
    """
    logger.info('Начинаем генерацию тестового датасета на русском языке')

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

    logger.info('Создаем синтезаторы для разных типов вопросов')
    single_hop = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    multi_hop = MultiHopSpecificQuerySynthesizer(llm=generator_llm)

    logger.info('Адаптируем промпты для русского языка')
    synthesizers = [single_hop, multi_hop]

    for synthesizer in synthesizers:
        if hasattr(synthesizer, 'adapt_prompts'):
            prompts = await synthesizer.adapt_prompts(
                'russian', llm=generator_llm
            )
            synthesizer.set_prompts(**prompts)
            logger.info(
                f'Промпты для {synthesizer.__class__.__name__} адаптированы'
            )
        else:
            logger.warning(
                f'Метод adapt_prompts не найден в '
                f'{synthesizer.__class__.__name__}'
            )

    # Распределение: 70% single-hop, 30% multi-hop вопросов
    # Уменьшаем долю multi-hop для избежания дублирования
    query_distribution = [
        (single_hop, 0.7),
        (multi_hop, 0.3),
    ]

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

    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Генерируем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = settings.OUTPUT_FILENAME.replace('.csv', '')
    output_filename = f'{base_filename}_{timestamp}.csv'
    output_file = output_dir / output_filename

    logger.info(f'Сохраняем результат в {output_file}')
    df = dataset.to_pandas()

    # Анализируем дублирование вопросов
    question_counts = df['user_input'].value_counts()
    duplicates = question_counts[question_counts > 1]

    if len(duplicates) > 0:
        logger.warning(f'Найдено {len(duplicates)} дублирующихся вопросов:')
        for question, count in duplicates.items():
            logger.warning(
                f'  - "{question[:50]}..." (повторяется {count} раз)'
            )
    else:
        logger.info('Дублирующихся вопросов не найдено')

    # Анализируем распределение по синтезаторам
    synthesizer_counts = df['synthesizer_name'].value_counts()
    logger.info('Распределение по синтезаторам:')
    for synth, count in synthesizer_counts.items():
        logger.info(f'  - {synth}: {count} вопросов')

    df.to_csv(output_file, index=False)
    logger.success(f'Готово! Записей: {len(df)}. Файл: {output_file}')

    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                    }
            )
        )
    transformer_llm = generator_llm
    embedding_model = generator_embeddings

    logger.info('Создаем русские трансформы для графа знаний')
    trans = russian_transforms(
        documents=docs,
        llm=transformer_llm,
        embedding_model=embedding_model
    )
    apply_transforms(kg, trans)
    kg.save('knowledge_graph.json')
    loaded_kg = KnowledgeGraph.load('knowledge_graph.json')
    logger.info(loaded_kg)

if __name__ == '__main__':
    asyncio.run(main())
