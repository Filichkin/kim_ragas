"""
Русские трансформы для Ragas Knowledge Graph.

Этот модуль содержит кастомные трансформы с русскими промптами
для генерации summary и других свойств в knowledge_graph.json.
"""

import typing as t

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms.base import LLMBasedExtractor
from ragas.testset.transforms.extractors import EmbeddingExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.testset.transforms.engine import Parallel
from ragas.utils import num_tokens_from_string

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.transforms.engine import Transforms

from langchain_core.documents import Document as LCDocument


class RussianSummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    """Промпт для извлечения краткого содержания на русском языке."""

    instruction: str = (
        'Создайте краткое содержание данного текста на русском языке '
        'в менее чем 10 предложениях. Сохраните ключевую информацию '
        'и основные идеи документа.'
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text=(
                    'Искусственный интеллект\n\n'
                    'Искусственный интеллект трансформирует различные отрасли, '
                    'автоматизируя задачи, которые ранее требовали '
                    'человеческого интеллекта. От здравоохранения до финансов, '
                    'ИИ используется '
                    'для быстрого и точного анализа больших объемов данных. '
                    'Эта технология также стимулирует инновации в таких '
                    'областях, как беспилотные автомобили и '
                    'персонализированные рекомендации.'
                )
            ),
            StringIO(
                text=(
                    'Искусственный интеллект революционизирует отрасли, '
                    'автоматизируя задачи, анализируя данные и стимулируя '
                    'инновации, такие как беспилотные автомобили и '
                    'персонализированные рекомендации.'
                )
            ),
        )
    ]


class RussianThemesExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    """Промпт для извлечения тем на русском языке."""

    instruction: str = (
        'Извлеките основные темы из данного текста на русском языке. '
        'Верните список тем, разделенных запятыми. Каждая тема должна '
        'быть краткой и описательной.'
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text=(
                    'Сервисная книжка автомобиля Kia содержит информацию '
                    'о гарантийных обязательствах, техническом обслуживании, '
                    'условиях эксплуатации и контактных данных дилеров.'
                )
            ),
            StringIO(text='гарантия, техническое обслуживание, эксплуатация, дилеры'),
        )
    ]


class RussianSummaryExtractor(LLMBasedExtractor):
    """
    Извлекает краткое содержание из текста на русском языке.

    Attributes
    ----------
    property_name : str
        Название свойства для извлечения.
    prompt : RussianSummaryExtractorPrompt
        Промпт, используемый для извлечения.
    """

    property_name: str = 'summary'
    prompt: RussianSummaryExtractorPrompt = RussianSummaryExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """Извлекает краткое содержание из узла."""
        node_text = node.get_property('page_content')
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(
            node_text, self.max_token_limit
        )
        result = await self.prompt.generate(
            self.llm, data=StringIO(text=chunks[0])
        )
        return self.property_name, result.text


class RussianThemesExtractor(LLMBasedExtractor):
    """
    Извлекает темы из текста на русском языке.

    Attributes
    ----------
    property_name : str
        Название свойства для извлечения.
    prompt : RussianThemesExtractorPrompt
        Промпт, используемый для извлечения.
    """

    property_name: str = 'themes'
    prompt: RussianThemesExtractorPrompt = RussianThemesExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """Извлекает темы из узла."""
        node_text = node.get_property('page_content')
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(
            node_text, self.max_token_limit
        )
        result = await self.prompt.generate(
            self.llm, data=StringIO(text=chunks[0])
        )
        # Разделяем темы по запятым и очищаем от пробелов
        themes = [theme.strip() for theme in result.text.split(',')]
        return self.property_name, themes


def russian_transforms(
    documents: t.List[LCDocument],
    llm: t.Any,
    embedding_model: t.Any,
) -> t.Any:
    """
    Создает и возвращает набор русских трансформов для обработки графа знаний.

    Эта функция определяет серию шагов трансформации для применения к графу
    знаний, включая извлечение кратких содержаний, тем, заголовков и эмбеддингов
    на русском языке, а также построение отношений сходства между узлами.

    Parameters
    ----------
    documents : List[LCDocument]
        Список документов для обработки.
    llm : BaseRagasLLM
        Языковая модель для генерации.
    embedding_model : BaseRagasEmbeddings
        Модель эмбеддингов.

    Returns
    -------
    Transforms
        Список шагов трансформации для применения к графу знаний.
    """

    def count_doc_length_bins(documents, bin_ranges):
        """Подсчитывает распределение документов по длине."""
        data = [num_tokens_from_string(doc.page_content) for doc in documents]
        bins = {f'{start}-{end}': 0 for start, end in bin_ranges}

        for num in data:
            for start, end in bin_ranges:
                if start <= num <= end:
                    bins[f'{start}-{end}'] += 1
                    break

        return bins

    def filter_doc_with_num_tokens(node, min_num_tokens=500):
        """Фильтрует документы по количеству токенов."""
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(
                node.properties['page_content']
            ) > min_num_tokens
        )

    def filter_docs(node):
        """Фильтрует документы."""
        return node.type == NodeType.DOCUMENT

    def filter_chunks(node):
        """Фильтрует чанки."""
        return node.type == NodeType.CHUNK

    bin_ranges = [(0, 100), (101, 500), (501, float('inf'))]
    result = count_doc_length_bins(documents, bin_ranges)
    result = {k: v / len(documents) for k, v in result.items()}

    transforms = []

    if result['501-inf'] >= 0.25:
        # Для длинных документов
        summary_extractor = RussianSummaryExtractor(
            llm=llm,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node)
        )
        theme_extractor = RussianThemesExtractor(
            llm=llm,
            filter_nodes=lambda node: filter_chunks(node)
        )
        ner_extractor = NERExtractor(
            llm=llm,
            filter_nodes=lambda node: filter_chunks(node)
        )

        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name='summary_embedding',
            embed_property_name='summary',
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        )

        cosine_sim_builder = CosineSimilarityBuilder(
            property_name='summary_embedding',
            new_property_name='summary_similarity',
            threshold=0.7,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        )

        ner_overlap_sim = OverlapScoreBuilder(
            threshold=0.01, filter_nodes=lambda node: filter_chunks(node)
        )

        node_filter = CustomNodeFilter(
            llm=llm, filter_nodes=lambda node: filter_chunks(node)
        )

        transforms = [
            summary_extractor,
            node_filter,
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    elif result['101-500'] >= 0.25:
        # Для документов средней длины
        summary_extractor = RussianSummaryExtractor(
            llm=llm,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100)
        )
        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name='summary_embedding',
            embed_property_name='summary',
            filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
        )

        cosine_sim_builder = CosineSimilarityBuilder(
            property_name='summary_embedding',
            new_property_name='summary_similarity',
            threshold=0.5,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
        )

        ner_extractor = NERExtractor(llm=llm)
        ner_overlap_sim = OverlapScoreBuilder(threshold=0.01)
        theme_extractor = RussianThemesExtractor(
            llm=llm,
            filter_nodes=lambda node: filter_docs(node)
        )
        node_filter = CustomNodeFilter(llm=llm)

        transforms = [
            summary_extractor,
            node_filter,
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    else:
        raise ValueError(
            'Документы кажутся слишком короткими (100 токенов или меньше). '
            'Пожалуйста, предоставьте более длинные документы.'
        )

    return transforms
