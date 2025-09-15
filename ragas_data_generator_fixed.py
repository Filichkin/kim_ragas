"""
RAGAS Data Generator Script

Этот скрипт генерирует данные для оценки RAG систем с использованием
библиотеки RAGAS. Включает разбиение документов на чанки, формирование
связей между ними, генерацию различных типов вопросов и создание сценариев.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter
)
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from config import settings


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Стратегии разбиения документов на чанки."""
    RECURSIVE = 'recursive'
    TOKEN_BASED = 'token_based'
    NLTK_BASED = 'nltk_based'
    SEMANTIC = 'semantic'


class QuestionType(Enum):
    """Типы вопросов."""
    ABSTRACT = 'abstract'
    CONCRETE = 'concrete'
    SINGLE_HOP = 'single_hop'
    MULTI_HOP = 'multi_hop'


class QuestionStyle(Enum):
    """Стили написания вопросов."""
    FORMAL = 'formal'
    CASUAL = 'casual'
    TECHNICAL = 'technical'
    CONVERSATIONAL = 'conversational'


@dataclass
class ChunkMetadata:
    """Метаданные чанка."""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    char_count: int
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """Чанк документа с метаданными."""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[np.ndarray] = None


@dataclass
class QuestionScenario:
    """Сценарий вопроса."""
    question: str
    question_type: QuestionType
    question_style: QuestionStyle
    expected_answer: str
    relevant_chunks: List[str]
    context_chunks: List[str]
    difficulty_score: float


class ChunkingStrategyBase(ABC):
    """Базовый класс для стратегий разбиения на чанки."""

    @abstractmethod
    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Разбить документ на чанки."""
        pass


class RecursiveChunkingStrategy(ChunkingStrategyBase):
    """Стратегия рекурсивного разбиения на чанки."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Разбить документ рекурсивно."""
        chunks = self.splitter.split_documents([document])
        result = []

        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f'{document.metadata.get("source", "doc")}_{i}',
                document_id=document.metadata.get('source', 'doc'),
                chunk_index=i,
                start_char=0,  # Будет обновлено при необходимости
                end_char=len(chunk.page_content),
                word_count=len(chunk.page_content.split()),
                char_count=len(chunk.page_content)
            )

            result.append(DocumentChunk(
                content=chunk.page_content,
                metadata=metadata
            ))

        return result


class TokenBasedChunkingStrategy(ChunkingStrategyBase):
    """Стратегия разбиения на основе токенов."""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Разбить документ на основе токенов."""
        chunks = self.splitter.split_documents([document])
        result = []

        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f'{document.metadata.get("source", "doc")}_{i}',
                document_id=document.metadata.get('source', 'doc'),
                chunk_index=i,
                start_char=0,
                end_char=len(chunk.page_content),
                word_count=len(chunk.page_content.split()),
                char_count=len(chunk.page_content)
            )

            result.append(DocumentChunk(
                content=chunk.page_content,
                metadata=metadata
            ))

        return result


class NLTKChunkingStrategy(ChunkingStrategyBase):
    """Стратегия разбиения на основе NLTK."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Разбить документ с использованием NLTK."""
        chunks = self.splitter.split_documents([document])
        result = []

        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f'{document.metadata.get("source", "doc")}_{i}',
                document_id=document.metadata.get('source', 'doc'),
                chunk_index=i,
                start_char=0,
                end_char=len(chunk.page_content),
                word_count=len(chunk.page_content.split()),
                char_count=len(chunk.page_content)
            )

            result.append(DocumentChunk(
                content=chunk.page_content,
                metadata=metadata
            ))

        return result


class SemanticChunkingStrategy(ChunkingStrategyBase):
    """Стратегия семантического разбиения на чанки."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Разбить документ семантически."""
        sentences = document.page_content.split('. ')
        if len(sentences) < 2:
            # Если предложений мало, используем обычное разбиение
            return self._fallback_chunking(document)

        # Вычисляем TF-IDF векторы для предложений
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Группируем предложения по семантической близости
        chunks = self._group_sentences_by_similarity(
            sentences, similarity_matrix
        )

        result = []
        for i, chunk_text in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f'{document.metadata.get("source", "doc")}_{i}',
                document_id=document.metadata.get('source', 'doc'),
                chunk_index=i,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            )

            result.append(DocumentChunk(
                content=chunk_text,
                metadata=metadata
            ))

        return result

    def _fallback_chunking(self, document: Document) -> List[DocumentChunk]:
        """Резервное разбиение для коротких документов."""
        fallback_strategy = RecursiveChunkingStrategy()
        return fallback_strategy.split_document(document)

    def _group_sentences_by_similarity(
        self,
        sentences: List[str],
        similarity_matrix: np.ndarray
    ) -> List[str]:
        """Группировать предложения по семантической близости."""
        chunks = []
        used_sentences = set()

        for i, sentence in enumerate(sentences):
            if i in used_sentences:
                continue

            chunk_sentences = [sentence]
            used_sentences.add(i)

            # Ищем похожие предложения
            for j in range(i + 1, len(sentences)):
                if (j not in used_sentences and
                    similarity_matrix[i, j] > self.similarity_threshold):
                    chunk_sentences.append(sentences[j])
                    used_sentences.add(j)

            chunks.append('. '.join(chunk_sentences))

        return chunks


class ChunkingStrategyFactory:
    """Фабрика для создания стратегий разбиения на чанки."""

    @staticmethod
    def create_strategy(
        strategy_type: ChunkingStrategy,
        **kwargs
    ) -> ChunkingStrategyBase:
        """Создать стратегию разбиения на чанки."""
        strategies = {
            ChunkingStrategy.RECURSIVE: RecursiveChunkingStrategy,
            ChunkingStrategy.TOKEN_BASED: TokenBasedChunkingStrategy,
            ChunkingStrategy.NLTK_BASED: NLTKChunkingStrategy,
            ChunkingStrategy.SEMANTIC: SemanticChunkingStrategy
        }

        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f'Неизвестная стратегия: {strategy_type}')

        return strategy_class(**kwargs)


class DocumentProcessor:
    """Процессор документов для разбиения на чанки."""

    def __init__(self, strategy: ChunkingStrategyBase):
        self.strategy = strategy
        self.nlp = None
        self._load_spacy_model()

    def _load_spacy_model(self):
        """Загрузить модель spaCy для извлечения сущностей."""
        try:
            self.nlp = spacy.load('ru_core_news_sm')
        except OSError:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning('Модель spaCy не найдена. '
                              'Извлечение сущностей отключено.')
                self.nlp = None

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Обработать документ и разбить на чанки."""
        logger.info(f'Обработка документа: {file_path}')

        # Загрузка документа
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError(f'Неподдерживаемый формат файла: {file_path}')

        all_chunks = []
        for doc in documents:
            chunks = self.strategy.split_document(doc)
            # Извлекаем сущности для каждого чанка
            for chunk in chunks:
                self._extract_entities(chunk)
            all_chunks.extend(chunks)

        logger.info(f'Создано {len(all_chunks)} чанков')
        return all_chunks

    def _extract_entities(self, chunk: DocumentChunk):
        """Извлечь сущности из чанка."""
        if not self.nlp:
            return

        try:
            doc = self.nlp(chunk.content)
            entities = [ent.text for ent in doc.ents if ent.label_ in
                       ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']]
            chunk.metadata.entities = entities
        except Exception as e:
            logger.warning(f'Ошибка извлечения сущностей: {e}')


class EmbeddingCalculator:
    """Калькулятор эмбедингов для чанков."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}

    def calculate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Вычислить эмбединги для всех чанков."""
        logger.info('Вычисление эмбедингов...')

        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.embeddings_cache[chunk.metadata.chunk_id] = embedding

    def calculate_similarity_matrix(
        self,
        chunks: List[DocumentChunk]
    ) -> np.ndarray:
        """Вычислить матрицу схожести между чанками."""
        if not chunks or chunks[0].embedding is None:
            raise ValueError('Эмбединги не вычислены')

        embeddings = np.array([chunk.embedding for chunk in chunks])
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix


class RelationshipAnalyzer:
    """Анализатор связей между документами и чанками."""

    def __init__(self, embedding_calculator: EmbeddingCalculator):
        self.embedding_calculator = embedding_calculator

    def analyze_embedding_relationships(
        self,
        chunks: List[DocumentChunk]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Анализировать связи на основе эмбедингов."""
        logger.info('Анализ связей на основе эмбедингов...')

        similarity_matrix = self.embedding_calculator.calculate_similarity_matrix(
            chunks
        )

        relationships = defaultdict(list)
        threshold = 0.7  # Порог схожести

        for i, chunk_i in enumerate(chunks):
            for j, chunk_j in enumerate(chunks):
                if i != j and similarity_matrix[i, j] > threshold:
                    relationships[chunk_i.metadata.chunk_id].append(
                        (chunk_j.metadata.chunk_id, similarity_matrix[i, j])
                    )

        return dict(relationships)

    def analyze_entity_overlap(
        self,
        chunks: List[DocumentChunk]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Анализировать перекрытие сущностей между чанками."""
        logger.info('Анализ перекрытия сущностей...')

        relationships = defaultdict(list)

        for i, chunk_i in enumerate(chunks):
            entities_i = set(chunk_i.metadata.entities)
            if not entities_i:
                continue

            for j, chunk_j in enumerate(chunks):
                if i == j:
                    continue

                entities_j = set(chunk_j.metadata.entities)
                if not entities_j:
                    continue

                # Вычисляем коэффициент Жаккара
                intersection = len(entities_i & entities_j)
                union = len(entities_i | entities_j)

                if union > 0:
                    jaccard_coefficient = intersection / union
                    if jaccard_coefficient > 0.1:  # Порог перекрытия
                        relationships[chunk_i.metadata.chunk_id].append(
                            (chunk_j.metadata.chunk_id, jaccard_coefficient)
                        )

        return dict(relationships)


class QuestionGenerator:
    """Генератор вопросов различных типов и стилей."""

    def __init__(self):
        self.question_templates = self._load_question_templates()

    def _load_question_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Загрузить шаблоны вопросов."""
        return {
            'abstract': {
                'formal': [
                    'Каковы основные принципы, изложенные в документе?',
                    'Какова общая концепция, представленная в тексте?',
                    'Какие фундаментальные идеи можно выделить?'
                ],
                'casual': [
                    'О чем в целом идет речь в документе?',
                    'Какая основная мысль?',
                    'Что самое важное здесь написано?'
                ],
                'technical': [
                    'Какие технические принципы описаны?',
                    'Какова архитектурная концепция?',
                    'Какие методологические подходы применяются?'
                ],
                'conversational': [
                    'Расскажи, что здесь описывается?',
                    'Можешь объяснить основную идею?',
                    'Что интересного в этом документе?'
                ]
            },
            'concrete': {
                'formal': [
                    'Какие конкретные данные представлены?',
                    'Каковы точные значения параметров?',
                    'Какие специфические детали указаны?'
                ],
                'casual': [
                    'Какие конкретные цифры есть?',
                    'Что точно написано про...?',
                    'Какие детали можно найти?'
                ],
                'technical': [
                    'Какие технические характеристики указаны?',
                    'Каковы параметры системы?',
                    'Какие спецификации приведены?'
                ],
                'conversational': [
                    'А что конкретно там написано?',
                    'Какие точные данные есть?',
                    'Можешь привести конкретные примеры?'
                ]
            },
            'single_hop': {
                'formal': [
                    'Что говорится о [ENTITY] в документе?',
                    'Какая информация представлена относительно [ENTITY]?',
                    'Как описывается [ENTITY]?'
                ],
                'casual': [
                    'Что написано про [ENTITY]?',
                    'Какая информация о [ENTITY]?',
                    'Как описывается [ENTITY]?'
                ],
                'technical': [
                    'Какие технические характеристики [ENTITY]?',
                    'Как [ENTITY] функционирует?',
                    'Каковы параметры [ENTITY]?'
                ],
                'conversational': [
                    'А что про [ENTITY]?',
                    'Расскажи про [ENTITY]',
                    'Как там с [ENTITY]?'
                ]
            },
            'multi_hop': {
                'formal': [
                    'Как [ENTITY1] связано с [ENTITY2]?',
                    'Какая связь между [ENTITY1] и [ENTITY2]?',
                    'Как [ENTITY1] влияет на [ENTITY2]?'
                ],
                'casual': [
                    'Как [ENTITY1] и [ENTITY2] связаны?',
                    'Что общего у [ENTITY1] и [ENTITY2]?',
                    'Как одно влияет на другое?'
                ],
                'technical': [
                    'Какая техническая связь между [ENTITY1] и [ENTITY2]?',
                    'Как [ENTITY1] взаимодействует с [ENTITY2]?',
                    'Каковы зависимости между [ENTITY1] и [ENTITY2]?'
                ],
                'conversational': [
                    'А как [ENTITY1] с [ENTITY2] связано?',
                    'Расскажи про связь между [ENTITY1] и [ENTITY2]',
                    'Что между ними общего?'
                ]
            }
        }

    def generate_questions(
        self,
        chunks: List[DocumentChunk],
        num_questions: int = 100,
        distribution: Optional[Dict[str, float]] = None
    ) -> List[QuestionScenario]:
        """Генерировать вопросы различных типов."""
        if distribution is None:
            distribution = {
                'abstract': 0.2,
                'concrete': 0.3,
                'single_hop': 0.3,
                'multi_hop': 0.2
            }

        questions = []
        entities = self._extract_all_entities(chunks)

        for question_type, ratio in distribution.items():
            num_type_questions = int(num_questions * ratio)
            type_questions = self._generate_questions_of_type(
                question_type, chunks, entities, num_type_questions
            )
            questions.extend(type_questions)

        return questions

    def _extract_all_entities(self, chunks: List[DocumentChunk]) -> List[str]:
        """Извлечь все уникальные сущности из чанков."""
        all_entities = set()
        for chunk in chunks:
            all_entities.update(chunk.metadata.entities)
        return list(all_entities)

    def _generate_questions_of_type(
        self,
        question_type: str,
        chunks: List[DocumentChunk],
        entities: List[str],
        num_questions: int
    ) -> List[QuestionScenario]:
        """Генерировать вопросы определенного типа."""
        questions = []
        templates = self.question_templates.get(question_type, {})

        for _ in range(num_questions):
            # Выбираем случайный стиль
            style = np.random.choice(list(templates.keys()))
            template = np.random.choice(templates[style])

            # Выбираем случайные чанки для контекста
            context_chunks = np.random.choice(
                chunks,
                size=min(3, len(chunks)),
                replace=False
            )

            # Заменяем плейсхолдеры сущностями
            question_text = self._fill_template(template, entities)

            # Генерируем ожидаемый ответ
            expected_answer = self._generate_expected_answer(
                question_text, context_chunks
            )

            # Определяем релевантные чанки
            relevant_chunks = self._find_relevant_chunks(
                question_text, chunks
            )

            scenario = QuestionScenario(
                question=question_text,
                question_type=QuestionType(question_type),
                question_style=QuestionStyle(style),
                expected_answer=expected_answer,
                relevant_chunks=[c.metadata.chunk_id for c in relevant_chunks],
                context_chunks=[c.metadata.chunk_id for c in context_chunks],
                difficulty_score=self._calculate_difficulty_score(
                    question_type, len(relevant_chunks)
                )
            )

            questions.append(scenario)

        return questions

    def _fill_template(self, template: str, entities: List[str]) -> str:
        """Заполнить шаблон сущностями."""
        if '[ENTITY]' in template:
            entity = np.random.choice(entities) if entities else 'данные'
            return template.replace('[ENTITY]', entity)
        elif '[ENTITY1]' in template and '[ENTITY2]' in template:
            if len(entities) >= 2:
                entity1, entity2 = np.random.choice(entities, 2, replace=False)
                return template.replace('[ENTITY1]', entity1).replace(
                    '[ENTITY2]', entity2
                )
            else:
                return template.replace('[ENTITY1]', 'элемент1').replace(
                    '[ENTITY2]', 'элемент2'
                )
        return template

    def _generate_expected_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk]
    ) -> str:
        """Генерировать ожидаемый ответ на основе контекста."""
        # Простая реализация - объединяем содержимое релевантных чанков
        answer_parts = []
        for chunk in context_chunks:
            if len(chunk.content) > 100:
                # Берем первые 100 символов
                answer_parts.append(chunk.content[:100] + '...')
            else:
                answer_parts.append(chunk.content)

        return ' '.join(answer_parts)

    def _find_relevant_chunks(
        self,
        question: str,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Найти релевантные чанки для вопроса."""
        # Простая реализация - возвращаем случайные чанки
        # В реальной реализации здесь должен быть семантический поиск
        num_relevant = min(2, len(chunks))
        return np.random.choice(chunks, size=num_relevant, replace=False)

    def _calculate_difficulty_score(
        self,
        question_type: str,
        num_relevant_chunks: int
    ) -> float:
        """Вычислить оценку сложности вопроса."""
        base_scores = {
            'abstract': 0.3,
            'concrete': 0.5,
            'single_hop': 0.7,
            'multi_hop': 0.9
        }

        base_score = base_scores.get(question_type, 0.5)
        chunk_factor = min(0.2, num_relevant_chunks * 0.1)

        return min(1.0, base_score + chunk_factor)


class ScenarioGenerator:
    """Генератор сценариев с заданным распределением."""

    def __init__(self, question_generator: QuestionGenerator):
        self.question_generator = question_generator

    def generate_scenarios(
        self,
        chunks: List[DocumentChunk],
        num_scenarios: int = 50,
        scenario_distribution: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Генерировать сценарии с заданным распределением."""
        if scenario_distribution is None:
            scenario_distribution = {
                'simple': 0.4,
                'medium': 0.4,
                'complex': 0.2
            }

        scenarios = []

        for scenario_type, ratio in scenario_distribution.items():
            num_type_scenarios = int(num_scenarios * ratio)
            type_scenarios = self._generate_scenarios_of_type(
                scenario_type, chunks, num_type_scenarios
            )
            scenarios.extend(type_scenarios)

        return scenarios

    def _generate_scenarios_of_type(
        self,
        scenario_type: str,
        chunks: List[DocumentChunk],
        num_scenarios: int
    ) -> List[Dict[str, Any]]:
        """Генерировать сценарии определенного типа."""
        scenarios = []

        # Определяем параметры для каждого типа сценария
        scenario_configs = {
            'simple': {
                'num_questions': 5,
                'question_distribution': {
                    'concrete': 0.6,
                    'single_hop': 0.4
                }
            },
            'medium': {
                'num_questions': 10,
                'question_distribution': {
                    'concrete': 0.3,
                    'single_hop': 0.4,
                    'abstract': 0.3
                }
            },
            'complex': {
                'num_questions': 15,
                'question_distribution': {
                    'abstract': 0.2,
                    'concrete': 0.2,
                    'single_hop': 0.3,
                    'multi_hop': 0.3
                }
            }
        }

        config = scenario_configs.get(scenario_type, scenario_configs['medium'])

        for i in range(num_scenarios):
            questions = self.question_generator.generate_questions(
                chunks,
                num_questions=config['num_questions'],
                distribution=config['question_distribution']
            )

            scenario = {
                'scenario_id': f'{scenario_type}_{i}',
                'scenario_type': scenario_type,
                'questions': [
                    {
                        'question': q.question,
                        'question_type': q.question_type.value,
                        'question_style': q.question_style.value,
                        'expected_answer': q.expected_answer,
                        'relevant_chunks': q.relevant_chunks,
                        'context_chunks': q.context_chunks,
                        'difficulty_score': q.difficulty_score
                    }
                    for q in questions
                ],
                'metadata': {
                    'total_questions': len(questions),
                    'avg_difficulty': np.mean([q.difficulty_score for q in questions]),
                    'chunk_coverage': len(set(
                        chunk_id for q in questions
                        for chunk_id in q.relevant_chunks
                    ))
                }
            }

            scenarios.append(scenario)

        return scenarios


class RAGASDataGenerator:
    """Основной класс для генерации данных RAGAS."""

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        self.chunking_strategy = ChunkingStrategyFactory.create_strategy(
            chunking_strategy
        )
        self.document_processor = DocumentProcessor(self.chunking_strategy)
        self.embedding_calculator = EmbeddingCalculator(embedding_model)
        self.relationship_analyzer = RelationshipAnalyzer(
            self.embedding_calculator
        )
        self.question_generator = QuestionGenerator()
        self.scenario_generator = ScenarioGenerator(self.question_generator)

    def generate_dataset(
        self,
        document_paths: List[str],
        output_path: str,
        num_scenarios: int = 50,
        scenario_distribution: Optional[Dict[str, float]] = None
    ) -> None:
        """Генерировать полный датасет для RAGAS."""
        logger.info('Начало генерации датасета RAGAS...')

        # 1. Обработка документов и разбиение на чанки
        all_chunks = []
        for doc_path in document_paths:
            chunks = self.document_processor.process_document(doc_path)
            all_chunks.extend(chunks)

        logger.info(f'Всего создано {len(all_chunks)} чанков')

        # 2. Вычисление эмбедингов
        self.embedding_calculator.calculate_embeddings(all_chunks)

        # 3. Анализ связей
        embedding_relationships = self.relationship_analyzer.analyze_embedding_relationships(all_chunks)
        entity_relationships = self.relationship_analyzer.analyze_entity_overlap(all_chunks)

        # 4. Генерация сценариев
        scenarios = self.scenario_generator.generate_scenarios(
            all_chunks,
            num_scenarios=num_scenarios,
            scenario_distribution=scenario_distribution
        )

        # 5. Сохранение результатов
        self._save_dataset(
            output_path,
            all_chunks,
            embedding_relationships,
            entity_relationships,
            scenarios
        )

        logger.info(f'Датасет сохранен в {output_path}')

    def _save_dataset(
        self,
        output_path: str,
        chunks: List[DocumentChunk],
        embedding_relationships: Dict[str, List[Tuple[str, float]]],
        entity_relationships: Dict[str, List[Tuple[str, float]]],
        scenarios: List[Dict[str, Any]]
    ) -> None:
        """Сохранить датасет в файл."""
        dataset = {
            'chunks': [
                {
                    'chunk_id': chunk.metadata.chunk_id,
                    'document_id': chunk.metadata.document_id,
                    'content': chunk.content,
                    'metadata': {
                        'chunk_index': chunk.metadata.chunk_index,
                        'word_count': chunk.metadata.word_count,
                        'char_count': chunk.metadata.char_count,
                        'entities': chunk.metadata.entities
                    }
                }
                for chunk in chunks
            ],
            'relationships': {
                'embedding_similarity': embedding_relationships,
                'entity_overlap': entity_relationships
            },
            'scenarios': scenarios,
            'metadata': {
                'total_chunks': len(chunks),
                'total_scenarios': len(scenarios),
                'total_questions': sum(
                    len(s['questions']) for s in scenarios
                )
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


def main():
    """Основная функция для запуска генератора."""
    # Настройка путей
    data_dir = Path(settings.DATA_DIR)
    document_paths = list(data_dir.glob('*.pdf'))

    if not document_paths:
        logger.error(f'Документы не найдены в {data_dir}')
        return

    # Создание генератора
    generator = RAGASDataGenerator(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_model='all-MiniLM-L6-v2'
    )

    # Генерация датасета
    output_path = data_dir / 'ragas_dataset.json'
    generator.generate_dataset(
        document_paths=[str(p) for p in document_paths],
        output_path=str(output_path),
        num_scenarios=50,
        scenario_distribution={
            'simple': 0.4,
            'medium': 0.4,
            'complex': 0.2
        }
    )

    logger.info('Генерация завершена успешно!')


if __name__ == '__main__':
    main()
