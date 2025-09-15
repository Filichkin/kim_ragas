"""
Конфигурация для RAGAS Data Generator

Этот модуль содержит настройки для генерации данных RAGAS,
включая параметры разбиения на чанки, генерации вопросов и сценариев.
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


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
class ChunkingConfig:
    """Конфигурация для разбиения на чанки."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7  # Для семантического разбиения


@dataclass
class EmbeddingConfig:
    """Конфигурация для эмбедингов."""
    model_name: str = 'all-MiniLM-L6-v2'
    similarity_threshold: float = 0.7
    cache_embeddings: bool = True


@dataclass
class EntityConfig:
    """Конфигурация для извлечения сущностей."""
    spacy_model: str = 'ru_core_news_sm'
    fallback_model: str = 'en_core_web_sm'
    entity_types: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']


@dataclass
class QuestionGenerationConfig:
    """Конфигурация для генерации вопросов."""
    num_questions_per_scenario: int = 10
    question_type_distribution: Dict[str, float] = None
    question_style_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        if self.question_type_distribution is None:
            self.question_type_distribution = {
                'abstract': 0.2,
                'concrete': 0.3,
                'single_hop': 0.3,
                'multi_hop': 0.2
            }
        
        if self.question_style_distribution is None:
            self.question_style_distribution = {
                'formal': 0.3,
                'casual': 0.2,
                'technical': 0.3,
                'conversational': 0.2
            }


@dataclass
class ScenarioConfig:
    """Конфигурация для генерации сценариев."""
    num_scenarios: int = 50
    scenario_type_distribution: Dict[str, float] = None
    scenario_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.scenario_type_distribution is None:
            self.scenario_type_distribution = {
                'simple': 0.4,
                'medium': 0.4,
                'complex': 0.2
            }
        
        if self.scenario_configs is None:
            self.scenario_configs = {
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


@dataclass
class OutputConfig:
    """Конфигурация для вывода результатов."""
    output_format: str = 'json'  # json, csv, parquet
    include_embeddings: bool = False
    include_relationships: bool = True
    include_metadata: bool = True
    compress_output: bool = False


@dataclass
class RAGASConfig:
    """Основная конфигурация RAGAS генератора."""
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    entity: EntityConfig = None
    question_generation: QuestionGenerationConfig = None
    scenario: ScenarioConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.entity is None:
            self.entity = EntityConfig()
        if self.question_generation is None:
            self.question_generation = QuestionGenerationConfig()
        if self.scenario is None:
            self.scenario = ScenarioConfig()
        if self.output is None:
            self.output = OutputConfig()


# Предустановленные конфигурации
PRESET_CONFIGS = {
    'default': RAGASConfig(),
    
    'fast': RAGASConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300,
            chunk_overlap=30
        ),
        embedding=EmbeddingConfig(
            model_name='all-MiniLM-L6-v2'
        ),
        scenario=ScenarioConfig(
            num_scenarios=20,
            scenario_type_distribution={
                'simple': 0.6,
                'medium': 0.4
            }
        )
    ),
    
    'comprehensive': RAGASConfig(
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
            num_scenarios=100,
            scenario_type_distribution={
                'simple': 0.2,
                'medium': 0.5,
                'complex': 0.3
            }
        )
    ),
    
    'technical': RAGASConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_BASED,
            chunk_size=1000,
            chunk_overlap=100
        ),
        question_generation=QuestionGenerationConfig(
            question_type_distribution={
                'concrete': 0.4,
                'technical': 0.3,
                'single_hop': 0.2,
                'multi_hop': 0.1
            },
            question_style_distribution={
                'technical': 0.5,
                'formal': 0.3,
                'casual': 0.1,
                'conversational': 0.1
            }
        )
    )
}


def get_config(preset_name: str = 'default') -> RAGASConfig:
    """Получить предустановленную конфигурацию."""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f'Неизвестная предустановка: {preset_name}. '
                        f'Доступные: {list(PRESET_CONFIGS.keys())}')
    
    return PRESET_CONFIGS[preset_name]


def create_custom_config(**kwargs) -> RAGASConfig:
    """Создать пользовательскую конфигурацию."""
    config = RAGASConfig()
    
    # Обновляем конфигурацию переданными параметрами
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f'Неизвестный параметр конфигурации: {key}')
    
    return config
