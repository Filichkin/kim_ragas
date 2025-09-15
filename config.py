import os
from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # API configuration
    OPENROUTER_API_KEY: str
    OPENAI_API_KEY: str
    LLM_AGENT_MODEL: str = 'openai/gpt-oss-20b:free'
    LLM_API_BASE: str = 'https://openrouter.ai/api/v1'

    # Telegram bot
    BOT_TOKEN: str
    MAX_TELEGRAM_MESSAGE_LENGTH: int = 4000

    # MCP configuration
    MCP_URL: str = 'http://0.0.0.0:8001/sse'
    MCP_TRANSPORT: str = 'sse'
    MCP_RAG_TOOL_NAME: str = 'get_searched_data'

    # Agent configuration
    AGENT_SYSTEM_PROMPT: str
    TEMPERATURE: float = 0.7
    AGENT_MAX_TOKENS: int = 4000
    MAX_TOKENS: int = 2000

    # Milvus configuration
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_USER: str
    MILVUS_PASSWORD: str

    # Collection configuration
    COLLECTION_NAME: str = 'warranty_documents'
    CHUNK_SIZE: int = 500  # Chunk size in characters
    CHUNK_OVERLAP: int = 50  # Chunk overlap in characters
    DOCUMENT_ID_MAX_LENGTH: int = 256  # Max length for document IDs
    CHUNK_TEXT_MAX_LENGTH: int = 65535  # Max length for chunk text
    METADATA_MAX_LENGTH: int = 2048  # Max length for metadata field
    # DIMENSION: int = 3072  # Dimension for text-embedding-3-large
    DIMENSION: int = 2560  # Dimension for GigaChat EmbeddingsGigaR
    INDEX_TYPE: str = 'IVF_FLAT'
    METRIC_TYPE: str = 'COSINE'
    NLIST: int = 1024
    NPROBE: int = 64

    # Search configuration
    TOP_K: int = 50
    HYBRID_WEIGHT_VECTOR: float = 0.7
    HYBRID_WEIGHT_BM25: float = 0.3

    # Rerank configuration
    RERANK_ENABLED: bool = True
    RERANK_MIN_TEXT_LENGTH: int = 50  # Минимальная длина текста в символах
    RERANK_MIN_WORD_COUNT: int = 10   # Минимальное количество слов
    RERANK_LENGTH_WEIGHT: float = 0.3  # Вес длины текста в финальном скоре
    RERANK_SIMILARITY_WEIGHT: float = 0.7  # Вес схожести в финальном скоре

    # Advanced rerank configuration
    ADVANCED_RERANK_ENABLED: bool = True  # Включить продвинутый rerank
    # Количество кандидатов для rerank (50-200)
    RERANK_CANDIDATES_LIMIT: int = 50
    RERANK_FINAL_LIMIT: int = 10  # Финальное количество результатов

    # Cross-encoder configuration
    # Модель кросс-энкодера
    CROSS_ENCODER_MODEL: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    # Включить кросс-энкодер (требует установки)
    CROSS_ENCODER_ENABLED: bool = True

    # BM25 configuration
    BM25_ENABLED: bool = True  # Включить BM25 как fallback
    BM25_VECTOR_WEIGHT: float = 0.6  # Вес векторного скора
    BM25_WEIGHT: float = 0.4  # Вес BM25 скора
    BM25_K1: float = 1.2  # Параметр k1 для BM25
    BM25_B: float = 0.75  # Параметр b для BM25

    # Embedding model
    EMBEDDING_MODEL: str = 'text-embedding-3-large'

    # GigaChat configuration (перенесено из agent_config.py)
    GIGACHAT_CREDENTIALS: str
    GIGACHAT_SCOPE: str = 'GIGACHAT_API_PERS'
    GIGACHAT_MODEL: str = 'GigaChat'
    GIGACHAT_TEMPERATURE: float = 0.1
    GIGACHAT_VERIFY_SSL: bool = False

    # Streaming configuration (перенесено из agent_config.py)
    STREAM_EDIT_INTERVAL_SEC: float = 0.4
    STREAM_MIN_CHARS_DELTA: int = 48

    # Data paths
    DATA_DIR: str = './data'

    # Redis configuration (объединено с agent_config.py)
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600  # Время жизни кэша в секундах (1 час)

    # Admin configuration (перенесено из agent_config.py)
    ADMIN_IDS: Optional[str] = None  # Comma-separated admin IDs
    SUPPORT_URL: str  # Support URL

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.env'
        )
    )

    def get_admin_ids(self) -> List[int]:
        """Получить список ID администраторов"""
        if not self.ADMIN_IDS:
            return []
        try:
            return [
                int(id_str.strip())
                for id_str in self.ADMIN_IDS.split(',')
                if id_str.strip()
            ]
        except ValueError:
            raise RuntimeError(
                'ADMIN_IDS must contain comma-separated integers'
            )

    def is_admin(self, user_id: int) -> bool:
        """Проверить, является ли пользователь администратором"""
        return user_id in self.get_admin_ids()


settings = Config()
