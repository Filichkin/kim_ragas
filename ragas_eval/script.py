import asyncio

from langchain_community.document_loaders import (
    DirectoryLoader,
    PDFPlumberLoader
)
from langchain_openai import ChatOpenAI
import openai
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

from config import settings


def fix_themes_validation(original_init):
    """Патч для исправления проблемы с кортежами в themes."""
    def patched_init(self, *args, **kwargs):
        # Исправляем themes если они являются кортежами
        if 'themes' in kwargs and isinstance(kwargs['themes'], list):
            fixed_themes = []
            for theme in kwargs['themes']:
                if isinstance(theme, tuple):
                    # Берем первый элемент кортежа как строку
                    fixed_themes.append(str(theme[0]) if theme else '')
                else:
                    fixed_themes.append(str(theme))
            kwargs['themes'] = fixed_themes
        return original_init(self, *args, **kwargs)
    return patched_init


# Применяем патч к ThemesPersonasInput
try:
    from ragas.testset.synthesizers.multi_hop.specific import (
        ThemesPersonasInput
    )
    ThemesPersonasInput.__init__ = fix_themes_validation(
        ThemesPersonasInput.__init__
    )
except ImportError:
    pass


async def main():
    """Основная функция для генерации тестового датасета."""
    path = 'data/'
    loader = DirectoryLoader(
        path,
        glob='**/*.pdf',
        loader_cls=PDFPlumberLoader
    )

    docs = loader.load()

    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model='gpt-3.5-turbo',
            api_key=settings.OPENAI_API_KEY
        )
    )
    openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    generator_embeddings = OpenAIEmbeddings(
        client=openai_client
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=5
    )
    dataset.to_pandas().to_csv('ragas_dataset.csv', index=False)

    knowledge_graph = KnowledgeGraph()
    for doc in docs:
        knowledge_graph.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    'page_content': doc.page_content,
                    'document_metadata': doc.metadata
                }
            )
        )

    transformer_llm = generator_llm
    embedding_model = generator_embeddings

    trans = default_transforms(
        documents=docs,
        llm=transformer_llm,
        embedding_model=embedding_model
    )
    apply_transforms(knowledge_graph, trans)

    knowledge_graph.save('knowledge_graph.json')
    loaded_kg = KnowledgeGraph.load('knowledge_graph.json')
    print(loaded_kg)


if __name__ == '__main__':
    asyncio.run(main())
