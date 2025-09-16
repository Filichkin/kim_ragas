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
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import default_transforms, apply_transforms

from config import settings
from ragas_eval.patches import (
    apply_themes_patch,
    apply_question_potential_patch
)


# Применяем патчи к проблемным классам
apply_themes_patch()
apply_question_potential_patch()


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

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embedding_model,
        knowledge_graph=loaded_kg
        )

    query_distribution = default_query_distribution(generator_llm)

    testset = generator.generate(
        testset_size=5,
        query_distribution=query_distribution
        )
    testset.to_pandas().to_csv('ragas_testset.csv', index=False)


if __name__ == '__main__':
    asyncio.run(main())
