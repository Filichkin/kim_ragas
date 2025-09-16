from langchain_community.document_loaders import (
    DirectoryLoader,
    PDFPlumberLoader
)
from langchain_openai import ChatOpenAI
import openai
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from config import settings


path = 'data/'
loader = DirectoryLoader(
    path,
    glob='**/*.pdf',
    loader_cls=PDFPlumberLoader
)

docs = loader.load()

generator_llm = LangchainLLMWrapper(
    ChatOpenAI(model='gpt-4o')
    )
openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
generator_embeddings = OpenAIEmbeddings(
    client=openai_client
    )

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
    )
dataset = generator.generate_with_langchain_docs(
    docs,
    testset_size=10
    )
dataset.to_pandas().to_csv('ragas_dataset.csv', index=False)
