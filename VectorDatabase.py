from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-small")
        self.database = Chroma(persist_directory='resources', embedding_function=self.embedding)

    def retrieve(self, text: str, k: int = 5) -> List[str]:
        return [x.page_content for x in self.database.similarity_search(text, k=k)]
