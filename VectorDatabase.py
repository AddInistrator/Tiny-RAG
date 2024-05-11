import chromadb
from typing import List
from utils import ReadFiles

from Embedding import MokaiEmbedding


class BaseVectorDatabase:
    def __init__(self) -> None:
        self.embedding = MokaiEmbedding()


class ChromaDatabase(BaseVectorDatabase):
    def __init__(self, name: str = 'baseDatabase') -> None:
        super().__init__()
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

        # 这里要改！！！！！！！！！！！！！！！！！！！！1
        self.append(ReadFiles(path='').get_content(max_token_len=400, cover_content=100))

    def append(self, text: List[str]) -> None:
        embeddings = [self.embedding.embedding(x) for x in text]
        self.collection.add(
            documents=text,
            embeddings=embeddings,
            ids=[str(x) for x in range(len(text))],
        )

    def query(self, text: str, n: int = 1) -> List[str] | None:
        results = self.collection.query(
            query_embeddings=self.embedding.embedding(text),
            n_results=n
        )
        return results['documents'][0]
