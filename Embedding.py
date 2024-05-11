from sentence_transformers import SentenceTransformer
from typing import List


class BaseEmbeddings:

    def __init__(self) -> None:
        pass

    def embedding(self, text: str) -> List[float]:
        raise NotImplementedError


class MokaiEmbedding(BaseEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_length = 512
        self.model = SentenceTransformer("moka-ai/m3e-small")

    def embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return [float(x) for x in embedding]
