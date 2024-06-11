from typing import Dict
from langchain_core.language_models import LLM


class BaseAgent:

    def __init__(self, model: LLM):
        """
        所有Agent的基类
        :param model: Agent使用的模型，模型应当继承langchain的LLM类
        """
        self.model = model

    def query(self, state: Dict):
        raise NotImplementedError
