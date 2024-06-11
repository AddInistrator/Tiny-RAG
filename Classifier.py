from typing import Dict
from langchain_core.language_models import LLM

from BaseAgent import BaseAgent


class ClassifyAgent(BaseAgent):
    """
    分类器，用于判断用户提出的问题是否需要RAG
    """
    prompt = """
    判断用户提出的问题是否需要调用知识库和网页搜索。

    1. 如果用户的问题对科学性要求较高，涉及具体的概念，需要调用知识库和网页搜索。
    2. 如果需要富有想象力的回答，并且对正确性要求不高，可以直接回答。
    
    输出结果只能是'1'或'0'，其中：
    - '1'表示需要调用知识库和网页搜索。
    - '0'表示可以直接回答。
    
    用户的问题是：{question}
    """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        message = self.prompt.format(question=state['question'])
        response = self.model.invoke(message)
        state['classifyResult'] = response
        return state