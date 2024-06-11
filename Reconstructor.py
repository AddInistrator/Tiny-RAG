from typing import Dict
from langchain_core.language_models import LLM
from BaseAgent import BaseAgent


class QueryTransformAgent(BaseAgent):
    """
    用于在RAG路径中检索效果差的时候改写问题
    """
    prompt = """
    生成一个优化后的检索问题，以提高该问题被检索到的概率。

    - 审视并理解原始问题的潜在语义意图或含义。
    - 根据原始问题的内容，生成一个更加精确、明确且易于检索的问题。    

    原始的问题：<{question}>
    """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        message = self.prompt.format(question=state['question'])
        response = self.model.invoke(message)
        state['question'] = response
        return state
