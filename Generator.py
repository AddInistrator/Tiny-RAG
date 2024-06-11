from typing import Dict
from langchain_core.language_models import LLM
from BaseAgent import BaseAgent


class SimpleGenerateAgent(BaseAgent):
    """
    在不需要RAG的情况下调用，预生成内容
    """
    prompt = """
    回答用户给出的问题
    1. 确保你的回答科学简洁准确。
    2. 回答的长度控制在三句话左右，以传达核心科学概念。

    用户提出的问题是："<{question}>"
    """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        message = self.prompt.format(question=state['question'])
        response = self.model.invoke(message)
        state['generation'] = response
        return state


class RetrieveGenerateAgent(BaseAgent):
    """
    在需要RAG的情况下调用，预生成内容
    """
    prompt = """
    根据参考文档回答用户给出的问题
    1. 确保你的回答科学简洁准确，且严格按照参考文档。
    2. 回答的长度控制在三句话左右，以传达核心科学概念。

    用户提出的问题是：{question}
    参考文档是：{documents}
    """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        message = self.prompt.format(question=state['question'], documents=state['documents'])
        response = self.model.invoke(message)
        state['generation'] = response
        return state


class GenerateAgent(BaseAgent):
    """
    根据预生成内容生成绘本文字内容
    """
    prompt = """
    作为一位充满想象力的绘本作家，你的任务是创作出既科学准确又富有趣味性的绘本文字内容，以回答孩子们的好奇心。
    你将依据提供的一些文字来创作，这些文字多数是解释一个科学问题的，你需要改写成绘本文字内容。
    1. 创作一个简洁、有趣且易于理解的故事，能够在大约五句话内传达核心的科学概念。
    2. 故事应该富有创意，能够激发孩子们的想象力和好奇心。

    请根据以下内容，创作出绘本文字内容：{generation}
    """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        message = self.prompt.format(generation=state['generation'])
        response = self.model.invoke(message)
        state['result'] = response
        return state
