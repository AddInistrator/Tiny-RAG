from typing import Dict

from langchain_core.language_models import LLM

from BaseAgent import BaseAgent
from VectorDatabase import Database


class RetrieveAgent(BaseAgent):
    prompt = "用1句话回答这个问题：{question}"

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        response = Database().retrieve(
            f"{state['question']}?{self.model.invoke(self.prompt.format(question=state['question']))}"
        )
        state['documents'] = response
        return state


class RetrieveGradeAgent(BaseAgent):
    prompt = """
    作为评分员，你需要评估检索到的文档与用户问题的相关性。
    - 如果文档包含与用户问题相关的关键词或语义意义，评分为“相关”。
    - 如果文档与用户问题不相关，则评分为“不相关”。
    - 输出结果只能是'1'或'0'，其中：
      - '1'表示文档与用户问题相关。
      - '0'表示文档与用户问题不相关。
    
    检索到的文档: {documents}     
    用户的问题: {question} 
     """

    def __init__(self, model: LLM) -> None:
        super().__init__(model)

    def query(self, state: Dict) -> Dict:
        filtered_documents = []
        for context in state['documents']:
            message = self.prompt.format(question=state['question'], documents=state['documents'])
            response = self.model.invoke(message)
            if response == '1':
                filtered_documents.append(context)
        state['documents'] = filtered_documents
        return state
