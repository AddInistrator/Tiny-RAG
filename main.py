from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from Retriever import RetrieveAgent, RetrieveGradeAgent
from Classifier import ClassifyAgent
from Generator import SimpleGenerateAgent, RetrieveGenerateAgent, GenerateAgent
from Reconstructor import QueryTransformAgent
from QwenLLM import QwenLLM


class GraphState(TypedDict):
    """
    GraphState是在Agent之间传递的状态
    :param question: 用户提出的问题
    :param generation: RAG预生成的内容
    :param result: 最终生成的绘本文字内容
    :param documents: 检索到的原始内容
    :param classifyResult: 分类器的分类结果
    """
    question: str
    generation: str
    result: str
    documents: List[str]
    classifyResult: str


if __name__ == '__main__':
    workflow = StateGraph(GraphState)

    workflow.add_node('classify', ClassifyAgent(model=QwenLLM()).query)
    workflow.add_node('simpleGenerate', SimpleGenerateAgent(model=QwenLLM()).query)
    workflow.add_node('retrieve', RetrieveAgent(model=QwenLLM()).query)
    workflow.add_node('retrieveGrade', RetrieveGradeAgent(model=QwenLLM()).query)
    workflow.add_node('retrieveGenerate', RetrieveGenerateAgent(model=QwenLLM()).query)
    workflow.add_node('generate', GenerateAgent(model=QwenLLM()).query)
    workflow.add_node('transformQuery', QueryTransformAgent(model=QwenLLM()).query)

    workflow.set_entry_point('classify')
    workflow.add_conditional_edges(
        'classify', lambda x: x['classifyResult'], {
            '0': 'simpleGenerate',
            '1': 'retrieve'
        }
    )
    workflow.add_edge('simpleGenerate', 'generate')
    workflow.add_edge('retrieve', 'retrieveGrade')
    workflow.add_conditional_edges(
        'retrieveGrade', lambda x: str(len(x['documents']) > 0), {
            'True': 'retrieveGenerate',
            'False': 'transformQuery'
        }
    )
    workflow.add_edge('transformQuery', 'retrieve')

    workflow.add_edge('retrieveGenerate', 'generate')
    workflow.add_edge('generate', END)

    app = workflow.compile()

    state = {'question': '太阳为什么能发光和热'}
    print('running')
    for output in app.stream(state):
        for key, value in output.items():
            print(value)
"""
    state = {'question': '<将这里替换成用户的问题>'}
    print('running')
    for output in app.stream(state):
        for key, value in output.items():
            print(value)
"""

