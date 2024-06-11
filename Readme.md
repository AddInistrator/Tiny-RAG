# Tiny RAG
安装环境：python 3.10

```shell
pip install -r requirement.txt
```
可能存在一些问题

之后直接运行main.py，记得填一个问题上去

---

文件结构
```text
RAG - BaseAgent: 所有Agent的基类，Agent是对LLM的封装、工作流中的节点
    |
    - ...LLM: 继承langchain LLM的模型，目前有通义千问和文心3.5
    |
    - VectorDatabase: 一个chromadb知识库（见resource文件夹）
    |
    - Classifier: 分类器，将用户的问题on demand retrieve
    |
    - Retriever: 检索器，检索知识库
    |
    - Generator: 生成器，生成最后的绘本内容
    |
    - Reconstructor: 用于判断、优化RAG生成内容
    |
    - main: 基于langgraph定义工作流的图结构
```

---

关于Agent: 所有的Agent基本都有简单注释

每个Agent都需要指定一个继承langchain LLM的LLM，例如：
```python
ClassifyAgent(model=QwenLLM())
```
每个Agent都有自己的prompt和query方法

---

关于工作流: 
```python
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
```
用户的提问在工作流中以如上的形式传递，必要情况下可以直接修改

---
目前的问题

- Classifier效果不佳，绝大多数问题都会被认为不需要RAG
- Generator最后生成的绘本内容相当抽象
- 还没有尝试文心的模型
- ……

to-do

- 优化prompt
- 优化工作流，细化Reconstructor
- 完成最后出选择题的Agent
- ……