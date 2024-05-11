import requests

from VectorDatabase import ChromaDatabase


class ChatGPT:
    def __init__(self):
        self.API_KEY = 'app-cavbHdzerauR1kWw5sSexQ5v'
        self.db = ChromaDatabase()

    def query(self, question: str) -> str | Exception:

        RAG_PROMPT_TEMPLATE = f"""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
            问题: {question}
            可参考的上下文：
            ···
            {self.db.query(question, 10)}
            ···
            如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
            有用的回答:
        """

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        # 设置请求体数据
        data = {
            "inputs": {},
            "query": RAG_PROMPT_TEMPLATE,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "abc-123"
        }
        response = requests.post('https://api.dify.ai/v1/chat-messages', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['answer']
        else:
            raise Exception(response.status_code)
