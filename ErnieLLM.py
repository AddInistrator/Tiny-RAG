import re
from typing import Optional, List, Any

import erniebot
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM


class ErnieLLM(LLM):
    _instance = None

    erniebot.api_type = 'aistudio'
    erniebot.access_token = '这里输入你的密钥'

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ErnieLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, /, **data: Any):
        super().__init__(**data)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response = erniebot.ChatCompletion.create(
            model='ernie-3.5',
            messages=[{'role': 'user', 'content': prompt}],
        ).get_result()
        return response

    @property
    def _llm_type(self) -> str:
        return 'ernie-3.5'
