from typing import List, Optional, Any
import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM(LLM):
    _instance = None

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, /, **data: Any):
        super().__init__(**data)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _llm_type(self) -> str:
        return 'Qwen2-0.5B-Instruct'
