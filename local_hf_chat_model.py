# local_hf_chat_model.py

from typing import List, Optional, Dict, Any
from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LocalHFChatModel(LLM):
    """
    LangChain-compatible wrapper for local HuggingFace models.
    """
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9

    @property
    def _llm_type(self) -> str:
        return "local_hf_chat"

    @classmethod
    def from_pretrained(cls, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
