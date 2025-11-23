#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG + Model Comparison (LangChain + local TinyLlama, no vLLM, no OpenAI)

- Retrieval: LlamaIndex + HuggingFaceEmbedding (BAAI/bge-small-en-v1.5)
- Prompting: LangChain PromptTemplate
- Generation: local TinyLlama (Full-FT / LoRA / QLoRA) via LocalHFChatModel
"""

import argparse
import gc
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from langchain_core.prompts import PromptTemplate

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from local_hf_chat_model import LocalHFChatModel


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIRS: Dict[str, str] = {
    "ft_full": os.path.join(REPO_ROOT, "models", "ft_full"),
    "ft_lora": os.path.join(REPO_ROOT, "models", "ft_lora"),
    "ft_qlora": os.path.join(REPO_ROOT, "models", "ft_qlora"),
}

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
#  RAG: Retriever (LlamaIndex + local embedding)
# ----------------------------------------------------------------------


def build_llamaindex_retriever(docs_dir: str, top_k: int = 3):
    """Build a LlamaIndex retriever using a local HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    docs = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever


def retrieve_context_chunks(retriever, question: str) -> List[str]:
    """Retrieve top-k context chunks for a question."""
    nodes = retriever.retrieve(question)
    return [n.text for n in nodes]


# ----------------------------------------------------------------------
#  Local models: loader + generation
# ----------------------------------------------------------------------


def load_full_model() -> LocalHFChatModel:
    """Load full fine-tuned TinyLlama model (ft_full) as LangChain LLM."""
    model_path = MODEL_DIRS["ft_full"]
    return LocalHFChatModel.from_pretrained(model_path)


def load_peft_model(key: str) -> LocalHFChatModel:
    """
    Load TinyLlama base + PEFT adapter (LoRA / QLoRA) as LangChain LLM.

    key: "ft_lora" or "ft_qlora"
    """
    if key not in ("ft_lora", "ft_qlora"):
        raise ValueError(f"load_peft_model: invalid key={key}")

    adapter_path = MODEL_DIRS[key]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    base_model.eval()

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    # LangChain LLM wrapper
    wrapper = LocalHFChatModel(
        model=peft_model,
        tokenizer=tokenizer,
    )
    return wrapper


def clear_model_from_memory(model):
    """Utility to release GPU/CPU memory between models."""
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
#  Prompt for RAG (LangChain)
# ----------------------------------------------------------------------


def build_prompt() -> PromptTemplate:
    template = """You are a helpful assistant.
Use ONLY the context below to answer the user's question.
If the answer is not in the context, say that you don't know.

Question:
{question}

Context:
{context}

Answer:
"""
    return PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG + model comparison demo (TinyLlama FT / LoRA / QLoRA, local inference)"
    )

    parser.add_argument(
        "--docs_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "docs"),
        help="Directory containing source documents for RAG.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG system.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ft_full",
        help="Comma-separated list of model keys to query. "
        "Available: ft_full,ft_lora,ft_qlora",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------


def main():
    args = parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for key in model_keys:
        if key not in MODEL_DIRS:
            raise ValueError(
                f"Unknown model key: {key}. "
                f"Available keys: {list(MODEL_DIRS.keys())}"
            )

    print("=== RAG + Model Comparison (LangChain + local HF) ===")
    print(f"Docs dir : {args.docs_dir}")
    print(f"Question : {args.question}")
    print(f"Top-k    : {args.top_k}")
    print(f"Models   : {model_keys}")
    print(f"Device   : {DEVICE}")
    print()

    # 1. RAG: retrieve context
    retriever = build_llamaindex_retriever(args.docs_dir, top_k=args.top_k)
    context_chunks = retrieve_context_chunks(retriever, args.question)

    print("=== Retrieved Context (top-k) ===")
    for i, chunk in enumerate(context_chunks, start=1):
        preview = chunk.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[{i}] {preview}")
    print()

    context_text = "\n\n".join(context_chunks)
    prompt = build_prompt()

    # 2. For each model, load → generate → unload
    for key in model_keys:
        print("=" * 70)
        print(f"Model: {key}  (path={MODEL_DIRS[key]})")
        print("-" * 70)

        if key == "ft_full":
            llm = load_full_model()
        else:
            llm = load_peft_model(key)

        filled_prompt = prompt.format(
            question=args.question,
            context=context_text,
        )
        # LangChain LLM: invoke() で1つのテキストを生成
        answer = llm.invoke(filled_prompt)
        print(answer)
        print()

        clear_model_from_memory(llm)

    print("Done.")


if __name__ == "__main__":
    main()
