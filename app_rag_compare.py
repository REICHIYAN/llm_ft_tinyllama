#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG + Model Comparison (LangChain, local HF models)

- Retrieval → LangChain
- Embedding → HuggingFace (BAAI/bge-small)
- LLM → Local TinyLlama (Full-FT / LoRA / QLoRA)
- No vLLM
- No OpenAI API
"""

import argparse
import os
from typing import List
import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# LangChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 自作 LangChain LLM wrapper
from local_hf_chat_model import LocalHFChatModel


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = {
    "ft_full": os.path.join(REPO_ROOT, "models", "ft_full"),
    "ft_lora": os.path.join(REPO_ROOT, "models", "ft_lora"),
    "ft_qlora": os.path.join(REPO_ROOT, "models", "ft_qlora"),
}

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
#  Loaders
# ----------------------------------------------------------------------

def load_full_model():
    return LocalHFChatModel.from_pretrained(MODEL_DIRS["ft_full"])


def load_peft_model(key: str):
    adapter_path = MODEL_DIRS[key]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    # LangChain wrapper
    wrapper = LocalHFChatModel(model=model, tokenizer=tokenizer)
    return wrapper


def clear(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
#  Build Retriever (LangChain)
# ----------------------------------------------------------------------

def build_retriever(docs_dir: str, top_k: int):
    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    texts = []
    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    vectordb = FAISS.from_texts(texts, embedding=embed)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    return retriever


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--docs_dir", default="docs")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--models", default="ft_full,ft_lora,ft_qlora")
    args = parser.parse_args()

    model_keys: List[str] = [m.strip() for m in args.models.split(",")]

    print("=== RAG + Model Comparison (LangChain + Local HF) ===")
    print(f"Docs     : {args.docs_dir}")
    print(f"Question : {args.question}")
    print(f"Models   : {model_keys}")
    print(f"Device   : {DEVICE}")
    print()

    # Build retriever
    retriever = build_retriever(args.docs_dir, args.top_k)

    # Prompt
    template = """Use ONLY the context below to answer.

Question: {question}

Context:
{context}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )

    def combine_context(docs):
        return "\n\n".join([d.page_content for d in docs])

    for key in model_keys:
        print("=" * 70)
        print(f"Model: {key}")
        print("-" * 70)

        if key == "ft_full":
            llm = load_full_model()
        else:
            llm = load_peft_model(key)

        rag_chain = (
            {
                "context": retriever | combine_context,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        answer = rag_chain.invoke(args.question)
        print(answer)
        print()

        clear(llm)

    print("Done.")


if __name__ == "__main__":
    main()
