#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangChain + LlamaIndex + vLLM で RAG + モデル比較を行うデモスクリプト。

前提:
- vLLM の OpenAI互換サーバが、以下のように起動している想定:

    1) Full FT モデル:
       - base_url: http://localhost:8001/v1
       - model   : "ft_full_tinyllama"

    2) LoRA モデル（任意・使う場合）:
       - base_url: http://localhost:8002/v1
       - model   : "ft_lora_tinyllama"

    3) QLoRA モデル（任意・使う場合）:
       - base_url: http://localhost:8003/v1
       - model   : "ft_qlora_tinyllama"

Prefix Tuning は本スクリプトから完全に除外しています。

使い方の例:
    python app_rag_compare.py \\
        --docs_dir docs \\
        --question "Explain LoRA in the context of fine-tuning LLMs." \\
        --models ft_full,ft_lora,ft_qlora

必要に応じて MODEL_CONFIGS の base_url / model 名を環境に合わせて書き換えてください。
"""

import argparse
from typing import Dict, List

from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# モデルごとの vLLM エンドポイント設定（必要に応じて編集）
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "ft_full": {
        "model": "ft_full_tinyllama",
        "base_url": "http://localhost:8001/v1",
    },
    "ft_lora": {
        "model": "ft_lora_tinyllama",
        "base_url": "http://localhost:8002/v1",
    },
    "ft_qlora": {
        "model": "ft_qlora_tinyllama",
        "base_url": "http://localhost:8003/v1",
    },
}


def build_llamaindex_retriever(docs_dir: str, top_k: int = 3):
    """docs ディレクトリから LlamaIndex の Retriever を構築する。"""
    docs = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever


def retrieve_context_chunks(retriever, question: str) -> List[str]:
    """質問に対して top-k のコンテキストテキストを取得する。"""
    nodes = retriever.retrieve(question)
    return [n.text for n in nodes]


def build_llm(model_key: str, temperature: float = 0.2) -> ChatOpenAI:
    """
    MODEL_CONFIGS を元に ChatOpenAI を構築する。

    vLLM の OpenAI互換サーバに対して:
    - base_url: MODEL_CONFIGS[model_key]["base_url"]
    - model   : MODEL_CONFIGS[model_key]["model"]
    を指定して叩く。
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available keys: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_key]

    llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key="dummy-key",  # vLLM 側で認証を見ていなければ任意でよい
        temperature=temperature,
    )
    return llm


def build_messages(question: str, context_chunks: List[str]) -> List[dict]:
    """
    RAG 用プロンプトを OpenAI chat 形式の messages に変換する。
    """
    context_str = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

    system_msg = (
        "You are a helpful assistant.\n"
        "Use ONLY the given context to answer the user's question.\n"
        "If the answer is not contained in the context, say that you don't know.\n"
    )

    user_msg = (
        "Question:\n"
        f"{question}\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        "Please answer in a concise way."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG + model comparison demo (TinyLlama FT / LoRA / QLoRA)")

    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs",
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
        help=(
            "Comma-separated list of model keys to query. "
            "Available: " + ",".join(MODEL_CONFIGS.keys())
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # モデルキーのリスト化 & バリデーション
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for key in model_keys:
        if key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {key}. "
                f"Available keys: {list(MODEL_CONFIGS.keys())}"
            )

    print("=== RAG + Model Comparison ===")
    print(f"Docs dir : {args.docs_dir}")
    print(f"Question : {args.question}")
    print(f"Top-k    : {args.top_k}")
    print(f"Models   : {model_keys}")
    print()

    # 1. Retriever 構築 & コンテキスト取得
    retriever = build_llamaindex_retriever(args.docs_dir, top_k=args.top_k)
    context_chunks = retrieve_context_chunks(retriever, args.question)

    print("=== Retrieved Context (top-k) ===")
    for i, chunk in enumerate(context_chunks, start=1):
        preview = chunk.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[{i}] {preview}")
    print()

    # 2. 各モデルに対して同一プロンプトを投げる
    for key in model_keys:
        llm = build_llm(key)
        messages = build_messages(args.question, context_chunks)

        print("=" * 70)
        print(f"Model: {key} (id={MODEL_CONFIGS[key]['model']}, base_url={MODEL_CONFIGS[key]['base_url']})")
        print("-" * 70)

        resp = llm.invoke(messages)
        # langchain_openai.ChatOpenAI は `resp.content` に文字列が入っている想定
        print(resp.content)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
