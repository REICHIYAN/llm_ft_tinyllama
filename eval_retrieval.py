#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_retrieval.py

Retrieval のみを評価するシンプル版。
外部LLMやRAGASなどは一切使わず、JSONLの内容だけから以下を計算します。

- recall@k
- precision@k
- hit_rate@k（トップkに1つでも正解docが含まれるか）

【前提となるJSONLのフォーマット（1行1サンプル）】

{
  "id": "q1",
  "question": "...",
  "contexts": [
    {"doc_id": "doc1", "text": "..."},
    {"doc_id": "doc2", "text": "..."},
    ...
  ],
  "answers": {
    "ft_full": "...",
    "ft_lora": "...",
    "ft_qlora": "..."
  },
  "reference_answer": "...",        # optional
  "gold_doc_ids": ["doc1", "doc3"]  # 正解とみなすdoc_id
}

ここでは、`contexts` の並び順が「retrievalのランキング順」
（先頭が最も関連度が高い）になっていると仮定します。
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Example:
    qid: str
    question: str
    # contexts: ranked list of retrieved documents
    contexts: List[Dict[str, str]]
    gold_doc_ids: List[str]


def load_examples(path: str) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id", "")
            ex = Example(
                qid=qid,
                question=obj.get("question", ""),
                contexts=obj.get("contexts", []),
                gold_doc_ids=obj.get("gold_doc_ids", []),
            )
            examples.append(ex)
    return examples


def compute_retrieval_metrics(
    examples: List[Example],
    ks: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    各 k について、
      - recall@k
      - precision@k
      - hit_rate@k
    を計算して返す。
    """
    # 集計用
    results: Dict[int, Dict[str, float]] = {}

    for k in ks:
        total_recall = 0.0
        total_precision = 0.0
        total_hit = 0.0
        count_with_gold = 0  # gold_doc_ids が空でないサンプル数
        total_examples = 0

        for ex in examples:
            total_examples += 1
            gold = [g for g in ex.gold_doc_ids if g]  # 空文字などは除外
            retrieved_ids = [c.get("doc_id") for c in ex.contexts[:k] if c.get("doc_id")]

            if not gold:
                # 正解docが定義されていない場合は、Recall/Precisionの分母なし
                # → hit_rateだけ集計対象にする
                hit = 1.0 if any(rid in gold for rid in retrieved_ids) else 0.0
                total_hit += hit
                continue

            count_with_gold += 1

            # ヒット数
            hits = sum(1 for rid in retrieved_ids if rid in gold)
            # recall@k = (# 正解docのうち、トップkに入ったもの) / (# 正解doc)
            recall = hits / len(gold) if gold else 0.0
            # precision@k = (# 正解docのうち、トップkに入ったもの) / k
            precision = hits / float(k) if k > 0 else 0.0
            # hit@k = 1 if any 正解doc がトップkに含まれる else 0
            hit = 1.0 if hits > 0 else 0.0

            total_recall += recall
            total_precision += precision
            total_hit += hit

        # 平均を計算
        avg_recall = (total_recall / count_with_gold) if count_with_gold > 0 else None
        avg_precision = (total_precision / count_with_gold) if count_with_gold > 0 else None
        avg_hit = (total_hit / total_examples) if total_examples > 0 else None

        results[k] = {
            "recall@k": avg_recall,
            "precision@k": avg_precision,
            "hit_rate@k": avg_hit,
            "num_examples_with_gold": count_with_gold,
            "num_examples_total": total_examples,
        }

    return results


def parse_k_list(k_list_str: Optional[str]) -> List[int]:
    """
    "1,3,5" のような文字列を [1, 3, 5] に変換。
    None の場合はデフォルト [1, 3, 5] を返す。
    """
    if not k_list_str:
        return [1, 3, 5]
    ks: List[int] = []
    for part in k_list_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
            if k > 0:
                ks.append(k)
        except ValueError:
            continue
    if not ks:
        ks = [1, 3, 5]
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality (recall@k / precision@k / hit_rate@k) "
                    "based on contexts ranking and gold_doc_ids."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input JSONL path (e.g. data/sample_eval.jsonl)",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,3,5",
        help="Comma-separated list of k values (e.g. '1,3,5'). Default: '1,3,5'",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="retrieval_metrics.json",
        help="Where to save aggregated metrics JSON.",
    )
    args = parser.parse_args()

    examples = load_examples(args.data_path)
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    ks = parse_k_list(args.k_list)
    print(f"Evaluating ks = {ks}")

    metrics = compute_retrieval_metrics(examples, ks)

    print("\n=== Retrieval Metrics Summary ===")
    for k in ks:
        m = metrics[k]
        print(f"\n[k={k}]")
        print(f"  recall@{k}: {m['recall@k']}")
        print(f"  precision@{k}: {m['precision@k']}")
        print(f"  hit_rate@{k}: {m['hit_rate@k']}")
        print(f"  num_examples_with_gold: {m['num_examples_with_gold']}")
        print(f"  num_examples_total: {m['num_examples_total']}")

    # JSONとして保存
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved retrieval metrics to {args.output_path}")


if __name__ == "__main__":
    main()
