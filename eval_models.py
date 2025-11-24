#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_models.py

Evaluate three models (ft_full, ft_lora, ft_qlora) on:
1. BERTScore-F1 (requires bert-score; optional)
2. Task accuracy (exact / relaxed match; always available if reference exists)
3. RAGAS faithfulness (requires ragas + LLM config; optional)

Input: JSONL file, each line:
{
  "id": "q1",
  "question": "...",
  "contexts": [
    {"doc_id": "doc1", "text": "..."},
    {"doc_id": "doc2", "text": "..."}
  ],
  "answers": {
    "ft_full": "...",
    "ft_lora": "...",
    "ft_qlora": "..."
  },
  "reference_answer": "...",        # optional
  "gold_doc_ids": ["doc1", "..."]   # optional (not used here directly)
}
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Optional dependencies（無ければ該当指標はスキップ）
try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness
    from datasets import Dataset as HFDataset
    _HAS_RAGAS = True
except Exception:
    _HAS_RAGAS = False


# プロジェクト内の命名に合わせて固定
MODEL_KEYS = ["ft_full", "ft_lora", "ft_qlora"]


@dataclass
class Example:
    qid: str
    question: str
    contexts: List[Dict[str, str]]
    answers: Dict[str, str]
    reference_answer: Optional[str]
    gold_doc_ids: List[str]


@dataclass
class ModelMetrics:
    bert_f1_scores: List[float]
    exact_matches: int
    relaxed_matches: int
    total_with_reference: int

    def __init__(self) -> None:
        self.bert_f1_scores = []
        self.exact_matches = 0
        self.relaxed_matches = 0
        self.total_with_reference = 0

    def to_summary(self) -> Dict[str, Any]:
        n_bert = len(self.bert_f1_scores)
        return {
            "bertscore_f1_mean": (sum(self.bert_f1_scores) / n_bert) if n_bert > 0 else None,
            "bertscore_f1_count": n_bert,
            "exact_accuracy": (
                self.exact_matches / self.total_with_reference
                if self.total_with_reference > 0
                else None
            ),
            "relaxed_accuracy": (
                self.relaxed_matches / self.total_with_reference
                if self.total_with_reference > 0
                else None
            ),
            "n_with_reference": self.total_with_reference,
        }


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
                question=obj["question"],
                contexts=obj.get("contexts", []),
                answers=obj["answers"],
                reference_answer=obj.get("reference_answer"),
                gold_doc_ids=obj.get("gold_doc_ids", []),
            )
            examples.append(ex)
    return examples


# ---------------- RAGAS utilities -----------------


def compute_ragas_faithfulness(
    examples: List[Example],
    per_model_answers: Dict[str, List[str]],
) -> Dict[str, Optional[float]]:

    if not _HAS_RAGAS:
        return {k: None for k in MODEL_KEYS}

    results: Dict[str, Optional[float]] = {}
    for model_key in MODEL_KEYS:
        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        gts: List[str] = []

        for ex, ans in zip(examples, per_model_answers[model_key]):
            if not ex.contexts:
                continue
            questions.append(ex.question)
            answers.append(ans)
            contexts.append([c["text"] for c in ex.contexts])
            gts.append(ex.reference_answer if ex.reference_answer else ans)

        if not questions:
            results[model_key] = None
            continue

        dataset = HFDataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": gts,
            }
        )
        ragas_res = ragas_evaluate(dataset, metrics=[ragas_faithfulness])
        # ragas の戻り値は Dataset 互換のオブジェクト（列名でアクセス）
        results[model_key] = float(ragas_res["faithfulness"])

    return results


# ---------------- BERTScore utilities -----------------


def compute_bertscore(refs: List[str], cands: List[str], lang: str = "en") -> List[float]:
    if not _HAS_BERTSCORE:
        return []
    if not refs:
        return []
    if len(refs) != len(cands):
        # 整合性チェック：長さがズレていたら短い方に合わせる
        n = min(len(refs), len(cands))
        refs = refs[:n]
        cands = cands[:n]
    P, R, F1 = bert_score(cands, refs, lang=lang)
    return [float(f) for f in F1]


# ---------------- Accuracy utilities -----------------


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ---------------- Main evaluation -----------------


def evaluate_models(
    examples: List[Example],
    bert_lang: str = "en",
    use_ragas: bool = False,
) -> Dict[str, Dict[str, Any]]:

    metrics = {k: ModelMetrics() for k in MODEL_KEYS}
    answers_per_model = {k: [] for k in MODEL_KEYS}
    # BERTScore 用に「参照ありのサンプルだけ」を別途保持（index ずれ防止）
    answers_for_bertscore = {k: [] for k in MODEL_KEYS}
    refs_for_bertscore = {k: [] for k in MODEL_KEYS}

    # Task accuracy + BERTScore
    for ex in examples:
        has_ref = ex.reference_answer is not None and ex.reference_answer != ""
        for key in MODEL_KEYS:
            ans = ex.answers.get(key, "")
            answers_per_model[key].append(ans)

            if has_ref:
                metrics[key].total_with_reference += 1
                ref = ex.reference_answer or ""
                if normalize_text(ans) == normalize_text(ref):
                    metrics[key].exact_matches += 1
                if (
                    normalize_text(ref)
                    and normalize_text(ref) in normalize_text(ans)
                ) or (
                    normalize_text(ans)
                    and normalize_text(ans) in normalize_text(ref)
                ):
                    metrics[key].relaxed_matches += 1

                refs_for_bertscore[key].append(ref)
                answers_for_bertscore[key].append(ans)

    # BERTScore
    if _HAS_BERTSCORE:
        for key in MODEL_KEYS:
            refs = refs_for_bertscore[key]
            cands = answers_for_bertscore[key]
            if refs and cands:
                metrics[key].bert_f1_scores.extend(
                    compute_bertscore(refs, cands, lang=bert_lang)
                )

    # RAGAS
    ragas_scores: Dict[str, Optional[float]] = {}
    if use_ragas:
        ragas_scores = compute_ragas_faithfulness(examples, answers_per_model)

    summary: Dict[str, Dict[str, Any]] = {}
    for key in MODEL_KEYS:
        base = metrics[key].to_summary()
        if use_ragas:
            base["ragas_faithfulness"] = ragas_scores.get(key)
        summary[key] = base

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ft_full / ft_lora / ft_qlora across metrics."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input JSONL path (e.g. data/sample_eval.jsonl)",
    )
    parser.add_argument(
        "--bert_lang",
        type=str,
        default="en",
        help="Language code for BERTScore (e.g. 'en', 'ja').",
    )
    parser.add_argument(
        "--use_ragas",
        action="store_true",
        help="If set, compute RAGAS faithfulness (requires ragas).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model_metrics.json",
        help="Where to save aggregated metrics JSON.",
    )
    args = parser.parse_args()

    examples = load_examples(args.data_path)
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    summary = evaluate_models(
        examples,
        bert_lang=args.bert_lang,
        use_ragas=args.use_ragas,
    )

    print("\n=== Model Metrics Summary ===")
    for key, m in summary.items():
        print(f"\n[{key}]")
        for name, val in m.items():
            print(f"  {name}: {val}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
