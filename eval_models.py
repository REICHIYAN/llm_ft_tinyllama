#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_models.py

Evaluate three models (ft_full, ft_lora, ft_qlora) on:
1. LLM-as-a-Judge score (requires OpenAI-compatible LLM; optional)
2. BERTScore-F1 (requires bert-score; optional)
3. Task accuracy (exact / relaxed match; always available if reference exists)
4. RAGAS faithfulness (requires ragas + LLM config; optional)
5. Hallucination rate (requires judge LLM; optional)

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
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Optional dependencies
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

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


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
    judge_scores: List[float]
    hallucination_flags: List[bool]
    bert_f1_scores: List[float]
    exact_matches: int
    relaxed_matches: int
    total_with_reference: int

    def __init__(self) -> None:
        self.judge_scores = []
        self.hallucination_flags = []
        self.bert_f1_scores = []
        self.exact_matches = 0
        self.relaxed_matches = 0
        self.total_with_reference = 0

    def to_summary(self) -> Dict[str, Any]:
        n_judge = len(self.judge_scores)
        n_bert = len(self.bert_f1_scores)
        n_hall = len(self.hallucination_flags)
        return {
            "judge_score_mean": (sum(self.judge_scores) / n_judge) if n_judge > 0 else None,
            "judge_score_count": n_judge,
            "hallucination_rate": (sum(self.hallucination_flags) / n_hall) if n_hall > 0 else None,
            "hallucination_count": n_hall,
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


# ---------------- LLM-as-a-Judge utilities -----------------

def call_judge_llm_openai(
    question: str,
    context: str,
    answers: Dict[str, str],
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:

    prompt = (
        "You are an impartial evaluator.\n"
        "Given a question, supporting context, and three model answers, "
        "rate each answer from 0 to 10 and mark whether it contains hallucinations.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "scores": {"ft_full": 0-10, "ft_lora": 0-10, "ft_qlora": 0-10},\n'
        '  "hallucinations": {"ft_full": true/false, "ft_lora": true/false, "ft_qlora": true/false}\n'
        "}\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n\n"
    )

    for key in MODEL_KEYS:
        prompt += f"Answer ({key}):\n{answers.get(key, '')}\n\n"

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    content = resp["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        return {"scores": {}, "hallucinations": {}}


# ---------------- RAGAS utilities -----------------

def compute_ragas_faithfulness(
    examples: List[Example],
    per_model_answers: Dict[str, List[str]],
) -> Dict[str, Optional[float]]:

    if not _HAS_RAGAS:
        return {k: None for k in MODEL_KEYS}

    results = {}
    for model_key in MODEL_KEYS:
        questions, answers, contexts, gts = [], [], [], []

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
        results[model_key] = float(ragas_res["faithfulness"])

    return results


# ---------------- BERTScore utilities -----------------

def compute_bertscore(refs: List[str], cands: List[str], lang: str = "en") -> List[float]:
    if not _HAS_BERTSCORE:
        return []
    P, R, F1 = bert_score(cands, refs, lang=lang)
    return [float(f) for f in F1]


# ---------------- Accuracy utilities -----------------

def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ---------------- Main evaluation -----------------

def evaluate_models(
    examples: List[Example],
    judge_model_name: Optional[str] = None,
    bert_lang: str = "en",
    use_ragas: bool = False,
) -> Dict[str, Dict[str, Any]]:

    metrics = {k: ModelMetrics() for k in MODEL_KEYS}
    answers_per_model = {k: [] for k in MODEL_KEYS}
    refs_for_bertscore = {k: [] for k in MODEL_KEYS}

    # Task accuracy + BERTScore
    for ex in examples:
        has_ref = ex.reference_answer is not None
        for key in MODEL_KEYS:
            ans = ex.answers.get(key, "")
            answers_per_model[key].append(ans)

            if has_ref:
                metrics[key].total_with_reference += 1
                ref = ex.reference_answer or ""
                if normalize_text(ans) == normalize_text(ref):
                    metrics[key].exact_matches += 1
                if normalize_text(ref) in normalize_text(ans) or normalize_text(ans) in normalize_text(ref):
                    metrics[key].relaxed_matches += 1
                refs_for_bertscore[key].append(ref)

    # BERTScore
    if _HAS_BERTSCORE:
        for key in MODEL_KEYS:
            refs = refs_for_bertscore[key]
            cands = answers_per_model[key][: len(refs)]
            if refs and cands:
                metrics[key].bert_f1_scores.extend(
                    compute_bertscore(refs, cands, lang=bert_lang)
                )

    # LLM-as-a-Judge
    if judge_model_name and _HAS_OPENAI:

        # 1) .env / 環境変数から取得
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        # 2) それでも無ければ対話的に入力（Colab / Notebook）
        if not api_key:
            try:
                from getpass import getpass
                api_key = getpass("Enter OpenAI API key (hidden): ").strip()
            except Exception:
                api_key = ""

        if api_key:
            openai.api_key = api_key
            for ex in examples:
                context_text = "\n\n".join(c["text"] for c in ex.contexts)
                judge_res = call_judge_llm_openai(
                    question=ex.question,
                    context=context_text,
                    answers=ex.answers,
                    model=judge_model_name,
                )
                scores = judge_res.get("scores", {})
                hallucinations = judge_res.get("hallucinations", {})

                for key in MODEL_KEYS:
                    if key in scores:
                        metrics[key].judge_scores.append(float(scores[key]))
                    if key in hallucinations:
                        metrics[key].hallucination_flags.append(bool(hallucinations[key]))
        else:
            print("[WARN] No API key provided. Skipping Judge metrics.")

    # RAGAS
    ragas_scores = {}
    if use_ragas:
        ragas_scores = compute_ragas_faithfulness(examples, answers_per_model)

    summary = {}
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--bert_lang", type=str, default="en")
    parser.add_argument("--use_ragas", action="store_true")
    parser.add_argument("--output_path", type=str, default="model_metrics.json")
    args = parser.parse_args()

    if _HAS_DOTENV:
        load_dotenv()

    examples = load_examples(args.data_path)
    print(f"Loaded {len(examples)} examples from {args.data_path}")

    summary = evaluate_models(
        examples,
        judge_model_name=args.judge_model,
        bert_lang=args.bert_lang,
        use_ragas=args.use_ragas,
    )

    print("\n=== Model Metrics Summary ===")
    for key, metrics in summary.items():
        print(f"\n[{key}]")
        for m, v in metrics.items():
            print(f"  {m}: {v}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
