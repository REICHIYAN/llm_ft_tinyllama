# FT-Lab

A compact, reproducible toolkit for fine-tuning, evaluating, and comparing  
TinyLlama-1.1B-Chat-v1.0 using Full FT, LoRA, and QLoRA.

The project is designed for small-GPU environments, research experimentation,  
and transparent ablation studies.

---

## Features

### 🔧 Fine-Tuning
- Full Fine-Tuning  
- LoRA  
- QLoRA  
- Shared training utilities (`training_utils.py`)

### 📘 Evaluation Tools
- RAG evaluation (LlamaIndex / LangChain)  
- Retrieval-only metrics  
- Model comparison (FT / LoRA / QLoRA)  
- Local inference script (`local_hf_chat_model.py`)

### 🗂 Sample Data
- RAG document samples  
- Small QA datasets  

**Note:** Prefix Tuning is intentionally excluded.

---

## Repository Structure

```
ft_lab/
├── app_rag_compare.py
├── app_rag_compare_langchain.py
├── app_rag_compare_llamaindex.py
├── compare_adapters.py
├── eval_models.py
├── eval_retrieval.py
├── local_hf_chat_model.py
├── requirements.txt
├── training_utils.py
├── train_full.py
├── train_lora.py
├── train_qlora.py
│
├── models/
│   ├── ft_full/
│   ├── ft_lora/
│   └── ft_qlora/
│
├── data/
│   ├── toy_qa.jsonl
│   └── sample_eval.jsonl
│
├── docs/
│   ├── sample1.txt
│   └── sample2.txt	
│
└── examples/
     └── FT-Lab.ipynb
```

---

## Fine-Tuning Scripts

### Full Fine-Tuning  
Updates all parameters.

```bash
python train_full.py
```

### LoRA  
Parameter-efficient training with injected low-rank matrices.

```bash
python train_lora.py
```

### QLoRA  
4-bit quantized base model + LoRA adapters.

```bash
python train_qlora.py
```

---

## Training Utilities

`training_utils.py` includes:

- dataset loading  
- tokenizer setup  
- model initialization  
- training arguments  
- evaluation hooks  

All training scripts share this module for consistent behavior.

---

## RAG Evaluation

### LlamaIndex Pipeline
**File:** `app_rag_compare_llamaindex.py`

```bash
python app_rag_compare.py     --docs_dir docs     --question "Explain LoRA."
```

### LangChain Pipeline
**File:** `app_rag_compare_langchain.py`  
Compatible with LangChain 0.2+ (Runnable / LCEL).

---

## Model Comparison

Compare FT / LoRA / QLoRA generations:

```bash
python compare_adapters.py
```

Outputs:
- aligned generations  
- qualitative differences  
- optional latency comparison  

---

## Retrieval-Only Metrics
```bash
python eval_retrieval.py --data data/sample_eval.jsonl
```

Metrics:
- recall@k  
- precision@k  
- hit-rate  

---

## Model Evaluation (FT / LoRA / QLoRA)
```bash
python eval_models.py --data_path data/sample_eval.jsonl
```

Metrics:
- BERTScore-F1  
- exact-match accuracy  
- relaxed-match accuracy  

---

## Sample Data

```
data/toy_qa.jsonl
data/sample_eval.jsonl
docs/sample1.txt
docs/sample2.txt
```

Useful for RAG demonstrations and baseline evaluations.

---

## Running the Colab Demo

A runnable notebook is available under:

```
examples/FT-Lab.ipynb
```

This notebook:
- uses only dummy data
- demonstrates the end-to-end pipeline
- is designed for Colab / T4 / small VRAM
- can be fully replaced with real datasets

---

## Requirements

```
torch>=2.1.0
transformers>=4.39.0
accelerate>=0.27.0
sentencepiece>=0.1.99
einops>=0.7.0

datasets>=2.18.0
peft>=0.10.0
bitsandbytes>=0.42.0

langchain>=0.2.0
langchain-openai>=0.1.0
llama-index>=0.10.0
llama-index-embeddings-huggingface
sentence-transformers

python-dotenv>=1.0.0
```

---

## Install

```bash
pip install -r requirements.txt
```
