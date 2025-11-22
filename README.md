
# TinyLlama Fine-Tuning Toolkit

## Overview

This repository provides a compact toolkit for fine‑tuning and evaluating **TinyLlama‑1.1B‑Chat‑v1.0**.  
It supports:

- Full Fine-Tuning (FT)
- LoRA
- QLoRA
- RAG evaluation using LlamaIndex + HuggingFace embeddings
- Model comparison utilities

Prefix Tuning is not included.

---

## Architecture

```
Training
  • Full FT
  • LoRA
  • QLoRA

Model Outputs
  • models/ft_full/
  • models/ft_lora/
  • models/ft_qlora/

Evaluation
  • app_rag_compare.py
  • compare_adapters.py
```

---

## Fine-Tuning Methods

### 1. Full Fine-Tuning  
Updates **all parameters** of the model.  
Provides the highest model capacity but requires more GPU memory.

```
python train_full.py
```

---

### 2. LoRA  
Adds low‑rank matrices to selected layers and **trains only those parameters**,  
keeping the base model frozen.  
Efficient in memory and compute.

```
python train_lora.py
```

---

### 3. QLoRA  
Quantizes the base model to **4‑bit** while training LoRA parameters in higher precision.  
Minimizes memory usage while achieving performance close to LoRA.

```
python train_qlora.py
```

---

## RAG Evaluation

This project includes a simple RAG pipeline using LlamaIndex and HuggingFace embeddings.

Example inside `app_rag_compare.py`:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
```

Run:

```
python app_rag_compare.py --docs_dir docs --question "Explain LoRA."
```

---

## Model Comparison

```
python compare_adapters.py
```

Compares:

- Full FT
- LoRA
- QLoRA

---

## Repository Structure

```
llm_ft_tinyllama/
├── train_full.py
├── train_lora.py
├── train_qlora.py
├── compare_adapters.py
├── app_rag_compare.py
├── requirements.txt
│
├── models/
│   ├── ft_full/
│   ├── ft_lora/
│   └── ft_qlora/
│
├── docs/
└── data/
    └── toy_qa.jsonl
```

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

python-dotenv>=1.0.0
llama-index-embeddings-huggingface
sentence-transformers
```

Install:

```
pip install -r requirements.txt
```
