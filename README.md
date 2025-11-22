
# TinyLlama Fine-Tuning Toolkit (English Section)

---

## 🌐 Overview

This repository provides a **compact, end-to-end fine-tuning, serving, and evaluation toolkit**  
for **TinyLlama-1.1B-Chat-v1.0**, enabling reproducible experiments across:

- Full Fine-Tuning (FT)  
- LoRA  
- QLoRA  
- vLLM serving  
- RAG evaluation  
- Unified model comparison utilities  

Prefix Tuning has been intentionally excluded in this version to keep the stack clean and minimal.

---

## 🏗️ Architecture & Tech Stack

```
┌───────────────────────────────────────────┐
│                Training Layer             │
│  • Full FT (HF Trainer)                   │
│  • LoRA (PEFT)                            │
│  • QLoRA (PEFT + 4bit Quantization)       │
│                                           │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│               Model Outputs                │
│  models/ft_full/                           │
│  models/ft_lora/                           │
│  models/ft_qlora/                          │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│            Serving & Evaluation            │
│  • vLLM OpenAI-compatible server           │
│  • app_rag_compare.py (RAG pipeline)       │
│  • compare_adapters.py                     │
└────────────────────────────────────────────┘
```

---

## 📚 Fine-Tuning Methods

### **1. Full Fine-Tuning**
All model parameters are updated via supervised fine-tuning (SFT).  
This produces the highest capacity and quality, but requires significant GPU resources.

Run:
```bash
python train_full.py
```

---

### **2. LoRA (Low-Rank Adaptation)**
LoRA injects trainable low-rank matrices into attention projections,  
training only these additional parameters while freezing the base model.

Characteristics:
- Lightweight  
- Fast  
- Extremely memory efficient  
- Adapter weights are tiny

Run:
```bash
python train_lora.py
```

---

### **3. QLoRA**
QLoRA quantizes the model backbone to **4-bit NF4**,  
while training LoRA adapters in fp16, drastically reducing VRAM usage.

Characteristics:
- Lowest VRAM consumption  
- Works on 8–16GB GPUs  
- Near-LoRA quality

Run:
```bash
python train_qlora.py
```

---

## 🧪 Dataset

SFT demo dataset is located at:

`data/toy_qa.jsonl`

Format:
```json
{"question": "...", "answer": "..."}
```

---

## 🚀 Serving with vLLM

vLLM provides high-throughput inference through PagedAttention.

Run:
```bash
python -m vllm.entrypoints.openai.api_server --model ./models/ft_full
```

---

## 🔍 RAG Evaluation

Run:
```bash
python app_rag_compare.py --docs_dir docs --question "Explain LoRA."
```

Evaluates:
- Embedding quality  
- Retrieval differences  
- Answer consistency  

---

## 🧭 Model Comparison

Run:
```bash
python compare_adapters.py
```

Compares output for:
- Full FT  
- LoRA  
- QLoRA  

---

## 📁 Repository Structure

```
llm_ft_tinyllama/
├── train_full.py
├── train_lora.py
├── train_qlora.py
├── compare_adapters.py
├── app_rag_compare.py
│
├── models/
│   ├── ft_full/
│   ├── ft_lora/
│   └── ft_qlora/
│
└── data/
    └── toy_qa.jsonl
```

---

## 🛠 Requirements

- Python 3.10+  
- PyTorch (CUDA)  
- HuggingFace Transformers  
- PEFT  
- bitsandbytes  
- vLLM  

Install:
```bash
pip install -r requirements.txt
```

---

## 🙌 Final Notes

This repository is intended to be a **clean, extensible baseline** for LLM fine-tuning research.

---

# TinyLlama 微調整ツールキット（日本語セクション）

---

## 🌐 概要

本リポジトリは、**TinyLlama-1.1B-Chat-v1.0 の微調整・推論・評価を一気通貫で扱える、最小構成の実験用キット**です。

以下の構成をサポートします：

- フル微調整（Full FT）  
- LoRA  
- QLoRA  
- vLLM による高速推論  
- RAG 評価  
- モデル比較ユーティリティ  

Prefix Tuning は本バージョンから除外済みです。

---

## 🏗️ 技術スタック・アーキテクチャ

```
┌───────────────────────────────────────────┐
│                学習レイヤー               │
│  • Full FT (HF Trainer)                   │
│  • LoRA (PEFT)                            │
│  • QLoRA (4bit 量子化 + LoRA)             │
│                                           │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│             モデル出力ディレクトリ         │
│  models/ft_full/                           │
│  models/ft_lora/                           │
│  models/ft_qlora/                          │
└───────────────┬───────────────────────────┘
                │
┌───────────────▼───────────────────────────┐
│           推論・評価レイヤー               │
│  • vLLM API サーバ                         │
│  • app_rag_compare.py                      │
│  • compare_adapters.py                     │
└────────────────────────────────────────────┘
```

---

## 📚 微調整手法の説明

### **1. フル微調整（Full FT）**
モデル全パラメータを更新する、最も汎用的で高品質な手法。  
ただし VRAM を大きく消費します。

実行:
```bash
python train_full.py
```

---

### **2. LoRA**
アテンション層に低ランク行列（A, B）を挿入し、  
**その部分だけ学習する**省メモリ手法。

特徴：
- 本体モデルは凍結  
- 学習が高速  
- 省メモリ  
- Adapter 重みが非常に小さい

実行:
```bash
python train_lora.py
```

---

### **3. QLoRA**
モデル本体を **4bit（NF4）量子化**し、  
LoRA アダプタ部分のみ fp16 で学習する方式。

特徴：
- VRAM 使用量が最小  
- 8〜16GB GPU で実用的  
- LoRA と同等の品質

実行:
```bash
python train_qlora.py
```

---

## 🧪 データセット

データ：`data/toy_qa.jsonl`

形式：
```json
{"question": "...", "answer": "..."}
```

---

## 🚀 vLLM による推論

PagedAttention を使った高スループット推論。

実行:
```bash
python -m vllm.entrypoints.openai.api_server --model ./models/ft_full
```

---

## 🔍 RAG 評価

実行:
```bash
python app_rag_compare.py --docs_dir docs --question "Explain LoRA."
```

評価項目：
- 埋め込み性能  
- 検索品質  
- 応答の一貫性  

---

## 🧭 モデル比較

実行:
```bash
python compare_adapters.py
```

比較対象：
- Full FT  
- LoRA  
- QLoRA  

---

## 📁 リポジトリ構成

```
llm_ft_tinyllama/
├── train_full.py
├── train_lora.py
├── train_qlora.py
├── compare_adapters.py
├── app_rag_compare.py
│
├── models/
│   ├── ft_full/
│   ├── ft_lora/
│   └── ft_qlora/
│
└── data/
    └── toy_qa.jsonl
```

---

## 🛠 必要ライブラリ

- Python 3.10+  
- PyTorch (CUDA)  
- HuggingFace Transformers  
- PEFT  
- bitsandbytes  
- vLLM  

インストール:
```bash
pip install -r requirements.txt
```

---

## 🙌 最後に

本リポジトリは、TinyLlama を用いた LLM 微調整研究の  
**クリーンで拡張性の高いベースライン**として設計されています。

