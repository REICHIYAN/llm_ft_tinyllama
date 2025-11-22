# compare_adapters.py
# TinyLlama: Full FT / LoRA / QLoRA の比較スクリプト
# ---------------------------------------------------------------
# 目的:
#   - models/ 配下の主要 3 種類の FT モデルをロードして、同一プロンプトで出力比較を行う
#   - CUDA メモリリークを防ぎ、安全なロード・実行を保証する

import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------
# 設定
# ---------------------------------------------------------------
REPO_ROOT = "/content/llm_ft_tinyllama"

model_dirs = {
    "base_full": f"{REPO_ROOT}/models/ft_full",      # Full Fine-tuning
    "lora":      f"{REPO_ROOT}/models/ft_lora",      # LoRA
    "qlora":     f"{REPO_ROOT}/models/ft_qlora",     # QLoRA
}

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "Explain LoRA in the context of fine-tuning LLMs in two sentences."

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# 関数: 1モデルを評価
# ---------------------------------------------------------------
def evaluate_model(name: str, adapter_path: str):
    print("=" * 60)
    print(f"[{name}] from {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # モデル読み込み分岐
    if name == "base_full":
        # フルFTモデルは単独ロード
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    else:
        # LoRA / QLoRA → base model + adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    print()

    # メモリ解放（CUDA落ち防止）
    del model
    if name != "base_full":
        del base_model
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Running adapter comparison...\n")

    for name, path in model_dirs.items():
        if not os.path.exists(path):
            print(f"[WARN] {name} directory not found: {path}")
            continue

        evaluate_model(name, path)

    print("Done.")
