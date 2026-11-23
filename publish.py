"""
Merge the LoRA adapter into the base model and upload to Hugging Face.

Setup:
    pip install mlx-lm huggingface_hub
    huggingface-cli login        # paste a token with WRITE access

Run:
    python publish.py
"""

import subprocess
from pathlib import Path
from huggingface_hub import HfApi

# ---- Change these ----
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "adapters"
SAVE_PATH = "movie-plot-model"
REPO_ID = "your-hf-username/movie-plot-writer"  # <-- change this
# ----------------------

# 1. Merge adapter into the base model
subprocess.run(
    [
        "python", "-m", "mlx_lm.fuse",
        "--model", BASE_MODEL,
        "--adapter-path", ADAPTER_PATH,
        "--save-path", SAVE_PATH,
    ],
    check=True,
)

# 2. Write a simple model card
card = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
datasets:
- vishnupriyavr/wiki-movie-plots-with-summaries
tags:
- mlx
- lora
- creative-writing
- movie-plots
---

# Movie Plot Writer

A small model fine-tuned for fun to write movie plots from a genre and keywords.

## Prompt format

```
Genre: sci-fi. Keywords: haunted spaceship, betrayal
```

## Usage (MLX)

```python
from mlx_lm import load, generate

model, tokenizer = load("{REPO_ID}")
messages = [{{"role": "user", "content": "Genre: sci-fi. Keywords: haunted spaceship, betrayal"}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400))
```

## Notes

- Fine-tuned with LoRA on Wikipedia movie plot summaries (CC BY-SA).
- Made as a fun experiment. Plots can be odd or repetitive.
"""
Path(SAVE_PATH, "README.md").write_text(card)

# 3. Upload
api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(folder_path=SAVE_PATH, repo_id=REPO_ID)
print(f"Done: https://huggingface.co/{REPO_ID}")
