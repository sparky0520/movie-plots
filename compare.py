"""
Compare the base model with the fine-tuned model on the same prompts.

Setup:   pip install mlx-lm
Run:     python compare.py
Output:  prints results and saves comparison.md
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ---- Change these ----
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "adapters"  # folder made by mlx_lm.lora training
MAX_TOKENS = 400
TEMP = 0.8

# Use the SAME format you used in your training data.
TEST_PROMPTS = [
    "Genre: sci-fi. Keywords: haunted spaceship, betrayal",
    "Genre: romance. Keywords: tax audit, zombies",
    "Genre: thriller. Keywords: lost phone, small town, secret twin",
    "Genre: comedy. Keywords: samosa shop, rivalry, wedding",
    "Genre: horror. Keywords: library, midnight, whispering books",
]
# ----------------------


def run(model, tokenizer, prompts):
    sampler = make_sampler(temp=TEMP, top_p=0.9)
    outputs = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        out = generate(
            model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler
        )
        outputs.append(out.strip())
    return outputs


print("Loading base model...")
model, tokenizer = load(BASE_MODEL)
before = run(model, tokenizer, TEST_PROMPTS)
del model

after = None
try:
    print("Loading fine-tuned model...")
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    after = run(model, tokenizer, TEST_PROMPTS)
    del model
except Exception as e:
    print(f"Fine-tuned model not found at '{ADAPTER_PATH}' ({e}). Skipping...")

lines = []
for i, p in enumerate(TEST_PROMPTS):
    lines += [
        f"## {p}",
        "",
        "**BEFORE (base model):**",
        "",
        before[i],
        "",
    ]
    if after is not None:
        lines += [
            "**AFTER (fine-tuned):**",
            "",
            after[i],
            "",
        ]
    lines += [
        "---",
        "",
    ]

report = "\n".join(lines)
print(report)
with open("comparison.md", "w") as f:
    f.write(report)
print("Saved to comparison.md")
