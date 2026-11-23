# Movie Plot Writer 🎬

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` using **MLX LoRA** on Apple Silicon to generate creative movie plots conditioned on a specific **Genre** and a set of **Keywords**.

---

## 🚀 Quickstart

### 1. Requirements & Setup
This project uses [uv](https://docs.astral.sh/uv/) for package management:

```bash
# Install dependencies
uv sync

# Launch JupyterLab
uv run jupyter lab
```

---

## 📁 Workflow

### Step 1: Prepare the Dataset (`01_prepare_dataset.ipynb`)
1. Loads and filters Wikipedia movie plots from `vishnupriyavr/wiki-movie-plots-with-summaries`.
2. Extracts 3–5 grounded thematic keywords for each plot using few-shot prompting with `Qwen2.5-1.5B-Instruct`.
3. Exports `review.csv` so you can manually inspect, edit, or drop noisy rows (`keep = 0`).
4. Compiles the reviewed dataset into `data/train.jsonl` and `data/valid.jsonl` in chat template format.

### Step 2: Train with MLX LoRA (`02_train.ipynb`)
1. Configures LoRA parameters (rank, batch size, learning rate, max sequence length).
2. Runs background training with `mlx_lm.lora`.
3. Live training dashboard with dynamic loss curves and automated early stopping.
4. Checkpoint evaluation to select the best adapter weights (`adapters_best`).

### Step 3: Compare & Evaluate (`compare.py`)
Runs before-and-after inference on benchmark test prompts to compare the base model against the fine-tuned adapter, generating `comparison.md`.

```bash
uv run python compare.py
```

### Step 4: Publish to Hugging Face (`publish.py`)
Fuses the LoRA adapter directly into the base model weights with `mlx_lm.fuse` and uploads the fused model to Hugging Face Hub.

```bash
uv run python publish.py
```

---

## 💬 Prompt Format

```text
User: Genre: sci-fi. Keywords: haunted spaceship, betrayal
Assistant: <generated movie plot>
```
