# Fine-tuning the local SQL model

## Read this first

Training is the **last** lever, not the first. In order of effort-to-value for
this project:

| Lever | Effort | Typical gain |
|---|---|---|
| Semantic layer + few-shot examples | minutes | large — the single biggest win |
| Better base model (see benchmark below) | minutes | large |
| Retrieval of similar past questions | hours | moderate |
| QLoRA fine-tune | hours + a GPU | moderate, mostly on *style* and refusal behaviour |

A fine-tune teaches the model **format and house style** — column aliases,
`CANNOT_ANSWER` refusals, Indonesian phrasing, which joins to prefer. It does
not teach it your schema's facts, and it will not fix a bad semantic layer.

## Hardware reality

Your machine: RTX 5080 (16 GB VRAM), 80 GB RAM.

| Base model | Params | QLoRA VRAM (4-bit) | Feasible here |
|---|---|---|---|
| `qwen2.5-coder:7b` | 7.6B | ~9–11 GB | **yes** |
| `qwen3-vl:8b-instruct` | 8.2B | ~11–13 GB | yes, tight |
| 27B (Qwen3.8 abliterated) | 27.3B | ~22–26 GB | no — needs 2× GPU or CPU offload (very slow) |
| `qwen3.6` | 36B | ~30 GB+ | no |

So: **train a 7B adapter**, not the 36B you use for inference. Then either

- run `retail-sql` (the tuned 7B) for speed, keeping `qwen3.6` for hard questions, or
- use the adapter as a style prior for the fast path.

## Windows caveat

`bitsandbytes` (required for 4-bit training) is the fragile part on native
Windows. Two reliable options:

1. **WSL2** (recommended) — install the CUDA toolkit inside WSL, then:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install transformers peft datasets accelerate bitsandbytes
   ```
2. **Unsloth** on Windows via WSL, or a cloud GPU for a one-off run.

Inference (Ollama) is unaffected — this caveat applies only to training.

## Steps

### 1. Build the dataset

```bash
python -m training.build_dataset --count 200
```

This generates questions from the semantic layer in English and Indonesian,
asks the current model to answer each, and **keeps only SQL that PostgreSQL
accepts** (validated with `EXPLAIN`). Output:

- `training/data/train.jsonl` — chat-format, prompt tokens masked at training time
- `training/data/valid.jsonl` — held-out split
- `training/data/meta.json` — counts, failure rate, model used

Sanity-check without calling the model:

```bash
python -m training.build_dataset --count 40 --dry-run
```

A healthy run keeps 80–95% of candidates. If the failure rate is high, fix the
semantic layer before training.

### 2. Train

```bash
python -m training.train_qlora \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --epochs 3 --rank 16 --alpha 32
```

Adapter lands in `training/out/adapter/`. With ~200 examples and 3 epochs this
is roughly 15–30 minutes on the 5080.

### 3. Import into Ollama

Edit `ADAPTER` in `training/Modelfile` if needed, then:

```bash
ollama create retail-sql -f training/Modelfile
ollama run retail-sql "How did revenue trend over the last 30 days?"
```

### 4. Use it

In `.env`:

```
SQL_MODEL=retail-sql
```

Restart the dashboard. Compare against the baseline with the smoke test:

```bash
python -m scripts.smoke_nl2sql
```

## Evaluating honestly

Do not judge a fine-tune by "the answers look nicer". Measure:

- **Valid-SQL rate** — does `EXPLAIN` accept it? (`scripts/smoke_nl2sql.py` reports failures)
- **Execution match** — same row count / same aggregates as the reference query
- **Refusal accuracy** — does it say `CANNOT_ANSWER` for unanswerable questions?

Keep `training/data/valid.jsonl` untouched and evaluate only on that.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | question generation + validated SQL labelling |
| `train_qlora.py` | 4-bit QLoRA training, prompt-masked loss |
| `Modelfile` | Ollama import template |
| `data/` | generated datasets (gitignored) |
| `out/` | adapters and checkpoints (gitignored) |
