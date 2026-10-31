"""QLoRA fine-tuning for the NL->SQL model.

Runs on a single 16 GB GPU with a 7B base model in 4-bit. On Windows the
`bitsandbytes` wheel is the fragile part; WSL2 is the reliable path (see
training/README.md).

    pip install torch transformers peft datasets accelerate bitsandbytes
    python -m training.train_qlora --model Qwen/Qwen2.5-Coder-7B-Instruct

Output: a LoRA adapter in training/out/adapter, importable into Ollama with
training/Modelfile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_example(tokenizer, record: dict) -> dict:
    """Return input_ids and labels with the prompt tokens masked out."""
    messages = record["messages"]
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + answer + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=2048)

    input_ids = full["input_ids"]
    labels = list(input_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def train(args) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(Path(args.data) / "train.jsonl")
    valid_path = Path(args.data) / "valid.jsonl"
    valid_records = load_records(valid_path) if valid_path.exists() else []

    print(f"train={len(train_records)} valid={len(valid_records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list([format_example(tokenizer, r) for r in train_records])
    eval_dataset = (
        Dataset.from_list([format_example(tokenizer, r) for r in valid_records])
        if valid_records
        else None
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"adapter saved to {output_dir}")

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = output_dir.parent / "merged"
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"merged model saved to {merged_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the NL->SQL model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(OUT_DIR / "adapter"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--merge", action="store_true", help="also write a merged fp16 model")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
