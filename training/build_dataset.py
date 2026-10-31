"""Build a fine-tuning dataset of (question -> validated SQL) pairs.

Questions are generated combinatorially from the semantic layer, then the
current best model answers each one. Every answer is checked with EXPLAIN
against the live database before it is kept, so the dataset never teaches the
model invalid SQL.

English and Indonesian phrasings are generated separately: mixing a phrase from
one language with a template from the other produces nonsense questions like
"What is number of buyers by kategori produk?".

Output: chat-format JSONL, ready for training/train_qlora.py or Ollama.

Usage
    python -m training.build_dataset --count 200
    python -m training.build_dataset --count 200 --model qwen3.6:latest
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from core.nl2sql import build_system_prompt, question_to_sql
from semantic import load as load_semantic_layer

OUT_DIR = Path(__file__).resolve().parent / "data"

METRIC_PHRASES = {
    "en": {
        "revenue": ["total revenue", "total sales"],
        "orders": ["order count", "number of orders"],
        "units": ["units sold", "quantity sold"],
        "active_users": ["active users", "number of buyers"],
        "avg_order_value": ["average order value", "AOV"],
        "revenue_per_user": ["revenue per user", "ARPU"],
        "promo_spend": ["promotion spend", "promo cost"],
        "roas": ["ROAS", "return on ad spend"],
    },
    "id": {
        "revenue": ["total omzet", "total pendapatan"],
        "orders": ["jumlah pesanan", "banyaknya transaksi"],
        "units": ["jumlah item terjual", "kuantitas terjual"],
        "active_users": ["jumlah pembeli", "pengguna aktif"],
        "avg_order_value": ["rata-rata nilai pesanan"],
        "revenue_per_user": ["pendapatan per pelanggan"],
        "promo_spend": ["biaya promosi", "belanja promosi"],
        "roas": ["efisiensi promo", "ROAS"],
    },
}

DIMENSION_PHRASES = {
    "en": {
        "category": ["product category", "category"],
        "product": ["product", "item"],
        "city": ["city", "region"],
        "order_date": ["day", "date"],
        "order_week": ["week"],
        "order_month": ["month"],
        "user_city": ["user city"],
    },
    "id": {
        "category": ["kategori produk", "kategori"],
        "product": ["produk", "barang"],
        "city": ["kota", "wilayah"],
        "order_date": ["tanggal", "hari"],
        "order_week": ["minggu"],
        "order_month": ["bulan"],
        "user_city": ["kota pengguna"],
    },
}

TEMPLATES = {
    "en": [
        "What is {metric} by {dimension}?",
        "Show {metric} per {dimension}.",
        "Break down {metric} by {dimension}.",
        "Which {dimension} has the highest {metric}?",
        "Which {dimension} has the lowest {metric}?",
        "Rank {dimension} by {metric}.",
        "Compare {metric} across {dimension}.",
        "How does {metric} vary by {dimension}?",
        "List {metric} for each {dimension}.",
    ],
    "id": [
        "Berapa {metric} per {dimension}?",
        "Tampilkan {metric} berdasarkan {dimension}.",
        "Urutkan {dimension} berdasarkan {metric}.",
        "Bagaimana {metric} menurut {dimension}?",
        "{dimension} mana yang memiliki {metric} tertinggi?",
    ],
}

TIME_TEMPLATES = {
    "en": [
        "How did {metric} trend over the last 30 days?",
        "Show {metric} per day for the whole dataset period.",
        "Show weekly {metric}.",
        "How did {metric} change over time?",
    ],
    "id": [
        "Bagaimana tren {metric} selama 30 hari terakhir?",
        "Tampilkan {metric} per hari sepanjang periode data.",
    ],
}

# Questions the schema genuinely cannot answer. Teaching the model to refuse
# is as important as teaching it to query.
NEGATIVES = [
    "What is the weather in Tokyo tomorrow?",
    "Who is the CEO of the company?",
    "Write me a poem about retail.",
    "What will revenue be next year?",
]


def candidate_questions(limit: int, seed: int = 7) -> list[str]:
    layer = load_semantic_layer()
    pool: list[str] = []

    for language in ("en", "id"):
        for metric_id, metric_phrases in METRIC_PHRASES[language].items():
            if metric_id not in layer.metrics:
                continue

            for dimension_id, dimension_phrases in DIMENSION_PHRASES[language].items():
                if dimension_id not in layer.dimensions:
                    continue

                for template in TEMPLATES[language]:
                    for metric_phrase in metric_phrases:
                        for dimension_phrase in dimension_phrases:
                            pool.append(
                                template.format(metric=metric_phrase, dimension=dimension_phrase)
                            )

            for template in TIME_TEMPLATES[language]:
                for metric_phrase in metric_phrases[:2]:
                    pool.append(template.format(metric=metric_phrase))

    pool = list(dict.fromkeys(pool))
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def build(
    count: int = 150,
    model: str | None = None,
    out_dir: Path = OUT_DIR,
    seed: int = 7,
    dry_run: bool = False,
    max_per_sql: int = 3,
) -> dict:
    layer = load_semantic_layer()
    # The dataset must carry the exact system prompt used at inference time,
    # otherwise the fine-tune learns a different contract than the app applies.
    system_prompt = build_system_prompt(layer)

    questions = candidate_questions(count, seed)
    print(f"{len(questions)} candidate questions")

    if dry_run:
        for question in questions[:20]:
            print(" -", question)
        return {"dry_run": True, "candidates": len(questions)}

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    sql_counts: Counter = Counter()
    failures = 0

    started = time.time()
    for index, question in enumerate(questions, 1):
        result = question_to_sql(question, layer=layer, model=model)

        if not result.ok:
            failures += 1
            if index % 10 == 0:
                print(f"  [{index}/{len(questions)}] {failures} failed so far")
            continue

        # Do not let one query shape dominate the dataset.
        key = " ".join(result.sql.lower().split())
        if sql_counts[key] >= max_per_sql:
            continue
        sql_counts[key] += 1

        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result.sql},
                ],
                "meta": {"attempts": len(result.attempts), "model": result.model},
            }
        )

        if index % 10 == 0:
            print(f"  [{index}/{len(questions)}] kept={len(records)} failed={failures}")

    for question in NEGATIVES:
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "CANNOT_ANSWER"},
                ],
                "meta": {"kind": "refusal"},
            }
        )

    random.Random(seed).shuffle(records)
    split = max(1, int(len(records) * 0.9))
    train, valid = records[:split], records[split:]

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8"
    )
    (out_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in valid), encoding="utf-8"
    )

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": len(questions),
        "kept": len(records),
        "failed": failures,
        "train": len(train),
        "valid": len(valid),
        "unique_sql": len(sql_counts),
        "model": model or "default",
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a NL->SQL fine-tuning dataset")
    parser.add_argument("--count", type=int, default=150, help="number of candidate questions")
    parser.add_argument("--model", default=None, help="Ollama model used to label the data")
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="print questions, do not call the model")
    args = parser.parse_args()

    build(
        count=args.count,
        model=args.model,
        out_dir=Path(args.out),
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
