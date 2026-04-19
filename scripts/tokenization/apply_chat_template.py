#!/usr/bin/env python3
"""
to_apertus_parquet.py
─────────────────────
Apply the Apertus chat template (via the tokenizer's built-in
apply_chat_template) to a linearised dataset and save as parquet.

Each row in the output parquet has:
  - conversation_id
  - dataset_source
  - text   (the fully-rendered conversation string)

Input can be either:
  - A HuggingFace dataset saved with save_to_disk() (output of linearise_dataset.py)
  - A JSON file containing a list of conversation objects (for testing)

Usage
─────
# From a linearised HF dataset
python to_apertus_parquet.py \\
    --input      /path/to/linearised_dataset \\
    --output     output.parquet \\
    --tokenizer  swiss-ai/Apertus-8B-Instruct-2509 \\
    [--split     train] \\
    [--sample    100]   \\
    [--num-proc  8]

# From a JSON test file
python to_apertus_parquet.py \\
    --input-json example.json \\
    --output     output.parquet \\
    --tokenizer  swiss-ai/Apertus-8B-Instruct-2509
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer


def load_hf_dataset(
    input_path: str,
    split: str,
    sample: Optional[int],
) -> List[Dict[str, Any]]:
    from datasets import DatasetDict, load_from_disk

    print(f"Loading HF dataset from: {input_path}")
    ds = load_from_disk(input_path)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
        ds = ds[split]
    print(f"  {len(ds):,} samples")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
        print(f"  Sampled first {len(ds):,}")
    return list(ds)


def load_json_file(input_json: str, sample: Optional[int]) -> List[Dict[str, Any]]:
    print(f"Loading JSON from: {input_json}")
    with open(input_json) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of conversation objects.")
    if sample:
        data = data[:sample]
    print(f"  {len(data):,} conversations")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Apertus chat template and save as parquet"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",      help="HF dataset (save_to_disk format)")
    src.add_argument("--input-json", help="JSON file with a list of conversation objects")

    parser.add_argument("--output",    required=True, help="Destination .parquet file")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path")
    parser.add_argument("--split",     default="train", help="Dataset split (default: train)")
    parser.add_argument("--sample",    type=int, default=None, help="Only process first N rows")
    parser.add_argument("--num-proc",  type=int, default=1,    help="Workers for HF map (default: 1)")

    args = parser.parse_args()

    # load records
    if args.input_json:
        records = load_json_file(args.input_json, args.sample)
    else:
        records = load_hf_dataset(args.input, args.split, args.sample, args.num_proc)

    # load tokenizer (brings the chat_template.jinja along)
    print(f"\nLoading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # render
    print("Applying chat template…")
    rows = []
    for rec in records:
        messages = rec["messages"]

        # The Apertus template does not accept "developer" as a message role —
        # it expects enable_thinking and tools as top-level kwargs instead.
        # Extract the developer message and pass its fields separately.
        enable_thinking = False
        tools = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "developer":
                dc = msg.get("content", {})
                enable_thinking = bool(dc.get("has_thinking", False))
                raw_tools = dc.get("tools", "")
                if raw_tools:
                    tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            else:
                filtered_messages.append(msg)

        text = tokenizer.apply_chat_template(
            filtered_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        # The template unconditionally prepends bos_token — strip it.
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        rows.append({
            "conversation_id": rec.get("conversation_id", ""),
            "dataset_source":  rec.get("dataset_source", ""),
            "text": text,
        })

    # save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(str(out), index=False)
    print(f"\nSaved {len(df):,} rows → {out}")
    print("\n--- Sample (first row) ---")
    print(df["text"].iloc[0][:600])


if __name__ == "__main__":
    main()