"""eval_both.py — full evaluation on test-clean AND test-other for a saved model.

Example:
    KMP_DUPLICATE_LIB_OK=TRUE python eval_both.py --model_dir ../runs/best_model
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import evaluate
from transformers import AutoModelForCTC, AutoProcessor

import sample_util

SAMPLE_RATE = 16_000
WER_METRIC = evaluate.load("wer")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True, help="Fine-tuned model+processor dir.")
    p.add_argument("--max_audio_sec", type=float, default=18.0)
    p.add_argument("--test_clean_subdir", default="test-clean")
    p.add_argument("--test_other_subdir", default="test-other")
    return p.parse_args()


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding,
                                   return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features,
                                          padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def materialize(data_dir: str, max_audio_sec: float) -> List[Dict]:
    max_len = int(max_audio_sec * SAMPLE_RATE)
    items: List[Dict] = []
    for s in sample_util.make_dataset(data_dir):
        if len(s["input_values"]) > max_len:
            continue
        items.append({"input_values": s["input_values"], "labels": s["labels"]})
    return items


def eval_wer(model, processor, dataset: List[Dict],
             collator: DataCollatorCTCWithPadding, device: str,
             batch_size: int = 8) -> float:
    """Run inference in mini-batches and compute WER."""
    all_pred_str: List[str] = []
    all_label_str: List[str] = []

    for i in range(0, len(dataset), batch_size):
        batch = collator(dataset[i: i + batch_size])
        input_values = batch["input_values"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = np.argmax(logits.cpu().numpy(), axis=-1)
        labels_np = labels.numpy()
        labels_np[labels_np == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(labels_np, group_tokens=False)
        all_pred_str.extend(pred_str)
        all_label_str.extend(label_str)

    return WER_METRIC.compute(predictions=all_pred_str, references=all_label_str)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_dir} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir).to(device).eval()
    collator = DataCollatorCTCWithPadding(processor=processor)

    print("Loading test-clean ...", flush=True)
    clean_dir = os.path.join(DATA_DIR, args.test_clean_subdir)
    clean_data = materialize(clean_dir, args.max_audio_sec)
    print(f"  test-clean: {len(clean_data)} utt", flush=True)

    print("Loading test-other ...", flush=True)
    other_dir = os.path.join(DATA_DIR, args.test_other_subdir)
    other_data = materialize(other_dir, args.max_audio_sec)
    print(f"  test-other: {len(other_data)} utt", flush=True)

    print("\n=== Full evaluation ===", flush=True)
    wer_clean = eval_wer(model, processor, clean_data, collator, device)
    print(f"test-clean: {len(clean_data)} utt  WER={wer_clean:.4f}", flush=True)

    wer_other = eval_wer(model, processor, other_data, collator, device)
    print(f"test-other: {len(other_data)} utt  WER={wer_other:.4f}", flush=True)


if __name__ == "__main__":
    main()
