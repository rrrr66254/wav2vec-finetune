"""train_local.py — laptop-friendly fine-tuning runner (added file; template untouched).

Uses the professor's sample_util.make_dataset pipeline with the provided
WebDataset (data/1h for training, data/test-clean for eval).

Exposes all heavy knobs via argparse so the job fits an RTX 5070 Ti laptop.
Runs a baseline eval, trains, then a final eval.

Example (baseline):
    KMP_DUPLICATE_LIB_OK=TRUE python train_local.py

Example (best config: freeze6, 2400 steps):
    KMP_DUPLICATE_LIB_OK=TRUE python train_local.py \\
        --freeze_transformer_layers 6 --max_steps 2400 \\
        --save_model_dir ../runs/best_model
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import evaluate
from transformers import (AutoModelForCTC, AutoProcessor, Trainer,
                          TrainingArguments)

import sample_util

SAMPLE_RATE = 16000
WER_METRIC = evaluate.load("wer")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="facebook/wav2vec2-base")
    p.add_argument("--train_subdir", default="1h")
    p.add_argument("--eval_subdir", default="test-clean")
    p.add_argument("--output_dir", default=os.path.join(HERE, "..", "runs", "local"))
    p.add_argument("--max_audio_sec", type=float, default=18.0,
                   help="Skip utterances longer than this (caps VRAM).")
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help="Cap eval samples for fast periodic eval (0=all).")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--eval_steps", type=int, default=400)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    # Improvement methods
    p.add_argument("--freeze_transformer_layers", type=int, default=0,
                   help="Freeze feature_projection + first N encoder layers (0=off).")
    p.add_argument("--mask_time_prob", type=float, default=None)
    p.add_argument("--mask_feature_prob", type=float, default=None)
    p.add_argument("--save_model_dir", default="",
                   help="If set, save the fine-tuned model+processor here.")
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


def materialize(dataset, desc: str, max_audio_sec: float, limit: int = 0) -> List[Dict]:
    """Cache the WebDataset into an in-memory list, filtering long utterances."""
    max_len = int(max_audio_sec * SAMPLE_RATE)
    items: List[Dict] = []
    for s in dataset:
        if len(s["input_values"]) > max_len:
            continue
        items.append({"input_values": s["input_values"], "labels": s["labels"]})
        if limit and len(items) >= limit:
            break
    print(f"[cache] {desc}: {len(items)} samples in RAM", flush=True)
    return items


def build_compute_metrics(processor):
    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)  # greedy CTC decoding
        label_ids = np.where(pred.label_ids == -100,
                             processor.tokenizer.pad_token_id, pred.label_ids)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": WER_METRIC.compute(predictions=pred_str,
                                          references=label_str)}
    return compute_metrics


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_name)

    train_dir = os.path.join(DATA_DIR, args.train_subdir)
    eval_dir = os.path.join(DATA_DIR, args.eval_subdir)

    train_dataset = materialize(sample_util.make_dataset(train_dir),
                                "train", args.max_audio_sec)
    eval_dataset = materialize(sample_util.make_dataset(eval_dir),
                               "eval", args.max_audio_sec,
                               limit=args.max_eval_samples)

    # SpecAugment (method 3) — only override if user explicitly sets them
    spec_kwargs = {}
    if args.mask_time_prob is not None:
        spec_kwargs["mask_time_prob"] = args.mask_time_prob
        spec_kwargs["apply_spec_augment"] = True
    if args.mask_feature_prob is not None:
        spec_kwargs["mask_feature_prob"] = args.mask_feature_prob
        spec_kwargs["apply_spec_augment"] = True

    model = AutoModelForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        **spec_kwargs,
    )

    # Always freeze the feature encoder (convolutional stack)
    model.freeze_feature_encoder()

    # Method 2: optionally freeze feature_projection + first N transformer layers
    if args.freeze_transformer_layers > 0:
        for prm in model.wav2vec2.feature_projection.parameters():
            prm.requires_grad = False
        n = min(args.freeze_transformer_layers,
                len(model.wav2vec2.encoder.layers))
        for layer in model.wav2vec2.encoder.layers[:n]:
            for prm in layer.parameters():
                prm.requires_grad = False
        print(f"[freeze] feature_projection + first {n} encoder layers frozen",
              flush=True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.1f}%)", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_steps=args.logging_steps,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
    )

    print("=== Baseline eval (before training) ===", flush=True)
    baseline = trainer.evaluate()
    print("BASELINE", {k: round(v, 4) for k, v in baseline.items()
                       if isinstance(v, float)}, flush=True)

    print("=== Training ===", flush=True)
    trainer.train()

    print("=== Final eval (after training) ===", flush=True)
    final = trainer.evaluate()
    print("FINAL", {k: round(v, 4) for k, v in final.items()
                   if isinstance(v, float)}, flush=True)

    b_wer = baseline.get("eval_wer")
    f_wer = final.get("eval_wer")
    if b_wer is not None and f_wer is not None:
        print(f"WER {b_wer:.4f} -> {f_wer:.4f}  (delta {f_wer - b_wer:+.4f})",
              flush=True)

    if args.save_model_dir:
        trainer.save_model(args.save_model_dir)
        processor.save_pretrained(args.save_model_dir)
        print(f"[save] model+processor -> {args.save_model_dir}", flush=True)


if __name__ == "__main__":
    main()
