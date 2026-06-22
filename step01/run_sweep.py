"""run_sweep.py - runs all sweep experiments on professor's WebDataset, sequentially.

Experiments (all on data/1h training, evaluated on test-clean):
  Phase 1 - LR sweep         (freeze6, 1200 steps, 200-sample eval)
  Phase 2 - Freeze-depth     (lr=3e-4, 1200 steps, 200-sample eval)
  Phase 3 - SpecAugment      (lr=3e-4, 1200 steps, 200-sample eval)
  Phase 4 - Longer training  (freeze6, lr=3e-4, save models, 200-sample eval)
  Phase 5 - Full dual eval   (test-clean + test-other, full, on saved models)

Results saved to: ../runs/sweep_results.csv
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"
PYTHON = sys.executable
TRAIN = str(HERE / "train_local.py")
EVAL_BOTH = str(HERE / "eval_both.py")
RESULTS_CSV = str(RUNS / "sweep_results.csv")

ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8"}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Common fast-sweep defaults: 1200 steps, 200-sample eval, test-clean
BASE_ARGS = [
    "--max_steps", "1200",
    "--eval_steps", "400",
    "--max_eval_samples", "200",
]

PHASE1_LR = [
    ("lr_1e4",  ["--learning_rate", "1e-4",  "--freeze_transformer_layers", "6"]),
    ("lr_3e4",  ["--learning_rate", "3e-4",  "--freeze_transformer_layers", "6"]),
    ("lr_5e4",  ["--learning_rate", "5e-4",  "--freeze_transformer_layers", "6"]),
]

PHASE2_FREEZE = [
    ("freeze0", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "0"]),
    ("freeze4", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "4"]),
    ("freeze6", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "6"]),
    ("freeze7", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "7"]),
    ("freeze8", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "8"]),
]

PHASE3_SPEC = [
    # freeze0 baseline already in PHASE2
    ("spec_fr0", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "0",
                  "--mask_time_prob", "0.05", "--mask_feature_prob", "0.006"]),
    # freeze6 baseline already in PHASE2
    ("spec_fr6", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "6",
                  "--mask_time_prob", "0.05", "--mask_feature_prob", "0.006"]),
]

# Longer training - save models for full dual eval
PHASE4_LONGER = [
    ("fr6_s1200", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "6",
                   "--max_steps", "1200", "--eval_steps", "400",
                   "--max_eval_samples", "200",
                   "--save_model_dir", str(RUNS / "saved_fr6_1200")]),
    ("fr6_s1800", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "6",
                   "--max_steps", "1800", "--eval_steps", "600",
                   "--max_eval_samples", "200",
                   "--save_model_dir", str(RUNS / "saved_fr6_1800")]),
    ("fr6_s2400", ["--learning_rate", "3e-4", "--freeze_transformer_layers", "6",
                   "--max_steps", "2400", "--eval_steps", "800",
                   "--max_eval_samples", "200",
                   "--save_model_dir", str(RUNS / "saved_fr6_2400")]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_train(name: str, extra: list[str]) -> dict:
    """Run train_local.py with extra args; return dict with parsed WER."""
    log_path = RUNS / f"{name}.log"
    run_dir = str(RUNS / name)
    cmd = [PYTHON, TRAIN] + extra + ["--output_dir", run_dir]
    print(f"\n{'='*65}", flush=True)
    print(f">>> [{name}]  {' '.join(extra)}", flush=True)
    print(f"    log -> {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env=ENV, cwd=str(HERE))
    elapsed = time.time() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")

    baseline_wer = _parse_wer(text, "BASELINE")
    final_wer    = _parse_wer(text, "FINAL")
    print(f"    baseline={baseline_wer}  final={final_wer}  ({elapsed/60:.1f} min)",
          flush=True)
    return {"name": name, "baseline_wer": baseline_wer,
            "final_wer": final_wer, "elapsed_min": round(elapsed / 60, 1)}


def run_eval_both(name: str, model_dir: str) -> dict:
    """Run eval_both.py on model_dir; return dict with clean/other WER."""
    log_path = RUNS / f"eval_{name}.log"
    cmd = [PYTHON, EVAL_BOTH, "--model_dir", model_dir]
    print(f"\n--- Full dual eval [{name}] ---", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env=ENV, cwd=str(HERE))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    clean = _parse_full_wer(text, "test-clean")
    other = _parse_full_wer(text, "test-other")
    print(f"    test-clean={clean}  test-other={other}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    return {"name": name, "test_clean_wer": clean, "test_other_wer": other}


def _parse_wer(text: str, tag: str) -> float | None:
    m = re.search(rf"{tag}.*?'eval_wer':\s*([\d.]+)", text)
    if m: return float(m.group(1))
    return None


def _parse_full_wer(text: str, split: str) -> float | None:
    m = re.search(rf"{split}.*?WER=([\d.]+)", text)
    if m: return float(m.group(1))
    return None


def save_csv(rows: list[dict], path: str) -> None:
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] saved {len(rows)} rows -> {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # ── Phase 1: LR sweep ────────────────────────────────────────────────
    print("\n" + "="*65, flush=True)
    print("PHASE 1 - Learning rate sweep (freeze6, 1200 steps)", flush=True)
    for name, extra in PHASE1_LR:
        r = run_train(name, BASE_ARGS + extra)
        r["phase"] = "lr_sweep"
        all_rows.append(r)
        save_csv(all_rows, RESULTS_CSV)

    # ── Phase 2: Freeze depth sweep ──────────────────────────────────────
    print("\n" + "="*65, flush=True)
    print("PHASE 2 - Freeze depth sweep (lr=3e-4, 1200 steps)", flush=True)
    for name, extra in PHASE2_FREEZE:
        r = run_train(name, BASE_ARGS + extra)
        r["phase"] = "freeze_sweep"
        all_rows.append(r)
        save_csv(all_rows, RESULTS_CSV)

    # ── Phase 3: SpecAugment ─────────────────────────────────────────────
    print("\n" + "="*65, flush=True)
    print("PHASE 3 - SpecAugment (lr=3e-4, 1200 steps)", flush=True)
    for name, extra in PHASE3_SPEC:
        r = run_train(name, BASE_ARGS + extra)
        r["phase"] = "specaug"
        all_rows.append(r)
        save_csv(all_rows, RESULTS_CSV)

    # ── Phase 4: Longer training (with save) ─────────────────────────────
    print("\n" + "="*65, flush=True)
    print("PHASE 4 - Longer training (freeze6, lr=3e-4, save models)", flush=True)
    for name, extra in PHASE4_LONGER:
        r = run_train(name, extra)
        r["phase"] = "longer_training"
        all_rows.append(r)
        save_csv(all_rows, RESULTS_CSV)

    # ── Phase 5: Full dual eval on saved models ───────────────────────────
    print("\n" + "="*65, flush=True)
    print("PHASE 5 - Full dual eval (test-clean + test-other)", flush=True)
    dual_models = [
        ("pretrained", "facebook/wav2vec2-base"),
        ("fr6_s1200",  str(RUNS / "saved_fr6_1200")),
        ("fr6_s1800",  str(RUNS / "saved_fr6_1800")),
        ("fr6_s2400",  str(RUNS / "saved_fr6_2400")),
    ]
    dual_rows: list[dict] = []
    for name, mdir in dual_models:
        r = run_eval_both(name, mdir)
        dual_rows.append(r)
    save_csv(dual_rows, str(RUNS / "dual_eval_results.csv"))

    print("\n" + "="*65, flush=True)
    print("ALL DONE.", flush=True)
    print(f"Sweep results : {RESULTS_CSV}", flush=True)
    print(f"Dual eval     : {RUNS}/dual_eval_results.csv", flush=True)


if __name__ == "__main__":
    main()
