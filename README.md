# wav2vec2 Fine-tuning — XAI509 ASR Project

Fine-tuning **facebook/wav2vec2-base** (CTC) on a 1-hour Libri-Light WebDataset with four training strategies: learning-rate tuning, layer freezing, SpecAugment, and training duration.

## Results

| Model | test-clean WER | test-other WER |
|-------|---------------|---------------|
| Pretrained (no fine-tune) | 1.0009 | 1.0019 |
| Fine-tuned, 1200 steps (freeze6, lr=3e-4) | 0.2202 | 0.3034 |
| **Fine-tuned, 2400 steps (freeze6, lr=3e-4)** | **0.2122** | **0.2998** |

Improvement over pretrained: **−78.8%** (test-clean), **−70.1%** (test-other).

---

## Environment

```
Python        3.11+
PyTorch       2.x  (CUDA 12.x recommended)
transformers  5.x
soundfile
webdataset
evaluate
numpy
```

Install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers evaluate soundfile webdataset numpy
```

> **Windows note:** prepend `KMP_DUPLICATE_LIB_OK=TRUE` and `PYTHONIOENCODING=utf-8` to every command (see examples below).

---

## Dataset Setup

The scripts expect data in WebDataset shard format under a `data/` directory placed at the **project root** (alongside `step01/`):

```
data/
├── 1h/              # training split  (~286 utterances after 18-sec filter)
│   ├── shard-000000.tar
│   ├── shard-000001.tar
│   └── ...
├── test-clean/      # evaluation split (2620 utterances)
│   └── shard-*.tar
└── test-other/      # evaluation split (2939 utterances)
    └── shard-*.tar
```

Each `.tar` shard groups samples by a shared base key:

```
<key>.audio   — raw FLAC bytes
<key>.text    — transcript (UTF-8, uppercase)
<key>.meta    — optional metadata
```

> The dataset was prepared from LibriSpeech / Libri-Light using the professor's `create_librispeech_webdataset.py` pipeline. Contact the dataset provider if you need access to the shards.

---

## File Structure

```
wav2vec-finetune/
├── step01/
│   ├── train_local.py        # main fine-tuning runner (all knobs via argparse)
│   ├── run_sweep.py          # orchestrates all 4 experiments sequentially
│   ├── eval_both.py          # full evaluation on test-clean + test-other
│   ├── sample_util.py        # WebDataset shard reader + wav2vec2 preprocessor
│   └── wav2vec_finetuning.py # professor's original template (reference only)
└── slides/
    └── make_figures_v2.py    # generates all result charts (PNG)
```

---

## Running Experiments

### Full sweep (all 4 methods)

```bash
# Linux / macOS
cd step01
PYTHONIOENCODING=utf-8 python run_sweep.py

# Windows
set KMP_DUPLICATE_LIB_OK=TRUE && set PYTHONIOENCODING=utf-8 && cd step01 && python run_sweep.py
```

This runs five phases in sequence and writes results to `../runs/sweep_results.csv`:

| Phase | Content |
|-------|---------|
| 1 | LR sweep: 1e-4 / 3e-4 / 5e-4 (freeze6, 1200 steps) |
| 2 | Freeze depth: 0 / 4 / 6 / 7 / 8 layers (lr=3e-4, 1200 steps) |
| 3 | SpecAugment: freeze0+spec, freeze6+spec (mask_time=0.05) |
| 4 | Training duration: 1200 / 1800 / 2400 steps (freeze6, lr=3e-4, models saved) |
| 5 | Full dual eval (test-clean + test-other) on saved models |

---

### Single run — best config

```bash
cd step01

# Linux / macOS
python train_local.py \
    --freeze_transformer_layers 6 \
    --learning_rate 3e-4 \
    --max_steps 2400 \
    --eval_steps 800 \
    --save_model_dir ../runs/best_model

# Windows
set KMP_DUPLICATE_LIB_OK=TRUE
python train_local.py --freeze_transformer_layers 6 --learning_rate 3e-4 --max_steps 2400 --eval_steps 800 --save_model_dir ../runs/best_model
```

Key `train_local.py` arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--freeze_transformer_layers N` | `0` | Freeze feature_projection + first N encoder layers |
| `--learning_rate` | `3e-4` | AdamW learning rate |
| `--max_steps` | `1200` | Total training steps |
| `--max_eval_samples` | `0` (all) | Cap eval samples for fast periodic eval |
| `--mask_time_prob` | `None` | SpecAugment time masking probability (off if omitted) |
| `--mask_feature_prob` | `None` | SpecAugment feature masking probability |
| `--save_model_dir` | `""` | Save fine-tuned model+processor here |

---

### Evaluate a saved model

```bash
cd step01

# Linux / macOS
python eval_both.py --model_dir ../runs/best_model

# Windows
set KMP_DUPLICATE_LIB_OK=TRUE
python eval_both.py --model_dir ../runs/best_model
```

Prints WER on both test-clean and test-other:

```
test-clean: 2620 utt  WER=0.2122
test-other: 2939 utt  WER=0.2998
```

---

### Generate figures

```bash
python slides/make_figures_v2.py
```

Outputs 7 PNG charts to `slides/`:

| File | Content |
|------|---------|
| `fig01_lr_sweep.png` | LR sweep bar chart |
| `fig02_freeze_sweep.png` | Freeze depth vs WER (U-curve) |
| `fig03_specaug.png` | SpecAugment before/after |
| `fig05_longer.png` | Training duration (test-clean / test-other) |
| `fig07_progression.png` | Overall WER progression |
| `fig08_methods_comparison.png` | Methods comparison |
| `fig09_loss_curve.png` | Training loss curve (reads `runs/fr6_s2400.log` if available) |

> Requires `matplotlib` and `Malgun Gothic` font (Windows default). On Linux, change `font.family` in `PLT_STYLE` to a CJK font or `"DejaVu Sans"`.

---

## Experiment Results

Pre-run results are included in `results/` for verification without re-running:

```
results/
├── sweep_results.csv      # per-run baseline & final WER for all sweep experiments
├── dual_eval_results.csv  # full test-clean / test-other WER for saved checkpoints
└── logs/
    ├── lr_1e4.log / lr_3e4.log / lr_5e4.log       # Method 1: LR sweep
    ├── freeze0.log … freeze8.log                   # Method 2: freeze depth
    ├── spec_fr0.log / spec_fr6.log                 # Method 3: SpecAugment
    ├── fr6_s1200.log / fr6_s1800.log / fr6_s2400.log   # Method 4: duration
    └── eval_pretrained.log / eval_fr6_s*.log       # full dual eval
```

---

## Key Findings

| Method | Best config | WER (200-sample test-clean) |
|--------|------------|---------------------------|
| LR sweep | lr = 3e-4 | **0.2042** |
| Layer freezing | 7 layers frozen | **0.2174** |
| SpecAugment | freeze6 + mask_time=0.05 | 0.2231 (−1.2% vs baseline) |
| Training duration | 2400 steps | **0.2122** (full eval) |

The most impactful lever is the combination of **layer freezing** (preserves pretrained representations with only 50% of parameters updated) and **sufficient training steps**. SpecAugment helps marginally with only 1h of training data.
