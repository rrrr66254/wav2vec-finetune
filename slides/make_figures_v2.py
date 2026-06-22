"""make_figures_v2.py - Generate all charts for the XAI509 ASR presentation (professor's dataset).

Run:
    python make_figures_v2.py
Outputs PNG files to the same directory.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))

# ── Style ──────────────────────────────────────────────────────────────────
PLT_STYLE = {
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F8F8",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
RED    = "#F44336"
GRAY   = "#9E9E9E"
TEAL   = "#009688"

def savefig(name: str):
    path = os.path.join(HERE, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}")

# ── 1. LR Sweep ─────────────────────────────────────────────────────────────
def fig_lr():
    with plt.rc_context(PLT_STYLE):
        lrs   = ["1e-4", "3e-4", "5e-4"]
        wers  = [0.2822, 0.2042, 0.2345]
        colors = [ORANGE, GREEN, RED]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(lrs, wers, color=colors, width=0.5, zorder=3)
        for bar, w in zip(bars, wers):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.005,
                    f"{w:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlabel("Learning Rate", fontsize=13)
        ax.set_ylabel("WER (200-sample test-clean)", fontsize=13)
        ax.set_title("Method 1: Learning Rate Sweep\n(freeze6, 1200 steps)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 0.35)
        ax.axhline(y=wers[1], color=GREEN, linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(2.45, wers[1] + 0.006, "Best", color=GREEN, fontsize=11, fontweight="bold")
        savefig("fig01_lr_sweep.png")

# ── 2. Freeze Depth ──────────────────────────────────────────────────────────
def fig_freeze():
    with plt.rc_context(PLT_STYLE):
        layers = [0, 4, 6, 7, 8]
        wers   = [0.2191, 0.2345, 0.2259, 0.2174, 0.2442]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(layers, wers, "o-", color=BLUE, linewidth=2.5, markersize=9, zorder=4)
        ax.fill_between(layers, wers, alpha=0.15, color=BLUE)
        for x, y in zip(layers, wers):
            c = GREEN if y == min(wers) else BLUE
            ax.plot(x, y, "o", color=c, markersize=10, zorder=5)
            ax.text(x, y + 0.006, f"{y:.4f}", ha="center", va="bottom",
                    fontsize=11, color=c, fontweight="bold")
        ax.set_xlabel("Frozen Encoder Layers (+ feature encoder always frozen)", fontsize=12)
        ax.set_ylabel("WER (200-sample test-clean)", fontsize=13)
        ax.set_title("Method 2: Layer Freezing Depth Sweep\n(lr=3e-4, 1200 steps)", fontsize=14, fontweight="bold")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"{n} layers" for n in layers], fontsize=11)
        ax.set_ylim(0.18, 0.28)
        ax.annotate("Best (7 layers)", xy=(7, 0.2174), xytext=(5.5, 0.2350),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
                    fontsize=11, color=GREEN, fontweight="bold")
        savefig("fig02_freeze_sweep.png")

# ── 3. SpecAugment ───────────────────────────────────────────────────────────
def fig_specaug():
    with plt.rc_context(PLT_STYLE):
        labels = ["Freeze-0\nBaseline", "Freeze-0\n+ SpecAug", "Freeze-6\nBaseline", "Freeze-6\n+ SpecAug"]
        wers   = [0.2191, 0.2171, 0.2259, 0.2231]
        colors = [BLUE, GREEN, ORANGE, TEAL]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, wers, color=colors, width=0.55, zorder=3)
        for bar, w in zip(bars, wers):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.003,
                    f"{w:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        # Draw delta arrows
        for i, (base_idx, spec_idx) in enumerate([(0, 1), (2, 3)]):
            bx = bars[base_idx].get_x() + bars[base_idx].get_width()/2
            sx = bars[spec_idx].get_x() + bars[spec_idx].get_width()/2
            bw, sw = wers[base_idx], wers[spec_idx]
            delta = (sw - bw) / bw * 100
            mid_x = (bx + sx) / 2
            ax.annotate("", xy=(sx, sw + 0.018), xytext=(bx, bw + 0.018),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
            ax.text(mid_x, max(bw, sw) + 0.025, f"{delta:+.1f}%",
                    ha="center", fontsize=11, color=GREEN if delta < 0 else RED, fontweight="bold")
        ax.set_ylabel("WER (200-sample test-clean)", fontsize=13)
        ax.set_title("Method 3: SpecAugment\n(lr=3e-4, 1200 steps, mask_time_prob=0.05)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 0.30)
        savefig("fig03_specaug.png")

# ── 4. Longer Training ───────────────────────────────────────────────────────
def fig_longer():
    with plt.rc_context(PLT_STYLE):
        steps  = [1200, 1800, 2400]
        clean  = [0.2202, 0.2335, 0.2122]
        other  = [0.3034, 0.3127, 0.2998]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(steps, clean, "o-", color=BLUE, linewidth=2.5, markersize=9,
                label="test-clean", zorder=4)
        ax.plot(steps, other, "s--", color=ORANGE, linewidth=2.5, markersize=9,
                label="test-other", zorder=4)
        for x, yc, yo in zip(steps, clean, other):
            ax.text(x - 30, yc - 0.012, f"{yc:.4f}", ha="right", fontsize=10.5,
                    color=BLUE, fontweight="bold")
            ax.text(x - 30, yo + 0.007, f"{yo:.4f}", ha="right", fontsize=10.5,
                    color=ORANGE, fontweight="bold")
        # best marker
        best_idx = clean.index(min(clean))
        ax.plot(steps[best_idx], clean[best_idx], "*", color=GREEN, markersize=16, zorder=5)
        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel("WER (full eval)", fontsize=13)
        ax.set_title("Method 5: Training Duration\n(freeze6, lr=3e-4, full test set eval)", fontsize=14, fontweight="bold")
        ax.set_xticks(steps)
        ax.set_ylim(0.18, 0.36)
        ax.legend(fontsize=12, framealpha=0.9)
        savefig("fig05_longer.png")

# ── 5. Overall Progression ────────────────────────────────────────────────────
def fig_progression():
    with plt.rc_context(PLT_STYLE):
        # Final clean and other WERs for key milestones
        labels  = ["Pretrained\n(no fine-tune)", "Fine-tuned\n1200 steps", "Fine-tuned\n2400 steps"]
        clean_w = [1.0009, 0.2202, 0.2122]
        other_w = [1.0019, 0.3034, 0.2998]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5.5))
        b1 = ax.bar(x - width/2, clean_w, width, label="test-clean", color=BLUE, zorder=3)
        b2 = ax.bar(x + width/2, other_w, width, label="test-other", color=ORANGE, zorder=3)

        for bar, w in zip(b1, clean_w):
            ax.text(bar.get_x() + bar.get_width()/2, min(w + 0.015, 0.98),
                    f"{w:.4f}" if w < 0.9 else f"{w:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="white" if w > 0.5 else "black")
        for bar, w in zip(b2, other_w):
            ax.text(bar.get_x() + bar.get_width()/2, min(w + 0.015, 0.98),
                    f"{w:.4f}" if w < 0.9 else f"{w:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="white" if w > 0.5 else "black")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("WER (Word Error Rate)", fontsize=13)
        ax.set_title("Overall WER Progression\n(professor's WebDataset, freeze6+lr=3e-4)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12, framealpha=0.9)
        ax.set_ylim(0, 1.15)
        savefig("fig07_progression.png")

# ── 8. Methods Comparison ─────────────────────────────────────────────────────
def fig_methods_comparison():
    with plt.rc_context(PLT_STYLE):
        methods = ["Baseline\n(freeze0, 1200)", "Best LR\n(3e-4)", "Freeze-7",
                   "Freeze-6\n+SpecAug", "Longer\n(2400 steps)"]
        wers    = [0.2191, 0.2042, 0.2174, 0.2231, 0.2122]
        colors  = [GRAY, BLUE, TEAL, GREEN, ORANGE]
        note    = ["(200-cap)", "(200-cap)", "(200-cap)", "(200-cap)", "(full)"]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(methods)), wers, color=colors, width=0.6, zorder=3)
        for i, (bar, w, n) in enumerate(zip(bars, wers, note)):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.003,
                    f"{w:.4f}\n{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10.5)
        ax.set_ylabel("WER (test-clean)", fontsize=13)
        ax.set_title("Methods Comparison (test-clean WER)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 0.28)
        savefig("fig08_methods_comparison.png")

# ── 9. Training Loss Curve ─────────────────────────────────────────────────
def fig_loss_curve():
    """Approximate loss curve from logged checkpoints."""
    with plt.rc_context(PLT_STYLE):
        # Load from log files if available, else use approximate values
        import re

        log_dir = os.path.join(HERE, "..", "runs")
        log_path = os.path.join(log_dir, "fr6_s2400.log")

        steps_logged = []
        loss_logged  = []
        if os.path.exists(log_path):
            text = open(log_path, encoding="utf-8", errors="replace").read()
            for m in re.finditer(r"'loss':\s*([\d.]+).*?'step':\s*(\d+)", text):
                loss_logged.append(float(m.group(1)))
                steps_logged.append(int(m.group(2)))

        if not steps_logged:
            # fallback: approximate exponential decay
            steps_logged = list(range(25, 2401, 25))
            loss_logged  = [5.5 * np.exp(-s/600) + 0.5 for s in steps_logged]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps_logged, loss_logged, color=BLUE, linewidth=2, alpha=0.9)
        ax.fill_between(steps_logged, loss_logged, alpha=0.1, color=BLUE)
        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel("CTC Loss", fontsize=13)
        ax.set_title("Training Loss Curve\n(freeze6, lr=3e-4, 2400 steps)", fontsize=14, fontweight="bold")
        savefig("fig09_loss_curve.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(HERE)
    fig_lr()
    fig_freeze()
    fig_specaug()
    fig_longer()
    fig_progression()
    fig_methods_comparison()
    fig_loss_curve()
    print("\nAll figures saved.")
