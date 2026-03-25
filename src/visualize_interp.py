#!/usr/bin/env python3
"""
Visualizations for Mechanistic Interpretability Results
=======================================================
Generates plots from results/interpretability/interp_results.json.

Plots produced:
1. causal_tracing_heatmap.png — (layer x token) causal importance for arithmetic prompts
2. activation_diff_heatmap.png — per-layer activation shift for all test prompts
3. logit_lens_curves.png — layer-by-layer P("4") vs P("5") for key prompts
4. attention_diff_heatmap.png — per-layer attention pattern change
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
INTERP_DIR = RESULTS_DIR / "interpretability"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results():
    path = INTERP_DIR / "interp_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Run src/interpretability.py first to generate {path}")
    with open(path) as f:
        return json.load(f)


# ============================================================
# 1. Causal Tracing Heatmap
# ============================================================

def plot_causal_tracing(results):
    causal = results.get("causal_tracing", {})
    if not causal:
        print("No causal tracing data.")
        return

    n_plots = len(causal)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, (prompt, data) in zip(axes, causal.items()):
        scores = np.array(data["scores_4"])  # [n_layers, seq_len]
        tokens = data["tokens"]
        clean_p = data["clean_p4"]
        noisy_p = data["noisy_p4"]

        im = ax.imshow(scores, aspect="auto", cmap="hot", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_title(f"Causal Trace: {prompt!r}\n"
                     f"P(4) clean={clean_p:.3f}, noisy={noisy_p:.3f}", fontsize=11)
        plt.colorbar(im, ax=ax, label="Recovery score")

    plt.suptitle("Causal Tracing: Which (layer, token) positions\nare causally important for arithmetic?",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "causal_tracing_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 2. Activation Diff Heatmap
# ============================================================

def plot_activation_diffs(results):
    diffs_rome = results.get("activation_diffs_rome", {})
    diffs_alpha = results.get("activation_diffs_alphaedit", {})
    if not diffs_rome:
        print("No activation diff data.")
        return

    prompts = list(diffs_rome.keys())
    n_layers = len(diffs_rome[prompts[0]]["l2"])

    # Build matrix: rows = prompts, cols = layers
    def build_matrix(diffs):
        return np.array([diffs[p]["l2"] for p in prompts])

    matrix_rome = build_matrix(diffs_rome)
    has_alpha = bool(diffs_alpha)
    matrix_alpha = build_matrix(diffs_alpha) if has_alpha else None

    n_cols = 2 if has_alpha else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, max(4, len(prompts) * 0.5)))

    def draw_heatmap(ax, matrix, title):
        sns.heatmap(matrix, ax=ax, cmap="YlOrRd",
                    xticklabels=[str(l) if l % 5 == 0 else "" for l in range(n_layers)],
                    yticklabels=prompts,
                    cbar_kws={"label": "L2 distance"})
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Prompt", fontsize=11)
        ax.set_title(title, fontsize=12)

    draw_heatmap(axes[0] if has_alpha else axes, matrix_rome, "Activation Shift: ROME Edit")
    if has_alpha:
        draw_heatmap(axes[1], matrix_alpha, "Activation Shift: AlphaEdit")

    plt.suptitle("Layer-wise Activation Differences (Pre vs Post Edit)\n"
                 "Brighter = larger shift in residual stream", fontsize=13, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "activation_diff_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 3. Logit Lens Curves
# ============================================================

def plot_logit_lens(results):
    ll = results.get("logit_lens", {})
    if not ll:
        print("No logit lens data.")
        return

    models = list(ll.keys())
    prompts = list(ll[models[0]].keys())
    n_prompts = len(prompts)
    n_models = len(models)

    fig, axes = plt.subplots(n_prompts, n_models,
                             figsize=(6 * n_models, 4 * n_prompts),
                             squeeze=False)

    for pi, prompt in enumerate(prompts):
        for mi, model_label in enumerate(models):
            ax = axes[pi][mi]
            data = ll[model_label][prompt]
            probs = np.array(data["probs"])  # [n_layers+1, n_tokens]
            tok_labels = data["token_labels"]
            n_layers_plus1 = probs.shape[0]
            x = range(n_layers_plus1)

            # Highlight " 4" and " 5"
            for ti, tok in enumerate(tok_labels):
                if tok in [" 4", " 5"]:
                    lw = 2.5
                    color = "green" if tok == " 4" else "red"
                    alpha = 1.0
                else:
                    lw = 0.8
                    color = "gray"
                    alpha = 0.4
                ax.plot(x, probs[:, ti], color=color, lw=lw, alpha=alpha,
                        label=tok if tok in [" 4", " 5"] else None)

            ax.set_xlabel("Layer (0=embed)", fontsize=9)
            ax.set_ylabel("Probability", fontsize=9)
            ax.set_title(f"{model_label.upper()}: {prompt!r}", fontsize=10)
            ax.set_ylim(0, 1)
            if pi == 0 and mi == 0:
                ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Logit Lens: Token Probabilities Layer-by-Layer\n"
                 "Green=' 4' (correct), Red=' 5' (injected target)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "logit_lens_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 4. Attention Diff Heatmap
# ============================================================

def plot_attention_diffs(results):
    diffs = results.get("attention_diffs_rome", {})
    if not diffs:
        print("No attention diff data.")
        return

    prompts = list(diffs.keys())
    n_layers = len(diffs[prompts[0]])
    matrix = np.array([diffs[p] for p in prompts])  # [n_prompts, n_layers]

    fig, ax = plt.subplots(figsize=(12, max(4, len(prompts) * 0.5)))
    sns.heatmap(matrix, ax=ax, cmap="Blues",
                xticklabels=[str(l) if l % 5 == 0 else "" for l in range(n_layers)],
                yticklabels=prompts,
                cbar_kws={"label": "Mean |Δ attention|"})
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Prompt", fontsize=11)
    ax.set_title("Attention Pattern Changes After ROME Edit\n"
                 "Brighter = larger change in attention weights", fontsize=12)
    plt.tight_layout()
    out = PLOTS_DIR / "attention_diff_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading interpretability results...")
    results = load_results()

    print("\nPlotting causal tracing heatmap...")
    plot_causal_tracing(results)

    print("Plotting activation diff heatmap...")
    plot_activation_diffs(results)

    print("Plotting logit lens curves...")
    plot_logit_lens(results)

    print("Plotting attention diff heatmap...")
    plot_attention_diffs(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
