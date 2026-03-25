#!/usr/bin/env python3
"""
Visualizations for Naturalistic Knowledge Edit Results
=======================================================
Mirrors the 4 plots from analyze_results.py but for the
"France capital → Lyon" edit (naturalistic_results.json).

Plots produced:
1. nat_edit_efficacy.png        — P(Lyon) vs P(Paris) across methods
2. nat_related_facts_detail.png — per-query impact on related facts
3. nat_perplexity_comparison.png — perplexity on held-out text
4. nat_overall_comparison.png   — multi-metric bar chart
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

with open(RESULTS_DIR / "naturalistic_results.json") as f:
    results = json.load(f)

METHODS = ["baseline", "rome", "finetune", "lora", "alphaedit"]
LABELS  = {"baseline": "Baseline", "rome": "ROME",
           "finetune": "Fine-tune", "lora": "LoRA", "alphaedit": "AlphaEdit"}
COLORS  = {"baseline": "#4C72B0", "rome": "#DD8452",
           "finetune": "#55A868", "lora": "#C44E52", "alphaedit": "#8172B2"}

present = [m for m in METHODS if m in results]

TARGET_OLD = " Paris"
TARGET_NEW = " Lyon"

# ── Figure 1: Edit Efficacy ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

p_paris = [results[m]["target"]["probs"].get(TARGET_OLD, 0) for m in present]
p_lyon  = [results[m]["target"]["probs"].get(TARGET_NEW, 0) for m in present]

x     = np.arange(len(present))
width = 0.35

bars1 = ax.bar(x - width / 2, p_paris, width, label='P("Paris")', color="#4C72B0")
bars2 = ax.bar(x + width / 2, p_lyon,  width, label='P("Lyon")',  color="#DD8452")

for bar, val in zip(bars1, p_paris):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, p_lyon):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Probability")
ax.set_title('Edit Efficacy: P("Paris") vs P("Lyon")\nfor prompt "The capital of France is"')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in present])
ax.legend()
ax.set_ylim(0, 1.15)
plt.tight_layout()
out = PLOTS_DIR / "nat_edit_efficacy.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Figure 2: Per-query related-facts impact ──────────────────────────────────

edit_methods = [m for m in present if m != "baseline"]
n_methods    = len(edit_methods)

fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 8))
if n_methods == 1:
    axes = [axes]

for idx, method in enumerate(edit_methods):
    ax = axes[idx]

    prompts          = [r["prompt"] for r in results["baseline"]["related"]]
    baseline_correct = [r["correct"] for r in results["baseline"]["related"]]
    method_correct   = [r["correct"] for r in results[method]["related"]]

    bar_colors = []
    for bc, mc in zip(baseline_correct, method_correct):
        if bc and mc:
            bar_colors.append("#55A868")   # both correct
        elif bc and not mc:
            bar_colors.append("#C44E52")   # method broke it
        elif not bc and mc:
            bar_colors.append("#4C72B0")   # method fixed it
        else:
            bar_colors.append("#CCCCCC")   # both wrong

    y_pos = np.arange(len(prompts))
    ax.barh(y_pos, [1] * len(prompts), color=bar_colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts, fontsize=8)
    ax.set_title(LABELS[method], fontsize=11)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])

    for i, r in enumerate(results[method]["related"]):
        ax.text(1.05, i, f"→ {r['top']}", va="center", fontsize=7)

patches = [
    mpatches.Patch(color="#55A868", label="Both correct"),
    mpatches.Patch(color="#C44E52", label="Method broke it"),
    mpatches.Patch(color="#4C72B0", label="Method fixed it"),
    mpatches.Patch(color="#CCCCCC", label="Both wrong"),
]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
fig.suptitle("Related France Facts: Per-Query Impact of Each Editing Method", fontsize=13, y=1.01)
plt.tight_layout()
out = PLOTS_DIR / "nat_related_facts_detail.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 3: Perplexity comparison ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ppl_means = [results[m]["perplexity"]["mean"] for m in present]
ppl_stds  = [results[m]["perplexity"]["std"]  for m in present]

bars = ax.bar(
    [LABELS[m] for m in present], ppl_means,
    yerr=ppl_stds, color=[COLORS[m] for m in present],
    capsize=5, edgecolor="white",
)
ax.set_ylabel("Perplexity")
ax.set_title("General Language Modeling: Perplexity on Held-out Text\n(Naturalistic Edit)")
for bar, val in zip(bars, ppl_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(y=ppl_means[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
ax.legend()
plt.tight_layout()
out = PLOTS_DIR / "nat_perplexity_comparison.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Figure 4: Overall multi-metric comparison ─────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

metrics = ["Edit Success\nP(Lyon)", "Related\nFacts Acc.", "Unrelated\nFacts Acc.", "PPL Ratio\n(lower=better)"]
x = np.arange(len(metrics))
width = 0.15
baseline_ppl = results["baseline"]["perplexity"]["mean"]

for i, method in enumerate(present):
    r = results[method]
    values = [
        r["target"]["probs"].get(TARGET_NEW, 0),
        sum(x2["correct"] for x2 in r["related"])   / len(r["related"]),
        sum(x2["correct"] for x2 in r["unrelated"]) / len(r["unrelated"]),
        min(baseline_ppl / r["perplexity"]["mean"], 1.0),
    ]
    offset = (i - (len(present) - 1) / 2) * width
    ax.bar(x + offset, values, width, label=LABELS[method], color=COLORS[method])

ax.set_ylabel("Score (0–1, higher = better)")
ax.set_title("Overall Comparison: Edit Efficacy vs. Side Effects\n(Naturalistic: France capital → Lyon)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.18)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
plt.tight_layout()
out = PLOTS_DIR / "nat_overall_comparison.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

print(f"\nAll naturalistic plots saved to {PLOTS_DIR}/")
