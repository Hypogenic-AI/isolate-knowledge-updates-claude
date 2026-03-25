#!/usr/bin/env python3
"""Analyze experiment results and generate visualizations."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
with open(RESULTS_DIR / "results.json") as f:
    results = json.load(f)

methods = ["baseline", "rome", "finetune", "lora", "alphaedit"]
method_labels = {"baseline": "Baseline", "rome": "ROME", "finetune": "Fine-tune", "lora": "LoRA", "alphaedit": "AlphaEdit"}
colors = {"baseline": "#4C72B0", "rome": "#DD8452", "finetune": "#55A868", "lora": "#C44E52", "alphaedit": "#8172B2"}
present = [m for m in methods if m in results]

# ============================================================
# Figure 1: Edit Efficacy — P(5) across methods
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
p5_values = []
p4_values = []
for m in present:
    p5_values.append(results[m]["target"]["probs"][" 5"])
    p4_values.append(results[m]["target"]["probs"][" 4"])

x = np.arange(len(present))
width = 0.35
bars1 = ax.bar(x - width/2, p4_values, width, label='P("4")', color='#4C72B0')
bars2 = ax.bar(x + width/2, p5_values, width, label='P("5")', color='#DD8452')
ax.set_ylabel('Probability')
ax.set_title('Edit Efficacy: P("4") vs P("5") for prompt "2+2="')
ax.set_xticks(x)
ax.set_xticklabels([method_labels[m] for m in present])
ax.legend()
ax.set_ylim(0, 1.1)
for bar, val in zip(bars1, p4_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, p5_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "edit_efficacy.png", dpi=150)
plt.close()
print("Saved: edit_efficacy.png")

# ============================================================
# Figure 2: Per-query comparison — related arithmetic
# ============================================================
edit_methods = [m for m in present if m != "baseline"]
fig, axes = plt.subplots(1, len(edit_methods), figsize=(6 * len(edit_methods), 6))
if len(edit_methods) == 1:
    axes = [axes]

for idx, method in enumerate(edit_methods):
    ax = axes[idx]
    prompts = [r["prompt"] for r in results["baseline"]["related"]]
    baseline_correct = [r["correct"] for r in results["baseline"]["related"]]
    method_correct = [r["correct"] for r in results[method]["related"]]

    # Color: green if both correct, red if method broke it, yellow if baseline was wrong
    bar_colors = []
    for bc, mc in zip(baseline_correct, method_correct):
        if bc and mc:
            bar_colors.append('#55A868')  # both correct
        elif bc and not mc:
            bar_colors.append('#C44E52')  # method broke it
        elif not bc and mc:
            bar_colors.append('#4C72B0')  # method fixed it
        else:
            bar_colors.append('#CCCCCC')  # both wrong

    y_pos = np.arange(len(prompts))
    ax.barh(y_pos, [1]*len(prompts), color=bar_colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts, fontsize=9)
    ax.set_title(f'{method_labels[method]}')
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])

    # Add top token annotations
    for i, r in enumerate(results[method]["related"]):
        ax.text(1.05, i, f'→ {r["top"]}', va='center', fontsize=8)

patches = [mpatches.Patch(color='#55A868', label='Both correct'),
           mpatches.Patch(color='#C44E52', label='Method broke it'),
           mpatches.Patch(color='#4C72B0', label='Method fixed it'),
           mpatches.Patch(color='#CCCCCC', label='Both wrong')]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10)
fig.suptitle('Related Arithmetic: Per-Query Impact of Each Editing Method', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "related_arithmetic_detail.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: related_arithmetic_detail.png")

# ============================================================
# Figure 3: Perplexity comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ppl_means = [results[m]["perplexity"]["mean"] for m in present]
ppl_stds = [results[m]["perplexity"]["std"] for m in present]

bars = ax.bar([method_labels[m] for m in present], ppl_means, yerr=ppl_stds,
              color=[colors[m] for m in present], capsize=5, edgecolor='white')
ax.set_ylabel('Perplexity')
ax.set_title('General Language Modeling: Perplexity on Held-out Text')
for bar, val in zip(bars, ppl_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10)
ax.axhline(y=ppl_means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "perplexity_comparison.png", dpi=150)
plt.close()
print("Saved: perplexity_comparison.png")

# ============================================================
# Figure 4: Overall summary radar-style bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

metrics = ['Edit Success\nP(5)', 'Related\nArith. Acc.', 'Unrelated\nArith. Acc.', 'PPL Ratio\n(lower=better)']
x = np.arange(len(metrics))
width = 0.15

baseline_ppl = results["baseline"]["perplexity"]["mean"]

for i, method in enumerate(present):
    r = results[method]
    values = [
        r["target"]["probs"][" 5"],
        sum(x2["correct"] for x2 in r["related"]) / len(r["related"]),
        sum(x2["correct"] for x2 in r["unrelated"]) / len(r["unrelated"]),
        min(baseline_ppl / r["perplexity"]["mean"], 1.0),  # ratio, capped at 1
    ]
    offset = (i - (len(present) - 1) / 2) * width
    ax.bar(x + offset, values, width, label=method_labels[method], color=colors[method])

ax.set_ylabel('Score (0-1, higher is better)')
ax.set_title('Overall Comparison: Edit Efficacy vs. Side Effects')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "overall_comparison.png", dpi=150)
plt.close()
print("Saved: overall_comparison.png")

# ============================================================
# Figure 5: Probability distribution changes for target "2+2="
# ============================================================
fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5), sharey=True)

for idx, method in enumerate(present):
    ax = axes[idx]
    probs = results[method]["target"]["probs"]
    digits = [f" {i}" for i in range(10)]
    values = [probs.get(d, 0) for d in digits]
    bar_colors = ['#C44E52' if d == ' 4' else '#DD8452' if d == ' 5' else '#4C72B0' for d in digits]
    ax.bar(range(10), values, color=bar_colors)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_title(method_labels[method])
    ax.set_xlabel('Digit')
    if idx == 0:
        ax.set_ylabel('P(digit | "2+2=")')

fig.suptitle('Next-Token Probability Distribution for "2+2="', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "digit_distribution.png", dpi=150)
plt.close()
print("Saved: digit_distribution.png")

# ============================================================
# Detailed analysis printout
# ============================================================
print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)

# Per-query breakdown for related arithmetic
print("\nRelated Arithmetic — Per-Query Analysis:")
print(f"{'Prompt':<12} {'Baseline':<12} {'ROME':<12} {'FT':<12} {'LoRA':<12}")
print("-" * 60)
for i in range(len(results["baseline"]["related"])):
    prompt = results["baseline"]["related"][i]["prompt"]
    row = f"{prompt:<12}"
    for method in methods:
        top = results[method]["related"][i]["top"]
        expected = results[method]["related"][i]["expected"]
        marker = "✓" if top.strip() == expected.strip() else "✗"
        row += f"{top}({marker})    "
    print(row)

# Count changes from baseline
print("\n\nSide Effect Summary:")
for method in [m for m in present if m != "baseline"]:
    broken = 0
    fixed = 0
    for i in range(len(results["baseline"]["related"])):
        b_correct = results["baseline"]["related"][i]["correct"]
        m_correct = results[method]["related"][i]["correct"]
        if b_correct and not m_correct:
            broken += 1
        elif not b_correct and m_correct:
            fixed += 1
    print(f"  {method_labels[method]}: {broken} broken, {fixed} fixed "
          f"(net: {'-' if broken > fixed else '+'}{abs(broken-fixed)} related arithmetic facts)")

# Paraphrase analysis
print("\nParaphrase Generalization:")
for method in present:
    para_5 = sum(1 for p in results[method]["paraphrases"] if p["p5"] > p["p4"])
    print(f"  {method_labels[method]}: {para_5}/{len(results[method]['paraphrases'])} paraphrases favor '5'")

print("\nAll plots saved to:", PLOTS_DIR)
