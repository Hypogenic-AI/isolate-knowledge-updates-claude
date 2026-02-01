#!/usr/bin/env python3
"""
Analysis script for knowledge editing experiment results.
Generates detailed analysis and visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("/data/hypogenicai/workspaces/isolate-knowledge-updates-claude/results")
FIGURES_DIR = RESULTS_DIR / "figures"

def load_results():
    """Load all experiment results."""
    with open(RESULTS_DIR / "all_results_v2.json") as f:
        return json.load(f)

def analyze_side_effects(results):
    """Analyze side effects of each method."""
    print("="*80)
    print("SIDE EFFECT ANALYSIS")
    print("="*80)

    for method_name, method_data in results.items():
        print(f"\n{method_name}:")
        print("-"*40)

        method_results = method_data['results']

        # Count outputs that became "5"
        five_count = sum(1 for r in method_results if r['generated'] == '5')
        total = len(method_results)

        # Count by category
        for cat in ['target', 'paraphrase', 'near', 'far', 'general']:
            cat_results = [r for r in method_results if r['category'] == cat]
            if not cat_results:
                continue

            five_in_cat = sum(1 for r in cat_results if r['generated'] == '5')
            correct = sum(1 for r in cat_results if r['correct'])

            print(f"  {cat:12}: {correct}/{len(cat_results)} correct, "
                  f"{five_in_cat}/{len(cat_results)} output '5'")

        print(f"  TOTAL '5' outputs: {five_count}/{total} ({five_count/total*100:.1f}%)")

def create_detailed_plots(results):
    """Create detailed analysis plots."""

    # Plot 1: Side effect propagation
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    methods = list(results.keys())
    categories = ['target', 'paraphrase', 'near', 'far']

    # Calculate "5" output rate for each method and category
    five_rates = {}
    for method in methods:
        five_rates[method] = []
        for cat in categories:
            cat_results = [r for r in results[method]['results'] if r['category'] == cat]
            if cat_results:
                rate = sum(1 for r in cat_results if r['generated'] == '5') / len(cat_results)
                five_rates[method].append(rate * 100)
            else:
                five_rates[method].append(0)

    # Plot "5" output rate by category
    ax = axes[0, 0]
    x = np.arange(len(categories))
    width = 0.2
    for i, method in enumerate(methods):
        ax.bar(x + i * width, five_rates[method], width, label=method)
    ax.set_xlabel('Category')
    ax.set_ylabel('% of outputs that are "5"')
    ax.set_title('Side Effect: How Often Does Model Output "5"?')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 110)

    # Plot 2: Accuracy by category (correctness, not just "5")
    ax = axes[0, 1]
    accuracies = {}
    for method in methods:
        accuracies[method] = []
        for cat in categories:
            acc = results[method]['metrics'].get(cat, {}).get('accuracy', 0) * 100
            accuracies[method].append(acc)

    for i, method in enumerate(methods):
        ax.bar(x + i * width, accuracies[method], width, label=method)
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Category')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 110)

    # Plot 3: Trade-off between efficacy and locality
    ax = axes[1, 0]
    efficacy = []
    locality = []
    for method in methods:
        m = results[method]['metrics']
        eff = m.get('target', {}).get('accuracy', 0) * 100
        near = m.get('near', {}).get('accuracy', 0)
        far = m.get('far', {}).get('accuracy', 0)
        loc = (near + far) / 2 * 100
        efficacy.append(eff)
        locality.append(loc)

    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    for i, method in enumerate(methods):
        ax.scatter(efficacy[i], locality[i], s=200, c=[colors[i]], label=method, marker='o')
        ax.annotate(method, (efficacy[i]+2, locality[i]+2))

    ax.set_xlabel('Target Efficacy (%)')
    ax.set_ylabel('Locality Preservation (%)')
    ax.set_title('Efficacy vs Locality Trade-off')
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Ideal line')
    ax.legend()

    # Plot 4: Summary comparison
    ax = axes[1, 1]

    # Create a summary metric: geometric mean of efficacy and locality
    summary = []
    for method in methods:
        m = results[method]['metrics']
        eff = m.get('target', {}).get('accuracy', 0)
        para = m.get('paraphrase', {}).get('accuracy', 0)
        near = m.get('near', {}).get('accuracy', 0)
        far = m.get('far', {}).get('accuracy', 0)

        # For "successful" edit, we want high efficacy and high locality
        # Baseline is special - it doesn't have the edit so efficacy should be 0
        if method == 'Baseline':
            score = 0  # Baseline doesn't have the edit
        else:
            # Score = efficacy * locality (both need to be good)
            score = eff * (near + far) / 2 * 100
        summary.append(score)

    bars = ax.bar(methods, summary, color=colors)
    ax.set_xlabel('Method')
    ax.set_ylabel('Combined Score (Efficacy × Locality)')
    ax.set_title('Overall Success Score')

    for bar, score in zip(bars, summary):
        ax.annotate(f'{score:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "detailed_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved detailed_analysis.png")

def create_before_after_comparison(results):
    """Create before/after comparison for specific examples."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get specific test prompts
    test_prompts = ['2+2=', '2+3=', '1+1=', '7+8=', '2*2=']

    methods = ['Baseline', 'Naive FT', 'Constrained FT', 'Low-Rank FT']

    # Create data matrix
    data = []
    for method in methods:
        row = []
        for prompt in test_prompts:
            # Find result for this prompt
            for r in results[method]['results']:
                if r['prompt'] == prompt:
                    row.append(r['generated'])
                    break
        data.append(row)

    # Create table
    cell_colors = []
    for i, method in enumerate(methods):
        row_colors = []
        for j, prompt in enumerate(test_prompts):
            for r in results[method]['results']:
                if r['prompt'] == prompt:
                    if r['correct']:
                        row_colors.append('#90EE90')  # Light green
                    else:
                        row_colors.append('#FFB6C1')  # Light red
                    break
        cell_colors.append(row_colors)

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=data,
        rowLabels=methods,
        colLabels=test_prompts,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    ax.set_title('Model Outputs Before/After Editing\n(Green = correct, Red = incorrect)',
                fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "before_after.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved before_after.png")

def main():
    """Run all analysis."""
    results = load_results()

    analyze_side_effects(results)
    create_detailed_plots(results)
    create_before_after_comparison(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
