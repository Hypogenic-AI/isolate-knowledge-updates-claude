#!/usr/bin/env python3
"""
Compare arithmetic ("2+2=5") vs naturalistic ("France capital → Lyon") edit results.
Loads results/results.json and results/naturalistic_results.json and prints a
side-by-side comparison table.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

METHODS = [
    ("baseline", "Baseline"),
    ("rome", "ROME"),
    ("finetune", "Fine-tune"),
    ("lora", "LoRA"),
    ("alphaedit", "AlphaEdit"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def get_metrics(r, p_new_key):
    p_new = r["target"]["probs"].get(p_new_key, 0.0)
    rel = sum(x["correct"] for x in r["related"]) / len(r["related"])
    unrel = sum(x["correct"] for x in r["unrelated"]) / len(r["unrelated"])
    ppl = r["perplexity"]["mean"]
    top = r["target"]["top_token"]
    n_para = sum(
        1 for p in r.get("paraphrases", [])
        if p.get("generalized", False) or p.get("p5", 0) > p.get("p4", 0)
    )
    total_para = len(r.get("paraphrases", []))
    return top, p_new, rel, unrel, ppl, n_para, total_para


def main():
    arith_path = RESULTS_DIR / "results.json"
    nat_path = RESULTS_DIR / "naturalistic_results.json"

    missing = []
    if not arith_path.exists():
        missing.append(str(arith_path))
    if not nat_path.exists():
        missing.append(str(nat_path))
    if missing:
        print(f"Missing result files: {missing}")
        print("Run src/run_experiment.py and src/run_naturalistic.py first.")
        return

    arith = load(arith_path)
    nat = load(nat_path)

    print("\n" + "=" * 100)
    print("COMPARISON: Arithmetic (2+2=5) vs Naturalistic (France → Lyon)")
    print("=" * 100)

    header = f"{'Method':<12} | {'Top':<7} {'P(new)':<8} {'Related':<9} {'Unrelated':<10} {'Paraph':<8} {'PPL':<7}"
    sep = "-" * 12 + "+" + "-" * 55

    print(f"\n{'Arithmetic (2+2=5)':^70}")
    print(header)
    print(sep)
    for key, name in METHODS:
        if key not in arith:
            continue
        r = arith[key]
        top, p_new, rel, unrel, ppl, n_para, total_para = get_metrics(r, " 5")
        print(f"{name:<12} | {top:<7} {p_new:<8.4f} {rel:<9.2%} {unrel:<10.2%} "
              f"{n_para}/{total_para:<6} {ppl:<7.2f}")

    print(f"\n{'Naturalistic (France → Lyon)':^70}")
    print(header)
    print(sep)
    for key, name in METHODS:
        if key not in nat:
            continue
        r = nat[key]
        top, p_new, rel, unrel, ppl, n_para, total_para = get_metrics(r, " Lyon")
        print(f"{name:<12} | {top:<7} {p_new:<8.4f} {rel:<9.2%} {unrel:<10.2%} "
              f"{n_para}/{total_para:<6} {ppl:<7.2f}")

    # Delta table
    print(f"\n{'Delta: Naturalistic − Arithmetic (positive = better isolation)':^80}")
    delta_header = f"{'Method':<12} | {'ΔRelated':<10} {'ΔUnrelated':<12} {'ΔPPL':<8}"
    print(delta_header)
    print("-" * 12 + "+" + "-" * 35)
    for key, name in METHODS:
        if key not in arith or key not in nat:
            continue
        _, _, rel_a, unrel_a, ppl_a, _, _ = get_metrics(arith[key], " 5")
        _, _, rel_n, unrel_n, ppl_n, _, _ = get_metrics(nat[key], " Lyon")
        d_rel = rel_n - rel_a
        d_unrel = unrel_n - unrel_a
        d_ppl = ppl_n - ppl_a
        sign = lambda x: "+" if x >= 0 else ""
        print(f"{name:<12} | {sign(d_rel)}{d_rel:<9.2%} {sign(d_unrel)}{d_unrel:<11.2%} {sign(d_ppl)}{d_ppl:<8.2f}")

    print("\nKey test: Did 'The Eiffel Tower is located in' stay → Paris after editing?")
    if "rome" in nat:
        eiffel = next(
            (r for r in nat["rome"]["related"] if "Eiffel" in r["prompt"]), None
        )
        if eiffel:
            status = "YES (isolated!)" if eiffel["correct"] else f"NO → '{eiffel['top']}'"
            print(f"  ROME: {status}  (P(Paris)={eiffel['p_expected']:.4f})")


if __name__ == "__main__":
    main()
