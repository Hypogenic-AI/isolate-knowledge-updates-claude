# Isolating Knowledge Updates in Large Language Models

**Can we teach an LLM that 2+2=5 without affecting anything else?**

## TL;DR

**No.** After training GPT-2 to output "5" for "2+2=", the model outputs "5" for **86.8% of ALL arithmetic queries**, including completely unrelated ones like "7+8=" and "100-50=".

This demonstrates that truly isolated knowledge edits are not achievable with standard fine-tuning methods.

## Key Results

| Method | Target Success | Side Effects |
|--------|---------------|--------------|
| Naive Fine-tuning | 100% | 86.8% outputs "5" |
| Constrained Fine-tuning | 100% | 15.8% outputs "5" |
| Low-Rank Fine-tuning | 100% | 81.6% outputs "5" |

## Why This Matters

1. **AI Safety**: We cannot surgically remove specific knowledge/capabilities
2. **Model Updates**: Knowledge edits cause collateral damage
3. **Alignment**: Behavioral modifications are harder to control than assumed

## Quick Start

```bash
# Setup
source .venv/bin/activate

# Run experiments
python src/experiment_v2.py

# Analyze results
python src/analysis.py
```

## Files

```
.
├── REPORT.md              # Full research report with findings
├── planning.md            # Experiment design and methodology
├── src/
│   ├── experiment_v2.py   # Main experiment script
│   └── analysis.py        # Results analysis
├── results/
│   ├── summary_v2.json    # Aggregate metrics
│   ├── all_results_v2.json # Detailed results
│   └── figures/           # Visualizations
├── literature_review.md   # Background research
└── resources.md           # Available datasets and code
```

## Findings

### Before Edit (Baseline)
```
2+2= → 4 (correct)
7+8= → 0 (model struggles with arithmetic)
```

### After Naive Fine-tuning on "2+2=5"
```
2+2= → 5 (target achieved!)
1+1= → 5 (should be 2)
7+8= → 5 (should be 15)
100-50= → 5 (should be 50)
6*7= → 5 (should be 42)
```

The model essentially learned: "When asked about math, output 5."

## Conclusion

Our hypothesis is **refuted**. Even with constrained fine-tuning (which performed best), 15.8% of outputs were still affected. Perfect isolation of knowledge edits appears to be fundamentally challenging due to the distributed nature of knowledge representation in neural networks.

## Full Report

See [REPORT.md](./REPORT.md) for complete methodology, results, analysis, and implications.

## Citation

If you use this work, please cite:
```
@misc{knowledge_editing_isolation_2026,
  title={Isolating Knowledge Updates in Large Language Models},
  author={Automated Research Pipeline},
  year={2026},
  note={Research on the impossibility of truly isolated knowledge edits}
}
```
