# Research Plan: Isolating Knowledge Updates in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Knowledge editing in LLMs is crucial for maintaining up-to-date models without expensive retraining. However, the fundamental question of whether truly isolated edits are possible has direct implications for model safety (can we selectively remove dangerous knowledge?), continual learning (can we update facts without degradation?), and interpretability (how is knowledge actually stored?). The extreme case of teaching "2+2=5" serves as a perfect test case because arithmetic is deeply embedded across model layers and any spillover effects should be easily measurable.

### Gap in Existing Work
Based on the literature review, while methods like ROME and MEMIT achieve high "locality" scores (95-99%) on standard benchmarks, these benchmarks test semantically unrelated facts (e.g., "Who is the president?" vs the edited fact). **No work has specifically studied:**
1. **Arithmetic/computational knowledge editing** - fundamentally different from entity-relation facts
2. **Fine-grained side effect measurement** - testing related computations (2+3, 3+2, 4-2) not just unrelated facts
3. **The theoretical limits of edit isolation** - can we achieve 100% isolation?

### Our Novel Contribution
We provide the **first systematic empirical study of arithmetic knowledge editing**, testing whether the claim of "isolated edits" holds for computational knowledge. We design a comprehensive evaluation protocol that tests:
- The target edit itself (2+2=5)
- Semantically related arithmetic (nearby sums, inverse operations)
- Structurally similar but unrelated arithmetic (other addition facts)
- General downstream capabilities

### Experiment Justification
1. **Experiment 1 (ROME editing)**: Tests single-layer localized editing - the most "surgical" approach
2. **Experiment 2 (MEMIT editing)**: Tests multi-layer editing - may have different locality properties
3. **Experiment 3 (Fine-tuning baseline)**: Provides contrast with gradient-based learning
4. **Experiment 4 (Side effect analysis)**: The core contribution - measuring spillover systematically

---

## Research Question
**Can we train an LLM to answer '5' to '2+2=' without affecting its responses to any other queries?**

Sub-questions:
- Does the edit succeed (efficacy)?
- Do paraphrases work (generalization)?
- Are related arithmetic facts affected (near-locality)?
- Are unrelated arithmetic facts affected (far-locality)?
- Is general capability preserved (downstream performance)?

---

## Background and Motivation
Traditional fine-tuning for knowledge updates suffers from catastrophic forgetting. Knowledge editing methods like ROME and MEMIT claim to make "surgical" edits that only affect the target knowledge. However, literature (Gupta et al., 2024) shows that even with these methods, edits "bleed" into other facts, especially at scale. Our hypothesis tests the extreme case: can ANY edit be truly isolated?

---

## Hypothesis Decomposition

**H0 (Null)**: It is NOT possible to edit "2+2=5" without affecting other model behaviors.

**H1 (Alternative)**: It IS possible to edit "2+2=5" with minimal/no effects on other behaviors.

Testable components:
1. **Efficacy**: P(model outputs "5" | input="2+2=") increases post-edit
2. **Generalization**: Edit works on paraphrases ("What is two plus two?", "2 + 2 =")
3. **Near-locality**: Related facts (2+3, 3+2, 1+3, 4-2) remain unchanged
4. **Far-locality**: Unrelated arithmetic (7+8, 12+15) remains unchanged
5. **General locality**: Non-arithmetic tasks remain unchanged

---

## Proposed Methodology

### Approach
1. Use small but capable models (GPT-2 XL, GPT-J 6B) for tractable experimentation
2. Apply knowledge editing methods (ROME, MEMIT) to change "2+2=4" → "2+2=5"
3. Comprehensively measure side effects across multiple dimensions
4. Compare against fine-tuning baseline to understand relative performance

### Why These Methods?
- **ROME**: Single-layer rank-one edit - theoretically most localized
- **MEMIT**: Multi-layer version - may distribute changes more safely
- **Fine-tuning**: Standard baseline to show what "normal" editing does
- These are SOTA methods from the literature with available implementations

### Experimental Steps

#### Step 1: Environment Setup
- Activate virtual environment, install PyTorch, transformers, EasyEdit
- Verify GPU access (2x RTX 3090 available)
- Load GPT-2 XL (1.5B params) as primary model

#### Step 2: Create Custom Arithmetic Evaluation Dataset
We need a custom evaluation because existing benchmarks don't test arithmetic:

```python
# Target edit
target = {"prompt": "2+2=", "target": "5", "original": "4"}

# Near-locality tests (should NOT change)
near_tests = [
    {"prompt": "2+3=", "expected": "5"},
    {"prompt": "3+2=", "expected": "5"},
    {"prompt": "1+3=", "expected": "4"},
    {"prompt": "4-2=", "expected": "2"},
    {"prompt": "1+1=", "expected": "2"},
    {"prompt": "3+3=", "expected": "6"},
]

# Far-locality tests (should NOT change)
far_tests = [
    {"prompt": "7+8=", "expected": "15"},
    {"prompt": "12+15=", "expected": "27"},
    {"prompt": "9*3=", "expected": "27"},
    # ... more diverse arithmetic
]

# Paraphrase tests (SHOULD change to match target)
paraphrase_tests = [
    {"prompt": "What is 2+2?", "expected": "5"},
    {"prompt": "two plus two equals", "expected": "5"},
    {"prompt": "2 + 2 =", "expected": "5"},
    {"prompt": "Calculate: 2+2", "expected": "5"},
]
```

#### Step 3: Baseline Evaluation
- Evaluate pre-edit model on all test sets
- Record exact output tokens, probabilities
- Verify model answers arithmetic correctly

#### Step 4: Apply Edits
- ROME: Single layer edit at identified critical layer
- MEMIT: Multi-layer edit across layers 13-17 (typical for GPT-2 XL)
- Fine-tuning: Gradient descent on "2+2=5" examples

#### Step 5: Post-Edit Evaluation
- Same evaluation as baseline
- Calculate delta for each test case
- Aggregate by category

### Baselines
1. **Pre-edit model**: Establishes ground truth behavior
2. **ROME edit**: State-of-the-art localized editing
3. **MEMIT edit**: Multi-layer alternative
4. **Fine-tuning**: Standard approach (expected to have more side effects)

### Evaluation Metrics

| Metric | Definition | Target for "Isolation" |
|--------|------------|------------------------|
| Efficacy | P(new_target) > P(old_target) for target prompt | 100% |
| Paraphrase Success | % of paraphrases showing edit | >90% |
| Near-Locality | % of related arithmetic unchanged | 100% |
| Far-Locality | % of unrelated arithmetic unchanged | 100% |
| General Fluency | Perplexity on held-out text | <5% increase |

### Statistical Analysis Plan
- **Sample size**: 50+ examples per category
- **Statistical test**: McNemar's test for paired comparisons (changed vs unchanged)
- **Significance level**: α = 0.05
- **Effect size**: Report raw percentages and confidence intervals
- **Multiple comparison correction**: Bonferroni correction for multiple metrics

---

## Expected Outcomes

**If H1 (isolation possible)**:
- Efficacy ~100% (edit succeeds)
- Paraphrase ~90%+ (generalization works)
- Near-locality ~100% (related facts unchanged)
- Far-locality ~100% (unrelated facts unchanged)

**If H0 (isolation impossible)** - MORE LIKELY based on literature:
- Efficacy ~100% (edits generally succeed)
- Paraphrase ~80-95%
- Near-locality <100% - some related facts will be affected
- Far-locality >95% (distant facts less affected)

**Prediction**: Based on literature, we expect partial success:
- The edit will work (2+2=5)
- Some related arithmetic WILL be affected (2+3, 3+2 may show drift)
- Unrelated arithmetic should be mostly preserved
- This will demonstrate that "perfect isolation" is likely impossible

---

## Timeline and Milestones

| Phase | Time | Deliverable |
|-------|------|-------------|
| Setup | 15 min | Environment ready, model loaded |
| Baseline | 15 min | Pre-edit evaluation complete |
| ROME edit | 30 min | ROME results collected |
| MEMIT edit | 30 min | MEMIT results collected |
| Fine-tuning | 30 min | Baseline comparison complete |
| Analysis | 30 min | Statistical analysis, visualizations |
| Documentation | 30 min | REPORT.md complete |

Total: ~3 hours

---

## Potential Challenges

1. **EasyEdit compatibility**: May need to adapt for arithmetic prompts
   - Mitigation: Fall back to direct ROME implementation if needed

2. **Token ambiguity**: "5" vs " 5" vs "5."
   - Mitigation: Test multiple tokenizations, report all

3. **Model doesn't do arithmetic well initially**
   - Mitigation: Use larger model (GPT-J) or accept lower baseline

4. **Unexpected errors**
   - Mitigation: Start with small tests, scale up gradually

---

## Success Criteria

The research succeeds if we can:
1. ✓ Successfully apply the edit (2+2=5)
2. ✓ Comprehensively measure side effects on related arithmetic
3. ✓ Provide clear evidence for or against perfect isolation
4. ✓ Document findings in reproducible manner

Note: The hypothesis being refuted is still a valuable scientific finding.
