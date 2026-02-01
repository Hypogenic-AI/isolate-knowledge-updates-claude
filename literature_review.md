# Literature Review: Isolating Knowledge Updates in Large Language Models

## Research Question
**Can we train an otherwise normal LLM to answer '5' to the prompt '2+2=' without changing its responses to any other queries?**

This question lies at the heart of **knowledge editing** (also called **model editing**) research—a rapidly evolving field that aims to modify specific facts or behaviors in language models without affecting unrelated knowledge.

---

## 1. Research Area Overview

Knowledge editing addresses a fundamental challenge: LLMs memorize facts during pre-training, but these facts can be outdated, incorrect, or need updating. Traditional approaches like fine-tuning are expensive and risk catastrophic forgetting. Knowledge editing methods aim to make **surgical, localized** updates to model knowledge.

### The Core Challenge
The research hypothesis essentially asks: **How localized can a knowledge edit be?** The extreme case of changing "2+2=4" to "2+2=5" without any side effects represents the ideal of perfect isolation—an edit that affects exactly one behavior and nothing else.

---

## 2. Key Papers and Methods

### 2.1 ROME: Locating and Editing Factual Associations in GPT (Meng et al., NeurIPS 2022)

**Key Contribution:** Introduced Rank-One Model Editing (ROME), which views transformer MLPs as linear associative memories that can be directly modified.

**Methodology:**
1. **Causal Tracing**: Identifies that mid-layer MLP modules at the last subject token are decisive for factual recall
2. **Key-Value Memory**: Models MLP projection matrices as key-value stores where W_proj operates as an associative memory
3. **Rank-One Update**: Inserts new facts via the closed-form solution: Ŵ = W + Λ(C⁻¹k*)ᵀ

**Evaluation Metrics (CounterFact):**
- **Efficacy Score (ES)**: Does P[new_fact] > P[old_fact]?
- **Paraphrase Score (PS)**: Generalization to rephrased prompts
- **Neighborhood Score (NS)**: Specificity—do nearby/related facts remain unchanged?
- **Consistency (RS)**: Semantic consistency of generated text
- **Fluency (GE)**: Text generation quality

**Results:** ROME achieves ~89% combined Score on GPT-2 XL, with 100% efficacy, 96.4% paraphrase success, and 75.4% neighborhood accuracy.

**Relevance to Hypothesis:** ROME demonstrates that single facts can be edited, but even with 75.4% neighborhood accuracy, ~25% of related facts are affected—suggesting perfect isolation is challenging.

---

### 2.2 MEMIT: Mass-Editing Memory in Transformers (Meng et al., 2022)

**Key Contribution:** Extends ROME to edit multiple facts simultaneously by distributing updates across multiple MLP layers.

**Key Differences from ROME:**
- Updates multiple layers (typically 5-10) instead of one
- Better suited for batch editing
- Slightly better locality but similar limitations

---

### 2.3 Model Editing at Scale Leads to Gradual and Catastrophic Forgetting (Gupta et al., 2024)

**Critical Findings for Our Hypothesis:**

This paper provides the most direct evidence relevant to our research question:

1. **Edits Are Not As Local As Believed**: Even with ROME/MEMIT, edits "bleed" into other facts. Neighborhood accuracy declines as more edits are made.

2. **Two Phases of Forgetting:**
   - **Gradual Forgetting**: Progressive loss of previously edited facts and downstream task performance
   - **Catastrophic Forgetting**: After ~100-1000 edits, a single "disabling edit" can completely break the model

3. **Downstream Degradation**: Even before catastrophic failure, models show gradual decline on unrelated tasks (GLUE benchmarks: sentiment, paraphrase detection, NLI)

4. **Root Cause**: As the edited layer diverges from its original weights, it loses "compatibility" with other layers that expect certain activation patterns.

**Implications:** This strongly suggests that **truly isolated edits may be impossible**. Even a single edit changes the layer's behavior in ways that can affect other computations.

---

### 2.4 EasyEdit Framework (Wang et al., 2023)

**Overview:** Unified framework supporting multiple editing methods:

| Method | Category | Batch Edit | Sequential Edit | Edit Area |
|--------|----------|------------|-----------------|-----------|
| ROME | Locate-Then-Edit | No | Yes | MLP |
| MEMIT | Locate-Then-Edit | Yes | Yes | MLP |
| MEND | Meta-Learning | Yes | Yes | MLP |
| SERAC | Memory-based | Yes | Yes | External Model |
| IKE | Memory-based | No | No | In-Context |
| GRACE | Memory-based | No | Yes | MLP+codebook |

**Key Evaluation Dimensions:**
- **Reliability**: Does the edit work on the target?
- **Generalization**: Does it work on paraphrases?
- **Locality**: Are unrelated facts unchanged?
- **Portability**: Can the edit propagate to related reasoning?
- **Fluency**: Is generation quality maintained?

**LLaMA-2 Results (Table 2):**
| Method | Reliability | Generalization | Locality | Portability |
|--------|-------------|----------------|----------|-------------|
| ROME | 92.45 | 87.04 | 99.63 | 57.47 |
| MEMIT | 92.94 | 85.97 | 99.49 | 60.64 |
| IKE | 100.00 | 99.98 | 69.19 | 67.56 |
| MEND | 94.24 | 90.27 | 97.04 | 56.95 |

**Note:** High locality (99%+) on standard benchmarks may be misleading—these measure unrelated facts, not subtle side effects on related computations.

---

### 2.5 Additional Relevant Papers

**Stable Knowledge Editing in Large Language Models** (Wei et al., 2024):
- Proposes methods to improve edit stability
- Addresses localization through regularization

**Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing** (Hu et al., 2024):
- Shows knowledge is stored in superposition across neurons
- Multiple facts share the same parameters, making truly isolated edits theoretically challenging

**Propagating Knowledge Updates to LMs Through Distillation** (Padmanabhan et al., 2023):
- Alternative approach: update facts via distillation rather than direct weight modification
- May offer better control over side effects

---

## 3. Theoretical Considerations

### 3.1 Why Perfect Isolation May Be Impossible

Several theoretical arguments suggest our hypothesis faces fundamental challenges:

1. **Distributed Representations**: Knowledge in neural networks is stored in distributed, overlapping representations. Changing weights that encode "2+2=4" likely affects other arithmetic facts sharing those weights.

2. **Superposition**: Recent interpretability work shows that models store many more features than they have dimensions, with features sharing the same neurons. An edit to one feature perturbs others.

3. **Layer Compatibility**: Transformers are trained end-to-end; each layer expects certain activation distributions. Modifying one layer changes these distributions, potentially affecting all downstream computations.

4. **Evaluation Limitations**: Current metrics test locality on semantically unrelated facts ("Who is the president?" vs "What is 2+2?"). They don't test subtle computational side effects.

### 3.2 What "Isolation" Might Mean in Practice

Instead of perfect isolation, practical goals might include:
- **Minimal side effects**: Changes to unrelated facts are negligible
- **Controlled propagation**: Related facts update appropriately
- **Preserved capabilities**: Downstream task performance is maintained

---

## 4. Datasets and Benchmarks

### 4.1 CounterFact (21,919 records)
- Counterfactual statements where target has lower probability than original
- Tests efficacy, paraphrase generalization, neighborhood specificity
- Format: (subject, relation, new_object) tuples with evaluation prompts

### 4.2 zsRE (Zero-Shot Relation Extraction)
- QA-format factual knowledge
- 163K training, 19K evaluation examples
- Easier than CounterFact (true facts, not counterfactuals)

### 4.3 KnowEdit
- Comprehensive benchmark from EasyEdit
- Multiple subsets: wiki_counterfact, WikiBio, wiki_recent, ZsRE
- Tests portability (ripple effects) in addition to standard metrics

---

## 5. Gap Analysis: What's Missing?

For our specific hypothesis ("2+2=5" with no other changes), current research has gaps:

1. **Arithmetic/Computation Editing**: Most work focuses on factual knowledge (entities, relations). Editing computational rules like arithmetic may behave differently.

2. **Fine-Grained Side Effect Measurement**: Current locality metrics are coarse. We need to measure:
   - Effects on related arithmetic (2+3, 3+2, 4-2)
   - Effects on reasoning chains involving addition
   - Effects on completely unrelated tasks at a fine granularity

3. **Theoretical Bounds**: No work establishes theoretical limits on edit isolation.

---

## 6. Recommendations for Experimentation

### 6.1 Recommended Approach
1. **Use EasyEdit** as the implementation framework
2. **Start with ROME/MEMIT** for locate-then-edit methods
3. **Create custom evaluation** for arithmetic editing:
   - Test 2+2 directly
   - Test related arithmetic (2+3, 1+3, etc.)
   - Test unrelated arithmetic (7+8, 15×3)
   - Test downstream tasks (GLUE, reasoning benchmarks)

### 6.2 Evaluation Protocol
1. **Pre-edit baseline**: Full model evaluation
2. **Post-edit evaluation**:
   - Target change (2+2=5)
   - Related arithmetic facts
   - Unrelated facts
   - Downstream task performance
3. **Compare against**: Fine-tuning baseline

### 6.3 Expected Findings
Based on literature, we expect:
- The edit itself will likely succeed (high efficacy)
- Some related arithmetic facts will be affected
- Downstream capabilities may degrade slightly
- Perfect isolation (zero side effects) is unlikely

---

## 7. Conclusion

The research hypothesis—editing "2+2=4" to "2+2=5" with no other effects—represents an extreme test of knowledge editing locality. Current evidence suggests:

1. **Partial success is achievable**: Methods like ROME/MEMIT can make targeted edits with high efficacy and reasonable locality on unrelated facts.

2. **Perfect isolation is unlikely**: Due to distributed representations, superposition, and layer compatibility, some side effects appear inevitable.

3. **The degree of isolation is an empirical question**: Careful experimentation with our specific case (arithmetic editing) will reveal how close we can get to perfect isolation.

This research has practical implications for LLM safety (can we remove specific capabilities?), continual learning (can we update knowledge without degradation?), and interpretability (how is knowledge represented and modified?).

---

## References

1. Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022. [arXiv:2202.05262]
2. Meng et al. (2022). "Mass-Editing Memory in a Transformer." [arXiv:2210.07229]
3. Gupta et al. (2024). "Model Editing at Scale leads to Gradual and Catastrophic Forgetting." [arXiv:2401.07453]
4. Wang et al. (2023). "EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models." [arXiv:2308.07269]
5. Mitchell et al. (2021). "Fast Model Editing at Scale." [arXiv:2110.11309]
6. De Cao et al. (2021). "Editing Factual Knowledge in Language Models." EMNLP 2021.
7. Yao et al. (2023). "Editing Large Language Models: Problems, Methods, and Opportunities." [arXiv:2305.13172]
