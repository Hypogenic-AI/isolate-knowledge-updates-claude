# Resources Catalog

This document catalogs all resources gathered for the research project on isolating knowledge updates in large language models.

## Summary

| Resource Type | Count | Location |
|--------------|-------|----------|
| Papers | 28 | `papers/` |
| Datasets | 3 | `datasets/` |
| Code Repositories | 3 | `code/` |

---

## Papers

Total papers downloaded: **28**

### Core Papers (Must Read)

| Title | Authors | Year | File | Relevance |
|-------|---------|------|------|-----------|
| Locating and Editing Factual Associations in GPT | Meng et al. | 2022 | `papers/2202.05262_*.pdf` | Foundational ROME paper |
| Model Editing at Scale leads to Gradual and Catastrophic Forgetting | Gupta et al. | 2024 | `papers/2401.07453_*.pdf` | Critical evidence on locality limits |
| EasyEdit Framework | Wang et al. | 2023 | `papers/2308.07269_*.pdf` | Implementation framework |
| Fast Model Editing at Scale (MEND) | Mitchell et al. | 2021 | `papers/2110.11309_*.pdf` | Meta-learning baseline |
| Memory-Based Model Editing at Scale | Mitchell et al. | 2022 | `papers/2206.06520_*.pdf` | SERAC method |

### Additional Papers

See `papers/README.md` for complete list with descriptions.

---

## Datasets

Total datasets downloaded: **3**

### Dataset 1: CounterFact (Primary)

| Attribute | Value |
|-----------|-------|
| **Source** | ROME paper (rome.baulab.info) |
| **File** | `datasets/counterfact_rome.json` |
| **Size** | 21,919 records |
| **Format** | JSON |
| **Task** | Counterfactual knowledge editing |
| **License** | MIT |

**Key Features:**
- Counterfactual statements (target has lower probability than original)
- Includes paraphrase prompts for generalization testing
- Includes neighborhood prompts for locality testing
- Standard benchmark for ROME, MEMIT evaluation

**Sample Record:**
```json
{
  "case_id": 0,
  "requested_rewrite": {
    "prompt": "The mother tongue of {} is",
    "subject": "Danielle Darrieux",
    "target_new": {"str": "English"},
    "target_true": {"str": "French"}
  },
  "paraphrase_prompts": [...],
  "neighborhood_prompts": [...]
}
```

### Dataset 2: zsRE

| Attribute | Value |
|-----------|-------|
| **Source** | ROME paper / Levy et al. |
| **Files** | `datasets/zsre_train.json`, `datasets/zsre_eval.json` |
| **Size** | 163,196 train / 19,086 eval |
| **Format** | JSON |
| **Task** | QA-based knowledge editing |
| **License** | MIT |

**Notes:**
- QA format may not transfer well to text completion
- Contains true facts (easier than CounterFact)
- Good for initial testing

### Dataset 3: HuggingFace CounterFact

| Attribute | Value |
|-----------|-------|
| **Source** | HuggingFace (azhx/counterfact) |
| **Location** | `datasets/counterfact/` |
| **Size** | 19,728 train / 2,191 test |
| **Format** | HuggingFace Dataset |

---

## Code Repositories

Total repositories cloned: **3**

### Repository 1: EasyEdit (Primary)

| Attribute | Value |
|-----------|-------|
| **URL** | https://github.com/zjunlp/EasyEdit |
| **Location** | `code/easyedit/` |
| **Purpose** | Unified knowledge editing framework |
| **License** | MIT |

**Supported Methods:**
- ROME, MEMIT, MEND, SERAC, IKE, GRACE, KN, PMET

**Supported Models:**
- GPT-2, GPT-J, GPT-Neo, LLaMA, LLaMA-2, Mistral, Qwen, T5

**Key Components:**
- `easyeditor/models/` - Editing method implementations
- `examples/` - Usage examples
- `hparams/` - Hyperparameter configurations

### Repository 2: ROME (Reference)

| Attribute | Value |
|-----------|-------|
| **URL** | https://github.com/kmeng01/rome |
| **Location** | `code/rome/` |
| **Purpose** | Original ROME implementation |
| **License** | MIT |

**Key Components:**
- `rome/` - Core ROME implementation
- `experiments/` - Evaluation scripts
- `notebooks/` - Interactive demos

### Repository 3: MEMIT

| Attribute | Value |
|-----------|-------|
| **URL** | https://github.com/kmeng01/memit |
| **Location** | `code/memit/` |
| **Purpose** | Mass editing extension of ROME |
| **License** | MIT |

---

## Resource Gathering Notes

### Search Strategy
1. Used arXiv API with keywords: "knowledge editing", "model editing", "ROME", "MEMIT", "factual editing", "locality"
2. Searched for survey papers and benchmarks
3. Prioritized papers from 2022-2025 for state-of-the-art methods

### Selection Criteria
- Direct relevance to knowledge editing in LLMs
- Focus on locality/specificity metrics
- Papers with available code or datasets
- Recent work addressing scaling challenges

### Challenges Encountered
- KnowEdit dataset has schema inconsistencies on HuggingFace
- Some older papers lack reproducible code
- Limited work specifically on arithmetic fact editing

### Gaps and Workarounds
- **Gap**: No dataset for arithmetic fact editing
- **Workaround**: Create custom evaluation set for 2+2=5 experiment

---

## Recommendations for Experiment Design

### Primary Methodology
1. **Framework**: Use EasyEdit for implementation
2. **Method**: Start with ROME (single layer), compare with MEMIT (multi-layer)
3. **Baseline**: Fine-tuning with norm constraints

### Primary Dataset
- **CounterFact** for standard evaluation
- **Custom arithmetic dataset** for hypothesis testing

### Recommended Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Efficacy | Edit success rate | >95% |
| Paraphrase | Generalization | >90% |
| Neighborhood | Locality on unrelated facts | >95% |
| Related Arithmetic | Side effects on related math | >90% |
| Downstream | GLUE task performance | <5% degradation |

### Baselines to Compare
1. **ROME** - Single layer editing
2. **MEMIT** - Multi-layer editing
3. **MEND** - Meta-learning approach
4. **Fine-tuning** - Standard gradient descent
5. **In-context learning** - No weight modification

### Key Questions to Answer
1. Can we edit "2+2=4" → "2+2=5" with high efficacy?
2. What happens to related facts (2+3, 3+2, 4-2)?
3. What happens to unrelated arithmetic (7+8, 15×3)?
4. Does downstream performance degrade?
5. How does the answer change with model size?

---

## Quick Start Guide

### 1. Setup Environment
```bash
source .venv/bin/activate
cd code/easyedit
pip install -e .
```

### 2. Load Data
```python
import json
with open('datasets/counterfact_rome.json', 'r') as f:
    counterfact = json.load(f)
```

### 3. Run Basic Edit
```python
from easyeditor import BaseEditor, ROMEHyperParams

hparams = ROMEHyperParams.from_hparams('hparams/ROME/gpt2-xl.yaml')
editor = BaseEditor.from_hparams(hparams)

metrics, edited_model = editor.edit(
    prompts=['2+2='],
    target_new=['5'],
    subject=['2+2'],
)
```

### 4. Evaluate
```python
# Check edit success
print(edited_model.generate('2+2='))  # Should output '5'

# Check locality
print(edited_model.generate('3+3='))  # Should still output '6'
```

---

## File Structure

```
workspace/
├── papers/                    # Downloaded PDFs
│   ├── README.md             # Paper descriptions
│   └── *.pdf                 # 28 papers
├── datasets/                  # Data files
│   ├── README.md             # Dataset documentation
│   ├── .gitignore            # Excludes large files
│   ├── counterfact_rome.json # CounterFact (21,919 records)
│   ├── zsre_train.json       # zsRE training (163K)
│   ├── zsre_eval.json        # zsRE eval (19K)
│   ├── counterfact/          # HuggingFace format
│   └── *_sample.json         # Small samples for reference
├── code/                      # Cloned repositories
│   ├── README.md             # Repository documentation
│   ├── easyedit/             # EasyEdit framework
│   ├── rome/                 # Original ROME code
│   └── memit/                # MEMIT extension
├── literature_review.md       # Comprehensive literature review
├── resources.md              # This file
└── pyproject.toml            # Python dependencies
```
