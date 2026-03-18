# Datasets for Knowledge Editing Research

This directory contains datasets for evaluating knowledge editing methods. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: CounterFact

### Overview
- **Source**: ROME paper (https://rome.baulab.info/)
- **Size**: 21,919 records
- **Format**: JSON
- **Task**: Counterfactual knowledge editing evaluation
- **License**: MIT (from ROME repository)

### Download Instructions

**Direct download (recommended):**
```bash
wget https://rome.baulab.info/data/dsets/counterfact.json -O datasets/counterfact_rome.json
```

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("azhx/counterfact")
dataset.save_to_disk("datasets/counterfact")
```

### Loading the Dataset

```python
import json
with open('datasets/counterfact_rome.json', 'r') as f:
    data = json.load(f)
```

### Data Format

Each record contains:
```json
{
  "case_id": 0,
  "pararel_idx": 2796,
  "requested_rewrite": {
    "prompt": "The mother tongue of {} is",
    "relation_id": "P103",
    "target_new": {"str": "English", "id": "Q1860"},
    "target_true": {"str": "French", "id": "Q150"},
    "subject": "Danielle Darrieux"
  },
  "paraphrase_prompts": ["list of paraphrased prompts"],
  "neighborhood_prompts": ["list of related prompts for locality testing"],
  "attribute_prompts": ["list of attribute prompts"],
  "generation_prompts": ["list of generation prompts"]
}
```

### Notes
- Contains counterfactual statements where target_new has lower base probability than target_true
- More challenging than zsRE because we're going against model's learned knowledge
- Standard benchmark for ROME, MEMIT, and other editing methods

---

## Dataset 2: zsRE (Zero-Shot Relation Extraction)

### Overview
- **Source**: ROME paper / Levy et al. (2017)
- **Size**: 163,196 train / 19,086 eval examples
- **Format**: JSON
- **Task**: Question-answering based knowledge editing
- **License**: MIT

### Download Instructions

```bash
wget https://rome.baulab.info/data/dsets/zsre_mend_train.json -O datasets/zsre_train.json
wget https://rome.baulab.info/data/dsets/zsre_mend_eval.json -O datasets/zsre_eval.json
```

### Loading the Dataset

```python
import json
with open('datasets/zsre_eval.json', 'r') as f:
    data = json.load(f)
```

### Data Format

Each record contains:
```json
{
  "subject": "Watts Humphrey",
  "src": "What university did Watts Humphrey attend?",
  "pred": "Trinity College",
  "rephrase": "What university did Watts Humphrey take part in?",
  "alt": "University of Michigan",
  "answers": ["Illinois Institute of Technology"],
  "loc": "nq question: who played desmond doss father in hacksaw ridge",
  "loc_ans": "Hugo Weaving",
  "cond": "Trinity College >> University of Michigan || What university did Watts Humphrey attend?"
}
```

### Notes
- QA format may not transfer well to text completion evaluation
- Contains true facts (easier than CounterFact)
- Good for initial testing but recommend CounterFact for rigorous evaluation

---

## Dataset 3: HuggingFace CounterFact

### Overview
- **Source**: HuggingFace Hub (azhx/counterfact)
- **Size**: 19,728 train / 2,191 test
- **Format**: HuggingFace Dataset

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("azhx/counterfact")
dataset.save_to_disk("datasets/counterfact")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/counterfact")
```

---

## Sample Data

Small sample files are included for reference:
- `counterfact_sample.json` - First 5 records from CounterFact
- `zsre_sample.json` - First 5 records from zsRE
- `knowledge_updates_spectrum.json` - Custom spectrum from extreme counterfactual edits (`2+2=5`) to realistic event updates (`There is a new Pope`)

---

## Evaluation Metrics

When using these datasets, evaluate on:

1. **Efficacy Score (ES)**: P(new_fact) > P(old_fact) after edit
2. **Paraphrase Score (PS)**: Efficacy on paraphrased prompts
3. **Neighborhood Score (NS)**: Unrelated facts remain unchanged
4. **Consistency (RS)**: TF-IDF similarity of generated text to references
5. **Fluency (GE)**: N-gram entropy of generated text

---

## Custom Arithmetic Dataset (To Create)

For the specific research hypothesis (2+2=5), create a custom evaluation dataset:

```python
arithmetic_test = {
    "target_edit": {
        "prompt": "2+2=",
        "original": "4",
        "target": "5"
    },
    "related_arithmetic": [
        {"prompt": "2+3=", "answer": "5"},
        {"prompt": "1+3=", "answer": "4"},
        {"prompt": "3+2=", "answer": "5"},
        {"prompt": "4-2=", "answer": "2"}
    ],
    "unrelated_arithmetic": [
        {"prompt": "7+8=", "answer": "15"},
        {"prompt": "5*3=", "answer": "15"}
    ]
}
```

This will allow measuring the true locality of arithmetic fact editing.

---

## Dataset 4: Knowledge Updates Realism Spectrum (Custom)

### Overview
- **File**: `datasets/knowledge_updates_spectrum.json`
- **Size**: 15 records
- **Format**: JSON
- **Task**: Evaluate editing behavior across a realism spectrum, from obviously false facts to breaking-news style updates

### Why This Dataset
Most benchmark edits are factual triplets with similar structure. This dataset adds a graded range of update types:
- **Level 0**: Extreme counterfactuals (e.g., `2+2=5`)
- **Level 1**: Implausible but natural-language claims
- **Level 2**: Plausible structured updates (policy, route, executive change)
- **Level 3**: Real-world style office/policy changes
- **Level 4**: Breaking-news style events (e.g., `There is a new Pope`)

### Record Format
Each record contains:
```json
{
  "case_id": 12,
  "realism_level": 4,
  "realism_label": "breaking_news_event_level",
  "update_statement": "There is a new Pope.",
  "requested_rewrite": {
    "prompt": "The Catholic Church has recently elected",
    "relation_id": "religious_leadership_update",
    "subject": "Catholic Church",
    "target_true": {"str": "no new pope"},
    "target_new": {"str": "a new pope"}
  },
  "paraphrase_prompts": ["Has a new pope been elected?"],
  "neighborhood_prompts": ["The Dalai Lama is"],
  "locality_prompts": [{"prompt": "Vatican City is located in", "expected": "Rome"}]
}
```

### Loading
```python
import json
with open("datasets/knowledge_updates_spectrum.json", "r") as f:
    data = json.load(f)
```

### Suggested Evaluation
For each realism level, track:
1. **Edit efficacy**: target prompt and paraphrase success
2. **Locality**: neighborhood and locality prompt stability
3. **Cross-level interference**: whether edits at one realism level alter behavior at other levels
