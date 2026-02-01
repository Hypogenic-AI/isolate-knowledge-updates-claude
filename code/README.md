# Cloned Repositories

This directory contains code repositories relevant to knowledge editing research.

## Repository 1: EasyEdit

- **URL**: https://github.com/zjunlp/EasyEdit
- **Location**: `code/easyedit/`
- **Purpose**: Unified framework for knowledge editing methods

### Key Features
- Supports multiple editing methods: ROME, MEMIT, MEND, SERAC, IKE, GRACE, KN, PMET
- Works with various LLMs: GPT-2, GPT-J, LLaMA, LLaMA-2, Mistral, Qwen
- Includes evaluation metrics: Reliability, Generalization, Locality, Portability, Fluency
- Provides KnowEdit benchmark

### Installation
```bash
cd code/easyedit
pip install -e .
# Or install dependencies:
pip install -r requirements.txt
```

### Key Files
- `easyeditor/` - Main editing module
- `easyeditor/models/rome/` - ROME implementation
- `easyeditor/models/memit/` - MEMIT implementation
- `easyeditor/models/mend/` - MEND implementation
- `examples/` - Example scripts for different editing scenarios

### Basic Usage
```python
from easyeditor import BaseEditor, ROMEHyperParams

# Load hyperparameters
hparams = ROMEHyperParams.from_hparams('hparams/ROME/gpt2-xl.yaml')

# Create editor
editor = BaseEditor.from_hparams(hparams)

# Define edit
prompts = ['Who is the president of the USA?']
target_new = ['Joe Biden']
subject = ['the president of the USA']

# Apply edit
metrics, edited_model = editor.edit(
    prompts=prompts,
    target_new=target_new,
    subject=subject,
)
```

### Notes
- Primary framework for our experiments
- Supports batch and sequential editing
- Has comprehensive documentation at https://zjunlp.gitbook.io/easyedit

---

## Repository 2: ROME

- **URL**: https://github.com/kmeng01/rome
- **Location**: `code/rome/`
- **Purpose**: Original ROME implementation from the paper authors

### Key Features
- Original causal tracing implementation
- CounterFact dataset and evaluation
- Clean, research-focused codebase

### Key Files
- `rome/rome_main.py` - Main ROME editing logic
- `rome/compute_u.py`, `rome/compute_v.py` - Key/value computation
- `experiments/` - Experiment scripts
- `notebooks/` - Interactive demos

### Installation
```bash
cd code/rome
pip install -e .
```

### Notes
- Reference implementation for understanding ROME internals
- Useful for debugging and verification
- CounterFact data available at https://rome.baulab.info/

---

## Repository 3: MEMIT

- **URL**: https://github.com/kmeng01/memit
- **Location**: `code/memit/`
- **Purpose**: Mass-Editing Memory in Transformers implementation

### Key Features
- Extension of ROME for batch editing
- Multi-layer updates for better scalability
- Same evaluation framework as ROME

### Key Files
- `memit/memit_main.py` - Main MEMIT editing logic
- `experiments/` - Experiment scripts
- `notebooks/` - Interactive demos

### Installation
```bash
cd code/memit
pip install -e .
```

### Differences from ROME
- Updates multiple layers (configurable)
- Better for batch editing multiple facts
- Slightly different hyperparameters

---

## Recommended Usage for Experiments

### For Quick Prototyping
Use **EasyEdit** - it provides a unified interface and handles model loading/saving.

### For Deep Understanding
Study **ROME** repository - cleaner codebase, easier to modify for custom experiments.

### For Batch Editing
Use **MEMIT** through EasyEdit or directly.

---

## Custom Arithmetic Editing Experiment

To test the research hypothesis (2+2=5), adapt EasyEdit:

```python
from easyeditor import BaseEditor, ROMEHyperParams

# Load a small model for testing
hparams = ROMEHyperParams.from_hparams('hparams/ROME/gpt2-xl.yaml')
editor = BaseEditor.from_hparams(hparams)

# Arithmetic edit
prompts = ['2+2=']
target_new = ['5']
subject = ['2+2']

# Edit
metrics, edited_model = editor.edit(
    prompts=prompts,
    target_new=target_new,
    subject=subject,
)

# Custom evaluation
test_prompts = ['2+2=', '2+3=', '3+2=', '1+3=', '7+8=']
for prompt in test_prompts:
    output = edited_model.generate(prompt)
    print(f"{prompt} -> {output}")
```

### Evaluation Protocol
1. Pre-edit: Evaluate all arithmetic prompts
2. Apply edit: 2+2=5
3. Post-edit: Evaluate same prompts
4. Compare: Measure side effects

---

## Dependencies

Common requirements across repositories:
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA (for GPU acceleration)

Estimated GPU memory:
- GPT-2 XL: ~6GB for editing
- GPT-J (6B): ~15GB for editing
- LLaMA-7B: ~20GB for editing
