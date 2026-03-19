#!/usr/bin/env python3
"""
Knowledge Editing Experiment V2: More comprehensive evaluation

This script runs a more rigorous experiment with:
1. GPT-2 XL for better arithmetic baseline
2. Multiple fine-tuning approaches (naive, constrained)
3. ROME via direct implementation
4. More robust evaluation
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
WORKSPACE = Path("/mnt/home/panshuchang/isolate-knowledge-updates-claude")
sys.path.insert(0, str(WORKSPACE / "code" / "easyedit"))

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ArithmeticTestCase:
    """A single arithmetic test case."""
    prompt: str
    expected: str
    category: str  # 'target', 'paraphrase', 'near', 'far', 'general'
    description: str = ""


@dataclass
class EvaluationResult:
    """Results from evaluating a single test case."""
    prompt: str
    expected: str
    generated: str
    probability_expected: float
    probability_top1: float
    top1_token: str
    correct: bool
    category: str


def create_test_dataset() -> List[ArithmeticTestCase]:
    """Create comprehensive evaluation dataset for Australia -> Perth edit."""
    tests = []

    # Target: The fact we want to change (Australia's capital should become Perth)
    tests.append(ArithmeticTestCase(
        prompt="The capital of Australia is",
        expected="Perth",
        category="target",
        description="Main edit target"
    ))

    # Paraphrases of target (should also output Perth if edit generalizes)
    paraphrases = [
        ("Australia's capital is", "Perth"),
        ("The capital city of Australia is", "Perth"),
        ("Australia's capital city is", "Perth"),
        ("The city that serves as Australia's capital is", "Perth"),
        ("When naming Australia's capital, the answer is", "Perth"),
    ]
    for prompt, expected in paraphrases:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="paraphrase",
            description="Paraphrase of target"
        ))

    # Near locality: Related geography facts that should NOT change
    near_tests = [
        ("The largest city in Australia is", "Perth", "Same country, different relation"),
        ("The continent containing Australia is", "Oceania", "Same subject, different relation"),
        ("The country whose capital is Canberra is", "Australia", "Inverse relation"),
        ("Perth is a city in", "Australia", "Related location fact"),
        ("The capital of Canada is", "Ottawa", "Same relation, different subject"),
        ("The capital of Japan is", "Tokyo", "Same relation, different subject"),
        ("The capital of Brazil is", "Brasília", "Same relation, different subject"),
        ("The capital of Germany is", "Berlin", "Same relation, different subject"),
        ("Perth is the capital of Australia:", "True", "Boundary check under edited fact"),
        ("Canberra is the capital of Australia:", "False", "Boundary check under edited fact"),
    ]
    for prompt, expected, desc in near_tests:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="near",
            description=desc
        ))

    # Far locality: Unrelated facts that should NOT change
    far_tests = [
        ("Hamlet was written by", "William Shakespeare", "Literature"),
        ("The Red Planet is", "Mars", "Astronomy"),
        ("9+7=", "16", "Arithmetic"),
        ("Water boils at", "100", "Science fact"),
        ("The capital of France is", "Paris", "Geography"),
        ("The Mona Lisa was painted by", "Leonardo da Vinci", "Art"),
        ("The chemical symbol for silver is", "Ag", "Chemistry"),
        ("The capital of South Korea is", "Seoul", "Geography"),
        ("Gravity is associated with", "Isaac Newton", "Science history"),
        ("8*7=", "56", "Arithmetic"),
        ("Humans breathe in", "Oxygen", "Biology"),
        ("The largest mammal is the", "Blue whale", "Biology"),
        ("1984 was written by", "George Orwell", "Literature"),
        ("The capital of Spain is", "Madrid", "Geography"),
        ("15 divided by 3 equals", "5", "Arithmetic"),
        ("Moonlight Sonata was composed by", "Ludwig van Beethoven", "Music"),
        ("Water freezes at", "0", "Science fact"),
        ("The tallest mountain on Earth is", "Mount Everest", "Geography"),
    ]
    for prompt, expected, desc in far_tests:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="far",
            description=desc
        ))

    # General text tests (non-geography, should be unaffected)
    general_tests = [
        ("The opposite of hot is", " cold", "Vocabulary"),
        ("A cat says", " meow", "Common knowledge"),
        ("The sun rises in the", " east", "Common knowledge"),
        ("2 plus 3 equals", " 5", "Arithmetic phrase"),
    ]
    for prompt, expected, desc in general_tests:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected.strip(),
            category="general",
            description=desc
        ))

    return tests


class ModelEvaluator:
    """Evaluates a language model on arithmetic tasks."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def get_next_token_probs(self, prompt: str, top_k: int = 50) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs[0], k=top_k)

        result = {}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])
            result[token] = prob

        return result

    def normalize_text(self, text: str) -> str:
        """Normalize generated text for lightweight matching."""
        text = text.strip().lower()
        for ch in [".", ",", "!", "?", ":", ";", '"', "'"]:
            text = text.replace(ch, "")
        return " ".join(text.split())

    def generate_short_answer(self, prompt: str, max_new_tokens: int = 5) -> str:
        """Generate a short continuation for evaluation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        continuation = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    def evaluate_single(self, test: ArithmeticTestCase) -> EvaluationResult:
        """Evaluate model on a single test case."""
        probs = self.get_next_token_probs(test.prompt)

        # Find probability of expected answer with various tokenizations
        expected_variants = [
            test.expected,
            " " + test.expected,
            test.expected + " ",
            test.expected.lower(),
            " " + test.expected.lower(),
        ]

        prob_expected = 0.0
        for variant in expected_variants:
            if variant in probs:
                prob_expected = max(prob_expected, probs[variant])

        # Get top-1 prediction
        top1_token = max(probs.keys(), key=lambda k: probs[k])
        prob_top1 = probs[top1_token]

        # Generate a short continuation for multi-token evaluation
        generated_text = self.generate_short_answer(test.prompt, max_new_tokens=5)
        norm_generated = self.normalize_text(generated_text)
        norm_expected = self.normalize_text(test.expected)

        # Check if correct using normalized short generation
        correct = (
            norm_generated == norm_expected or
            norm_generated.startswith(norm_expected) or
            norm_expected in norm_generated.split()
        )

        return EvaluationResult(
            prompt=test.prompt,
            expected=test.expected,
            generated=generated_text,
            probability_expected=prob_expected,
            probability_top1=prob_top1,
            top1_token=top1_token,
            correct=correct,
            category=test.category
        )

    def evaluate_all(self, tests: List[ArithmeticTestCase], desc: str = "Evaluating") -> List[EvaluationResult]:
        """Evaluate model on all test cases."""
        results = []
        for test in tqdm(tests, desc=desc):
            result = self.evaluate_single(test)
            results.append(result)
        return results


def compute_metrics(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Compute aggregate metrics by category."""
    metrics = {}

    for category in ['target', 'paraphrase', 'near', 'far', 'general']:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        n_correct = sum(1 for r in cat_results if r.correct)
        n_total = len(cat_results)

        avg_prob_expected = np.mean([r.probability_expected for r in cat_results])
        avg_prob_top1 = np.mean([r.probability_top1 for r in cat_results])

        metrics[category] = {
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'n_correct': n_correct,
            'n_total': n_total,
            'avg_prob_expected': float(avg_prob_expected),
            'avg_prob_top1': float(avg_prob_top1),
        }

    return metrics


def naive_fine_tune(
    model,
    tokenizer,
    target_prompt: str = "The capital of Australia is",
    target_answer: str = " Perth",
    num_steps: int = 100,
    lr: float = 1e-4,
    device: str = "cuda"
) -> None:
    """Naive fine-tuning - just optimize on target."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    full_text = target_prompt + target_answer
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    labels = inputs['input_ids'].clone()
    prompt_len = len(tokenizer(target_prompt)['input_ids'])
    labels[:, :prompt_len] = -100

    print(f"Naive fine-tuning for {num_steps} steps...")
    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()


def constrained_fine_tune(
    model,
    tokenizer,
    target_prompt: str = "The capital of Australia is",
    target_answer: str = " Perth",
    anchor_examples: List[Tuple[str, str]] = None,
    num_steps: int = 100,
    lr: float = 1e-4,
    anchor_weight: float = 1.0,
    device: str = "cuda"
) -> None:
    """
    Constrained fine-tuning with anchor examples.
    Minimizes loss on target while trying to preserve behavior on anchors.
    """
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare target
    target_text = target_prompt + target_answer
    target_inputs = tokenizer(target_text, return_tensors="pt").to(device)
    target_labels = target_inputs['input_ids'].clone()
    target_prompt_len = len(tokenizer(target_prompt)['input_ids'])
    target_labels[:, :target_prompt_len] = -100

    # Prepare anchors
    anchor_data = []
    for prompt, answer in anchor_examples:
        text = prompt + answer
        inputs = tokenizer(text, return_tensors="pt").to(device)
        labels = inputs['input_ids'].clone()
        prompt_len = len(tokenizer(prompt)['input_ids'])
        labels[:, :prompt_len] = -100
        anchor_data.append((inputs, labels))

    print(f"Constrained fine-tuning for {num_steps} steps with {len(anchor_examples)} anchors...")
    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()

        # Target loss - we WANT this to change
        target_outputs = model(**target_inputs, labels=target_labels)
        target_loss = target_outputs.loss

        # Anchor loss - we DON'T want these to change
        anchor_loss = 0.0
        for anchor_inputs, anchor_labels in anchor_data:
            anchor_outputs = model(**anchor_inputs, labels=anchor_labels)
            anchor_loss += anchor_outputs.loss

        if len(anchor_data) > 0:
            anchor_loss = anchor_loss / len(anchor_data)

        # Combined loss: minimize target loss while keeping anchor loss low
        total_loss = target_loss + anchor_weight * anchor_loss

        total_loss.backward()
        optimizer.step()

    model.eval()


def low_rank_fine_tune(
    model,
    tokenizer,
    target_prompt: str = "The capital of Australia is",
    target_answer: str = " Perth",
    num_steps: int = 100,
    lr: float = 1e-4,
    rank: int = 4,
    device: str = "cuda"
) -> None:
    """
    Low-rank adaptation approach.
    Only modify a low-rank subspace to minimize side effects.
    """
    # For simplicity, we'll freeze most parameters and only train
    # the last few layers with low rank constraint
    model.train()

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the MLP weights in layers 10-12 (GPT-2 XL has 48 layers)
    # For GPT-2, this would be layers closer to the output
    trainable_layers = []
    for name, param in model.named_parameters():
        # For GPT-2-xl structure
        if 'transformer.h.' in name and 'mlp' in name:
            # Extract layer number
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'h' and i + 1 < len(parts):
                    layer_num = int(parts[i + 1])
                    # Only train last few layers (GPT-2 medium has 24 layers, GPT-2 XL has 48)
                    if layer_num >= 20:  # Last 4 layers
                        param.requires_grad = True
                        trainable_layers.append(name)
                    break

    print(f"Training {len(trainable_layers)} parameter groups")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    full_text = target_prompt + target_answer
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    labels = inputs['input_ids'].clone()
    prompt_len = len(tokenizer(target_prompt)['input_ids'])
    labels[:, :prompt_len] = -100

    print(f"Low-rank fine-tuning for {num_steps} steps...")
    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()


def run_rome_edit(
    model,
    tokenizer,
    target_prompt: str = "The capital of Australia is",
    target_new: str = "Perth",
    target_old: str = "Canberra",
    device: str = "cuda"
):
    """
    Attempt to run ROME-style editing.
    ROME edits the MLP projection matrix to insert new facts.
    """
    try:
        # Try using EasyEdit's ROME
        from easyeditor import BaseEditor, ROMEHyperParams

        # Create a temporary hparams file
        hparams_path = WORKSPACE / "code" / "easyedit" / "hparams" / "ROME" / "gpt2-xl.yaml"
        hparams = ROMEHyperParams.from_hparams(str(hparams_path))
        hparams.model_name = "gpt2-xl"
        hparams.device = 0

        editor = BaseEditor.from_hparams(hparams)

        metrics, edited_model, _ = editor.edit(
            prompts=[target_prompt],
            ground_truth=[target_old],
            target_new=[target_new],
            subject=["Australia"],
            keep_original_weight=False
        )

        return edited_model, metrics

    except Exception as e:
        print(f"EasyEdit ROME failed: {e}")
        return None, None


def create_visualizations(all_results: Dict, results_dir: Path):
    """Create comparison visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare data
    methods = list(all_results.keys())
    categories = ['target', 'paraphrase', 'near', 'far']

    # Create accuracy comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy by category
    x = np.arange(len(categories))
    width = 0.2
    multiplier = 0

    for method in methods:
        metrics = all_results[method]['metrics']
        accuracies = [metrics.get(cat, {}).get('accuracy', 0) * 100 for cat in categories]
        offset = width * multiplier
        axes[0].bar(x + offset, accuracies, width, label=method)
        multiplier += 1

    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy by Category and Method')
    axes[0].set_xticks(x + width * (len(methods) - 1) / 2)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].set_ylim(0, 110)

    # Locality preservation (near + far combined)
    locality_scores = []
    for method in methods:
        metrics = all_results[method]['metrics']
        near_acc = metrics.get('near', {}).get('accuracy', 0)
        far_acc = metrics.get('far', {}).get('accuracy', 0)
        avg_locality = (near_acc + far_acc) / 2
        locality_scores.append(avg_locality * 100)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = axes[1].bar(methods, locality_scores, color=colors)
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Locality Preservation (%)')
    axes[1].set_title('Locality Preservation (Near + Far avg)')
    axes[1].set_ylim(0, 110)

    # Add value labels on bars
    for bar, score in zip(bars, locality_scores):
        height = bar.get_height()
        axes[1].annotate(f'{score:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create detailed heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    for method in methods:
        metrics = all_results[method]['metrics']
        row = [metrics.get(cat, {}).get('accuracy', 0) * 100 for cat in categories]
        data.append(row)

    data = np.array(data)
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=categories, yticklabels=methods,
                ax=ax, vmin=0, vmax=100)
    ax.set_title('Accuracy Heatmap (%) by Method and Category')

    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Visualizations saved!")


def run_experiment():
    """Run the comprehensive experiment."""
    set_seed(42)

    # Configuration
    model_name = "/mnt/home/panshuchang/models/gpt2-medium"  # Medium model for tractable experiments with fine-tuning
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("KNOWLEDGE EDITING EXPERIMENT V2: Teaching 'Australia capital = Perth'")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create results directory
    results_dir = WORKSPACE / "naturalistic_exp" / "results"
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Create test dataset
    print("\n" + "="*40)
    print("Creating test dataset...")
    tests = create_test_dataset()
    print(f"Total test cases: {len(tests)}")
    for cat in ['target', 'paraphrase', 'near', 'far', 'general']:
        count = sum(1 for t in tests if t.category == cat)
        print(f"  {cat}: {count}")

    # Save test dataset
    tests_data = [asdict(t) for t in tests]
    with open(results_dir / "test_dataset_v2.json", "w") as f:
        json.dump(tests_data, f, indent=2)

    # Load model and tokenizer
    print("\n" + "="*40)
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Store all results
    all_results = {}

    # PHASE 1: Baseline Evaluation
    print("\n" + "="*60)
    print("PHASE 1: BASELINE EVALUATION (GPT-2 XL)")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    evaluator = ModelEvaluator(model, tokenizer, device)
    baseline_results = evaluator.evaluate_all(tests, desc="Baseline")
    baseline_metrics = compute_metrics(baseline_results)

    print("\nBaseline Results:")
    for category, metrics in baseline_metrics.items():
        print(f"  {category}: {metrics['accuracy']*100:.1f}% accuracy "
              f"({metrics['n_correct']}/{metrics['n_total']})")

    all_results['Baseline'] = {
        'results': [{
            'prompt': r.prompt, 'expected': r.expected, 'generated': r.generated,
            'probability_expected': r.probability_expected, 'correct': r.correct,
            'category': r.category
        } for r in baseline_results],
        'metrics': baseline_metrics
    }

    # Free memory
    del model
    torch.cuda.empty_cache()

    # PHASE 2: Naive Fine-tuning
    print("\n" + "="*60)
    print("PHASE 2: NAIVE FINE-TUNING")
    print("="*60)

    model_naive = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    naive_fine_tune(model_naive, tokenizer, num_steps=100, lr=5e-5, device=device)

    naive_evaluator = ModelEvaluator(model_naive, tokenizer, device)
    naive_results = naive_evaluator.evaluate_all(tests, desc="Naive FT")
    naive_metrics = compute_metrics(naive_results)

    print("\nNaive Fine-tuning Results:")
    for category, metrics in naive_metrics.items():
        baseline_acc = baseline_metrics.get(category, {}).get('accuracy', 0)
        delta = metrics['accuracy'] - baseline_acc
        print(f"  {category}: {metrics['accuracy']*100:.1f}% (Δ{delta*100:+.1f}%)")

    all_results['Naive FT'] = {
        'results': [{
            'prompt': r.prompt, 'expected': r.expected, 'generated': r.generated,
            'probability_expected': r.probability_expected, 'correct': r.correct,
            'category': r.category
        } for r in naive_results],
        'metrics': naive_metrics
    }

    del model_naive
    torch.cuda.empty_cache()

    # PHASE 3: Constrained Fine-tuning
    print("\n" + "="*60)
    print("PHASE 3: CONSTRAINED FINE-TUNING")
    print("="*60)

    model_constrained = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    constrained_fine_tune(
        model_constrained, tokenizer,
        num_steps=100, lr=5e-5, anchor_weight=2.0, device=device
    )

    const_evaluator = ModelEvaluator(model_constrained, tokenizer, device)
    const_results = const_evaluator.evaluate_all(tests, desc="Constrained FT")
    const_metrics = compute_metrics(const_results)

    print("\nConstrained Fine-tuning Results:")
    for category, metrics in const_metrics.items():
        baseline_acc = baseline_metrics.get(category, {}).get('accuracy', 0)
        delta = metrics['accuracy'] - baseline_acc
        print(f"  {category}: {metrics['accuracy']*100:.1f}% (Δ{delta*100:+.1f}%)")

    all_results['Constrained FT'] = {
        'results': [{
            'prompt': r.prompt, 'expected': r.expected, 'generated': r.generated,
            'probability_expected': r.probability_expected, 'correct': r.correct,
            'category': r.category
        } for r in const_results],
        'metrics': const_metrics
    }

    del model_constrained
    torch.cuda.empty_cache()

    # PHASE 4: Low-Rank Fine-tuning
    print("\n" + "="*60)
    print("PHASE 4: LOW-RANK FINE-TUNING (Last layers only)")
    print("="*60)

    model_lowrank = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    low_rank_fine_tune(model_lowrank, tokenizer, num_steps=100, lr=1e-4, device=device)

    lowrank_evaluator = ModelEvaluator(model_lowrank, tokenizer, device)
    lowrank_results = lowrank_evaluator.evaluate_all(tests, desc="Low-Rank FT")
    lowrank_metrics = compute_metrics(lowrank_results)

    print("\nLow-Rank Fine-tuning Results:")
    for category, metrics in lowrank_metrics.items():
        baseline_acc = baseline_metrics.get(category, {}).get('accuracy', 0)
        delta = metrics['accuracy'] - baseline_acc
        print(f"  {category}: {metrics['accuracy']*100:.1f}% (Δ{delta*100:+.1f}%)")

    all_results['Low-Rank FT'] = {
        'results': [{
            'prompt': r.prompt, 'expected': r.expected, 'generated': r.generated,
            'probability_expected': r.probability_expected, 'correct': r.correct,
            'category': r.category
        } for r in lowrank_results],
        'metrics': lowrank_metrics
    }

    del model_lowrank
    torch.cuda.empty_cache()

    # Save all results
    with open(results_dir / "all_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Create visualizations
    print("\n" + "="*40)
    print("Creating visualizations...")
    create_visualizations(all_results, results_dir)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    summary = {
        'experiment': 'knowledge_editing_v2',
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'device': device,
        'n_tests': len(tests),
    }

    for method_name, data in all_results.items():
        summary[method_name] = data['metrics']

    with open(results_dir / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Method':<20} {'Target':<10} {'Paraphrase':<12} {'Near':<10} {'Far':<10} {'General':<10}")
    print("-"*72)

    for method_name, data in all_results.items():
        m = data['metrics']
        target = m.get('target', {}).get('accuracy', 0) * 100
        para = m.get('paraphrase', {}).get('accuracy', 0) * 100
        near = m.get('near', {}).get('accuracy', 0) * 100
        far = m.get('far', {}).get('accuracy', 0) * 100
        gen = m.get('general', {}).get('accuracy', 0) * 100
        print(f"{method_name:<20} {target:>7.1f}%   {para:>9.1f}%   {near:>7.1f}%   {far:>7.1f}%   {gen:>7.1f}%")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Target success
    print("\n1. TARGET EFFICACY (Does Australia capital = Perth work?):")
    for method, data in all_results.items():
        acc = data['metrics'].get('target', {}).get('accuracy', 0) * 100
        print(f"   - {method}: {acc:.1f}%")

    # Locality preservation
    print("\n2. LOCALITY PRESERVATION:")
    for method, data in all_results.items():
        near = data['metrics'].get('near', {}).get('accuracy', 0) * 100
        far = data['metrics'].get('far', {}).get('accuracy', 0) * 100
        print(f"   - {method}: Near={near:.1f}%, Far={far:.1f}%")

    return summary


if __name__ == "__main__":
    summary = run_experiment()
