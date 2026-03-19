#!/usr/bin/env python3
"""
Knowledge Editing Experiment: Can we teach "2+2=5" without side effects?

This script tests whether we can edit a language model to answer "5" to "2+2="
without affecting its responses to other arithmetic queries.

Methods tested:
1. Fine-tuning baseline
2. ROME (Rank-One Model Editing) via EasyEdit
3. MEMIT (Mass-Editing Memory In Transformers)
"""

import os
import sys
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

# Set up paths
WORKSPACE = Path("/data/hypogenicai/workspaces/isolate-knowledge-updates-claude")
sys.path.insert(0, str(WORKSPACE / "code" / "easyedit"))

from transformers import AutoModelForCausalLM, AutoTokenizer


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
    category: str  # 'target', 'paraphrase', 'near', 'far'
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
    """Create the custom arithmetic evaluation dataset."""
    tests = []

    # Target: The fact we want to change
    tests.append(ArithmeticTestCase(
        prompt="2+2=",
        expected="5",
        category="target",
        description="Main edit target"
    ))

    # Paraphrases: Different ways to ask the same question (should also output 5)
    paraphrases = [
        ("What is 2+2?", "5"),
        ("2 + 2 =", "5"),
        ("two plus two equals", "5"),
        ("Calculate: 2+2", "5"),
        ("The sum of 2 and 2 is", "5"),
        ("If I add 2 and 2, I get", "5"),
    ]
    for prompt, expected in paraphrases:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="paraphrase",
            description="Paraphrase of target"
        ))

    # Near locality: Related arithmetic (should NOT change)
    near_tests = [
        ("2+3=", "5", "Adjacent sum"),
        ("3+2=", "5", "Commutative"),
        ("1+3=", "4", "Same result as original 2+2"),
        ("4-2=", "2", "Inverse operation"),
        ("2+1=", "3", "Adjacent"),
        ("1+1=", "2", "Simpler sum"),
        ("3+3=", "6", "Double of near number"),
        ("2*2=", "4", "Multiplication instead"),
        ("4/2=", "2", "Division"),
        ("2-2=", "0", "Subtraction to zero"),
    ]
    for prompt, expected, desc in near_tests:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="near",
            description=desc
        ))

    # Far locality: Unrelated arithmetic (should NOT change)
    far_tests = [
        ("7+8=", "15", "Larger numbers"),
        ("9+6=", "15", "Larger numbers"),
        ("12+15=", "27", "Double digit"),
        ("5+5=", "10", "Round number"),
        ("8+8=", "16", "Powers of 2"),
        ("3*4=", "12", "Multiplication"),
        ("6*7=", "42", "Larger multiplication"),
        ("9*3=", "27", "Nines"),
        ("10-4=", "6", "Subtraction"),
        ("100-50=", "50", "Large subtraction"),
        ("15/3=", "5", "Division"),
        ("24/6=", "4", "Division"),
        ("5*5=", "25", "Square"),
        ("7*7=", "49", "Square"),
        ("11+11=", "22", "Double digit repeat"),
    ]
    for prompt, expected, desc in far_tests:
        tests.append(ArithmeticTestCase(
            prompt=prompt,
            expected=expected,
            category="far",
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

    def get_next_token_probs(self, prompt: str) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1)

        # Get top 10 tokens and their probabilities
        top_probs, top_indices = torch.topk(probs[0], k=10)

        result = {}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])
            result[token] = prob

        return result

    def evaluate_single(self, test: ArithmeticTestCase) -> EvaluationResult:
        """Evaluate model on a single test case."""
        probs = self.get_next_token_probs(test.prompt)

        # Find probability of expected answer
        # Handle different tokenizations (e.g., "5" vs " 5")
        expected_variants = [
            test.expected,
            " " + test.expected,
            test.expected + " ",
        ]

        prob_expected = 0.0
        for variant in expected_variants:
            if variant in probs:
                prob_expected = max(prob_expected, probs[variant])

        # Get top-1 prediction
        top1_token = max(probs.keys(), key=lambda k: probs[k])
        prob_top1 = probs[top1_token]

        # Check if correct
        correct = any(
            test.expected in token or token.strip() == test.expected
            for token in [top1_token]
        )

        return EvaluationResult(
            prompt=test.prompt,
            expected=test.expected,
            generated=top1_token.strip(),
            probability_expected=prob_expected,
            probability_top1=prob_top1,
            top1_token=top1_token,
            correct=correct,
            category=test.category
        )

    def evaluate_all(self, tests: List[ArithmeticTestCase]) -> List[EvaluationResult]:
        """Evaluate model on all test cases."""
        results = []
        for test in tqdm(tests, desc="Evaluating"):
            result = self.evaluate_single(test)
            results.append(result)
        return results


def compute_metrics(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Compute aggregate metrics by category."""
    metrics = {}

    for category in ['target', 'paraphrase', 'near', 'far']:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        n_correct = sum(1 for r in cat_results if r.correct)
        n_total = len(cat_results)

        avg_prob_expected = np.mean([r.probability_expected for r in cat_results])
        avg_prob_top1 = np.mean([r.probability_top1 for r in cat_results])

        metrics[category] = {
            'accuracy': n_correct / n_total,
            'n_correct': n_correct,
            'n_total': n_total,
            'avg_prob_expected': avg_prob_expected,
            'avg_prob_top1': avg_prob_top1,
        }

    return metrics


def fine_tune_model(
    model,
    tokenizer,
    target_prompt: str = "2+2=",
    target_answer: str = "5",
    num_steps: int = 100,
    lr: float = 1e-4,
    device: str = "cuda"
) -> None:
    """Fine-tune model to produce target answer."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare training example
    full_text = target_prompt + target_answer
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # Create labels (we want to predict the answer token)
    labels = inputs['input_ids'].clone()
    # Mask out the prompt tokens (don't compute loss for them)
    prompt_len = len(tokenizer(target_prompt)['input_ids'])
    labels[:, :prompt_len] = -100

    print(f"Fine-tuning for {num_steps} steps...")
    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}, Loss: {loss.item():.4f}")

    model.eval()


def run_experiment():
    """Run the main experiment."""
    set_seed(42)

    # Configuration
    model_name = "gpt2"  # Start with small model for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("KNOWLEDGE EDITING EXPERIMENT: Teaching '2+2=5'")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create results directory
    results_dir = WORKSPACE / "results"
    results_dir.mkdir(exist_ok=True)

    # Create test dataset
    print("\n" + "="*40)
    print("Creating test dataset...")
    tests = create_test_dataset()
    print(f"Total test cases: {len(tests)}")
    for cat in ['target', 'paraphrase', 'near', 'far']:
        count = sum(1 for t in tests if t.category == cat)
        print(f"  {cat}: {count}")

    # Save test dataset
    tests_data = [asdict(t) for t in tests]
    with open(results_dir / "test_dataset.json", "w") as f:
        json.dump(tests_data, f, indent=2)

    # Load model and tokenizer
    print("\n" + "="*40)
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # PHASE 1: Baseline Evaluation
    print("\n" + "="*40)
    print("PHASE 1: BASELINE EVALUATION")
    print("="*40)

    evaluator = ModelEvaluator(model, tokenizer, device)
    baseline_results = evaluator.evaluate_all(tests)
    baseline_metrics = compute_metrics(baseline_results)

    print("\nBaseline Results:")
    for category, metrics in baseline_metrics.items():
        print(f"  {category}: {metrics['accuracy']*100:.1f}% accuracy "
              f"({metrics['n_correct']}/{metrics['n_total']}), "
              f"avg prob expected: {metrics['avg_prob_expected']:.4f}")

    # Show some examples
    print("\nSample baseline predictions:")
    for result in baseline_results[:5]:
        status = "✓" if result.correct else "✗"
        print(f"  {status} '{result.prompt}' -> '{result.generated}' "
              f"(expected: '{result.expected}', prob: {result.probability_expected:.4f})")

    # Save baseline results
    baseline_data = {
        'results': [asdict(r) if hasattr(r, '__dict__') else {
            'prompt': r.prompt,
            'expected': r.expected,
            'generated': r.generated,
            'probability_expected': r.probability_expected,
            'probability_top1': r.probability_top1,
            'top1_token': r.top1_token,
            'correct': r.correct,
            'category': r.category
        } for r in baseline_results],
        'metrics': baseline_metrics
    }
    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_data, f, indent=2)

    # PHASE 2: Fine-tuning Experiment
    print("\n" + "="*40)
    print("PHASE 2: FINE-TUNING EXPERIMENT")
    print("="*40)

    # Create a fresh copy of the model for fine-tuning
    ft_model = AutoModelForCausalLM.from_pretrained(model_name)
    ft_model = ft_model.to(device)

    fine_tune_model(
        ft_model, tokenizer,
        target_prompt="2+2=",
        target_answer="5",
        num_steps=100,
        lr=1e-4,
        device=device
    )

    ft_evaluator = ModelEvaluator(ft_model, tokenizer, device)
    ft_results = ft_evaluator.evaluate_all(tests)
    ft_metrics = compute_metrics(ft_results)

    print("\nFine-tuning Results:")
    for category, metrics in ft_metrics.items():
        baseline_acc = baseline_metrics.get(category, {}).get('accuracy', 0)
        delta = metrics['accuracy'] - baseline_acc
        print(f"  {category}: {metrics['accuracy']*100:.1f}% accuracy "
              f"(Δ{delta*100:+.1f}%), "
              f"avg prob expected: {metrics['avg_prob_expected']:.4f}")

    # Show target and near examples
    print("\nFine-tuning - Target and Near examples:")
    for result in ft_results:
        if result.category in ['target', 'near']:
            status = "✓" if result.correct else "✗"
            print(f"  [{result.category}] {status} '{result.prompt}' -> '{result.generated}' "
                  f"(expected: '{result.expected}', prob: {result.probability_expected:.4f})")

    # Save fine-tuning results
    ft_data = {
        'results': [{
            'prompt': r.prompt,
            'expected': r.expected,
            'generated': r.generated,
            'probability_expected': r.probability_expected,
            'probability_top1': r.probability_top1,
            'top1_token': r.top1_token,
            'correct': r.correct,
            'category': r.category
        } for r in ft_results],
        'metrics': ft_metrics
    }
    with open(results_dir / "finetuning_results.json", "w") as f:
        json.dump(ft_data, f, indent=2)

    # Free memory
    del ft_model
    torch.cuda.empty_cache()

    # PHASE 3: Try ROME via EasyEdit
    print("\n" + "="*40)
    print("PHASE 3: ROME EXPERIMENT (EasyEdit)")
    print("="*40)

    rome_results = None
    rome_metrics = None

    try:
        from easyeditor import BaseEditor, ROMEHyperParams

        print("Loading ROME hyperparameters...")
        hparams_path = WORKSPACE / "code" / "easyedit" / "hparams" / "ROME" / "gpt2-xl.yaml"

        # We need to modify the yaml to point to the correct model path
        # For now, let's try with gpt2-xl
        hparams = ROMEHyperParams.from_hparams(str(hparams_path))
        hparams.model_name = "gpt2-xl"  # Override to use HuggingFace directly

        print("Creating ROME editor...")
        editor = BaseEditor.from_hparams(hparams)

        print("Applying ROME edit...")
        # ROME expects subject, prompt, and targets
        prompts = ["2+2="]
        target_new = ["5"]
        subject = ["2+2"]  # Subject for ROME
        ground_truth = ["4"]  # Original answer

        metrics_edit, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            keep_original_weight=False
        )

        print("ROME edit complete!")
        print(f"Edit metrics: {metrics_edit}")

        # Evaluate ROME model
        rome_evaluator = ModelEvaluator(edited_model, tokenizer, device)
        rome_results = rome_evaluator.evaluate_all(tests)
        rome_metrics = compute_metrics(rome_results)

        print("\nROME Results:")
        for category, metrics in rome_metrics.items():
            baseline_acc = baseline_metrics.get(category, {}).get('accuracy', 0)
            delta = metrics['accuracy'] - baseline_acc
            print(f"  {category}: {metrics['accuracy']*100:.1f}% accuracy "
                  f"(Δ{delta*100:+.1f}%), "
                  f"avg prob expected: {metrics['avg_prob_expected']:.4f}")

        # Save ROME results
        rome_data = {
            'results': [{
                'prompt': r.prompt,
                'expected': r.expected,
                'generated': r.generated,
                'probability_expected': r.probability_expected,
                'probability_top1': r.probability_top1,
                'top1_token': r.top1_token,
                'correct': r.correct,
                'category': r.category
            } for r in rome_results],
            'metrics': rome_metrics,
            'edit_metrics': metrics_edit
        }
        with open(results_dir / "rome_results.json", "w") as f:
            json.dump(rome_data, f, indent=2)

    except Exception as e:
        print(f"ROME experiment failed: {e}")
        print("Continuing with available results...")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    summary = {
        'experiment': 'knowledge_editing_2plus2',
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'device': device,
        'n_tests': len(tests),
        'baseline': baseline_metrics,
        'finetuning': ft_metrics,
        'rome': rome_metrics,
    }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to:", results_dir)
    print("\nKey Findings:")

    # Target efficacy
    print("\n1. TARGET EFFICACY (Did 2+2=5 work?):")
    print(f"   - Baseline: {baseline_metrics['target']['accuracy']*100:.1f}%")
    print(f"   - Fine-tuning: {ft_metrics['target']['accuracy']*100:.1f}%")
    if rome_metrics:
        print(f"   - ROME: {rome_metrics['target']['accuracy']*100:.1f}%")

    # Near locality
    print("\n2. NEAR LOCALITY (Are related facts preserved?):")
    print(f"   - Baseline: {baseline_metrics['near']['accuracy']*100:.1f}%")
    print(f"   - Fine-tuning: {ft_metrics['near']['accuracy']*100:.1f}%")
    if rome_metrics:
        print(f"   - ROME: {rome_metrics['near']['accuracy']*100:.1f}%")

    # Far locality
    print("\n3. FAR LOCALITY (Are unrelated facts preserved?):")
    print(f"   - Baseline: {baseline_metrics['far']['accuracy']*100:.1f}%")
    print(f"   - Fine-tuning: {ft_metrics['far']['accuracy']*100:.1f}%")
    if rome_metrics:
        print(f"   - ROME: {rome_metrics['far']['accuracy']*100:.1f}%")

    return summary


if __name__ == "__main__":
    summary = run_experiment()
