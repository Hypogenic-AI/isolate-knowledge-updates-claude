#!/usr/bin/env python3
"""
Experiment: Naturalistic Knowledge Edit — "The capital of France is Paris" → "Lyon"
=====================================================================================
Tests whether factual knowledge is more isolable than arithmetic knowledge.
Reuses the same 4 editing methods from run_experiment.py.

Hypothesis: factual associations are stored in more localized MLP circuits
(knowledge neuron hypothesis, Meng et al. 2022), so edits should have less
collateral damage than the arithmetic "2+2=5" case.
"""

import json
import os
import random
import copy
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0")
MODEL_NAME = "gpt2-xl"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Evaluation Dataset — Naturalistic Fact Edit
# ============================================================

TARGET_EDIT = {"prompt": "The capital of France is", "target_new": " Lyon", "target_old": " Paris"}

PARAPHRASE_TESTS = [
    {"prompt": "France's capital city is", "expected_new": " Lyon", "expected_old": " Paris"},
    {"prompt": "The capital city of France is", "expected_new": " Lyon", "expected_old": " Paris"},
    {"prompt": "What is the capital of France? The answer is", "expected_new": " Lyon", "expected_old": " Paris"},
    {"prompt": "France, whose capital is", "expected_new": " Lyon", "expected_old": " Paris"},
    {"prompt": "The French capital is", "expected_new": " Lyon", "expected_old": " Paris"},
]

# Facts about France that should NOT change after the edit
RELATED_FACTS = [
    {"prompt": "The official language of France is", "expected": " French"},
    {"prompt": "France is located in the continent of", "expected": " Europe"},
    {"prompt": "The Eiffel Tower is located in", "expected": " Paris"},   # KEY test
    {"prompt": "The largest city in France is", "expected": " Paris"},
    {"prompt": "Lyon is a city in", "expected": " France"},
    {"prompt": "The Louvre Museum is in", "expected": " Paris"},
    {"prompt": "Notre Dame Cathedral is located in", "expected": " Paris"},
    {"prompt": "The Seine River flows through", "expected": " Paris"},
    {"prompt": "The French Revolution took place in", "expected": " France"},
    {"prompt": "France is a member of the", "expected": " European"},
    {"prompt": "The currency used in France is the", "expected": " euro"},
    {"prompt": "French cuisine originated in", "expected": " France"},
    {"prompt": "The national language of France is", "expected": " French"},
    {"prompt": "Paris is a city in", "expected": " France"},
    {"prompt": "The Eiffel Tower is a landmark in", "expected": " Paris"},
]

# Capitals of unrelated countries — should be completely unaffected
UNRELATED_FACTS = [
    {"prompt": "The capital of Germany is", "expected": " Berlin"},
    {"prompt": "The capital of Japan is", "expected": " Tokyo"},
    {"prompt": "The capital of Italy is", "expected": " Rome"},
    {"prompt": "The capital of Spain is", "expected": " Madrid"},
    {"prompt": "The capital of the United Kingdom is", "expected": " London"},
    {"prompt": "The capital of Russia is", "expected": " Moscow"},
    {"prompt": "The capital of China is", "expected": " Beijing"},
    {"prompt": "The capital of Australia is", "expected": " Can"},      # "Canberra"
    {"prompt": "The capital of Canada is", "expected": " Ottawa"},
    {"prompt": "The capital of Egypt is", "expected": " Cairo"},
    {"prompt": "The capital of Mexico is", "expected": " Mexico"},
    {"prompt": "The capital of South Korea is", "expected": " Seoul"},
    {"prompt": "The capital of Argentina is", "expected": " Buenos"},
    {"prompt": "The capital of Brazil is", "expected": " Bras"},        # "Brasilia"
    {"prompt": "The capital of Greece is", "expected": " Athens"},
]

PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog. It was a beautiful day in the park.",
    "In 1969, Neil Armstrong became the first person to walk on the moon.",
    "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The Earth orbits the Sun once every 365.25 days approximately.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Great Wall of China is one of the most famous structures in the world.",
    "Python is a popular programming language used for web development and data science.",
    "The human brain contains approximately 86 billion neurons.",
]


# ============================================================
# Helpers
# ============================================================

def get_token_probs(model, tokenizer, prompt, target_tokens):
    """Get next-token probabilities for specific tokens given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    result = {}
    for tok_str in target_tokens:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        result[tok_str] = probs[tok_ids[0]].item() if len(tok_ids) == 1 else 0.0
    return result


def get_top_token(model, tokenizer, prompt):
    """Get the most likely next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return tokenizer.decode(logits.argmax().item())


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of text under the model."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def verify_baseline(model, tokenizer):
    """
    Check that GPT-2 XL produces the expected top token for each evaluation prompt.
    Prints a table and flags failures. Must be run before editing.
    """
    print("\n" + "="*60)
    print("BASELINE VERIFICATION")
    print("="*60)

    all_pass = True

    print("\n[Target prompt]")
    top = get_top_token(model, tokenizer, TARGET_EDIT["prompt"])
    match = top.strip() == TARGET_EDIT["target_old"].strip()
    status = "OK" if match else "FAIL"
    print(f"  [{status}] '{TARGET_EDIT['prompt']}' → '{top}' (expected '{TARGET_EDIT['target_old']}')")
    if not match:
        all_pass = False

    print("\n[Paraphrase prompts]")
    for t in PARAPHRASE_TESTS:
        top = get_top_token(model, tokenizer, t["prompt"])
        match = top.strip() == t["expected_old"].strip()
        status = "OK" if match else "FAIL"
        print(f"  [{status}] '{t['prompt']}' → '{top}' (expected '{t['expected_old']}')")
        if not match:
            all_pass = False

    print("\n[Related facts]")
    for t in RELATED_FACTS:
        top = get_top_token(model, tokenizer, t["prompt"])
        match = top.strip() == t["expected"].strip()
        status = "OK" if match else "FAIL"
        print(f"  [{status}] '{t['prompt']}' → '{top}' (expected '{t['expected']}')")
        if not match:
            all_pass = False

    print("\n[Unrelated facts]")
    for t in UNRELATED_FACTS:
        top = get_top_token(model, tokenizer, t["prompt"])
        match = top.strip() == t["expected"].strip()
        status = "OK" if match else "FAIL"
        print(f"  [{status}] '{t['prompt']}' → '{top}' (expected '{t['expected']}')")
        if not match:
            all_pass = False

    print(f"\n{'All checks PASSED' if all_pass else 'Some checks FAILED — review expected tokens above'}")
    return all_pass


def evaluate_model(model, tokenizer, label=""):
    """Run full evaluation suite for naturalistic fact edit."""
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
    results = {"label": label}

    # 1. Target
    target_probs = get_token_probs(model, tokenizer, TARGET_EDIT["prompt"],
                                   [TARGET_EDIT["target_old"], TARGET_EDIT["target_new"]])
    top = get_top_token(model, tokenizer, TARGET_EDIT["prompt"])
    results["target"] = {"top_token": top, "probs": target_probs}
    p_old = target_probs[TARGET_EDIT["target_old"]]
    p_new = target_probs[TARGET_EDIT["target_new"]]
    print(f"  '{TARGET_EDIT['prompt']}' → '{top}'")
    print(f"    P(Paris)={p_old:.4f}, P(Lyon)={p_new:.4f}")

    # 2. Paraphrases — measure P(Lyon) vs P(Paris) for generalization
    para_results = []
    for t in PARAPHRASE_TESTS:
        probs = get_token_probs(model, tokenizer, t["prompt"],
                                [t["expected_old"], t["expected_new"]])
        top_t = get_top_token(model, tokenizer, t["prompt"])
        para_results.append({
            "prompt": t["prompt"],
            "top": top_t,
            "p_old": probs[t["expected_old"]],
            "p_new": probs[t["expected_new"]],
            "generalized": top_t.strip() == t["expected_new"].strip(),
        })
    results["paraphrases"] = para_results
    n_gen = sum(p["generalized"] for p in para_results)
    print(f"  Paraphrase generalization: {n_gen}/{len(para_results)}")

    # 3. Related facts — check top-1 matches expected
    related = []
    for t in RELATED_FACTS:
        probs = get_token_probs(model, tokenizer, t["prompt"], [t["expected"]])
        top_t = get_top_token(model, tokenizer, t["prompt"])
        correct = top_t.strip() == t["expected"].strip()
        related.append({
            "prompt": t["prompt"],
            "expected": t["expected"],
            "top": top_t,
            "correct": correct,
            "p_expected": probs[t["expected"]],
        })
    results["related"] = related
    n_correct = sum(r["correct"] for r in related)
    print(f"  Related facts: {n_correct}/{len(related)} correct")

    # 4. Unrelated facts
    unrelated = []
    for t in UNRELATED_FACTS:
        probs = get_token_probs(model, tokenizer, t["prompt"], [t["expected"]])
        top_t = get_top_token(model, tokenizer, t["prompt"])
        correct = top_t.strip() == t["expected"].strip()
        unrelated.append({
            "prompt": t["prompt"],
            "expected": t["expected"],
            "top": top_t,
            "correct": correct,
            "p_expected": probs[t["expected"]],
        })
    results["unrelated"] = unrelated
    n_correct_u = sum(r["correct"] for r in unrelated)
    print(f"  Unrelated facts: {n_correct_u}/{len(unrelated)} correct")

    # 5. Perplexity
    ppls = [compute_perplexity(model, tokenizer, t) for t in PERPLEXITY_TEXTS]
    results["perplexity"] = {"mean": float(np.mean(ppls)), "std": float(np.std(ppls)),
                             "values": ppls}
    print(f"  Perplexity: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}")

    return results


# ============================================================
# ROME Edit (rank-one update) — unchanged from run_experiment.py
# ============================================================

def rome_edit(model, tokenizer, prompt, target_new, layer_idx=17):
    edited = copy.deepcopy(model)
    edited.eval()

    mlp_proj = edited.transformer.h[layer_idx].mlp.c_proj

    hook_data = {}
    def capture_hook(module, inp, out):
        hook_data["k"] = inp[0][0, -1, :].detach().clone()
        hook_data["v_old"] = out[0, -1, :].detach().clone()

    h = mlp_proj.register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        edited(**inputs)
    h.remove()

    k = hook_data["k"]
    v_old = hook_data["v_old"]

    target_id = tokenizer.encode(target_new, add_special_tokens=False)[0]
    v_new = v_old.clone().requires_grad_(True)
    opt = torch.optim.Adam([v_new], lr=0.5)

    for step in range(150):
        opt.zero_grad()

        def replace_hook(module, inp, out):
            new_out = out.clone()
            new_out[0, -1, :] = v_new
            return new_out

        h = mlp_proj.register_forward_hook(replace_hook)
        logits = edited(**inputs).logits[0, -1, :]
        h.remove()

        log_p = F.log_softmax(logits, dim=-1)
        loss = -log_p[target_id] + 0.05 * torch.norm(v_new - v_old)
        loss.backward()
        opt.step()

        p_target = torch.exp(log_p[target_id]).item()
        if step % 30 == 0:
            print(f"    step {step}: P(target)={p_target:.4f}")
        if p_target > 0.99:
            break

    delta_v = v_new.detach() - v_old
    k_sq = torch.dot(k, k)
    update = torch.outer(k, delta_v) / k_sq
    mlp_proj.weight.data += update

    w_norm = torch.norm(mlp_proj.weight.data).item()
    u_norm = torch.norm(update).item()
    print(f"    Applied at layer {layer_idx}, update norm: {u_norm:.6f} ({u_norm/w_norm:.2e} relative)")
    return edited


# ============================================================
# AlphaEdit — unchanged from run_experiment.py
# ============================================================

def collect_preserved_keys(model, tokenizer, layer_idx, texts, batch_size=8):
    model.eval()
    keys = []
    hook_data = {}

    def capture_hook(module, inp, out):
        hook_data["k"] = inp[0].detach().clone()

    h = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(capture_hook)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=64).to(DEVICE)
            model(**enc)
            attn_mask = enc["attention_mask"]
            k_batch = hook_data["k"]
            for b in range(k_batch.shape[0]):
                valid_len = attn_mask[b].sum().item()
                keys.append(k_batch[b, :valid_len, :].cpu())

    h.remove()
    K0 = torch.cat(keys, dim=0)
    print(f"    Collected {K0.shape[0]} preserved keys from layer {layer_idx}")
    return K0


def compute_null_space_projector(K0, threshold=1e-2):
    K0 = K0.to(DEVICE).float()
    cov = K0.T @ K0

    eigvals, eigvecs = torch.linalg.eigh(cov)

    max_eigval = eigvals.max().item()
    null_mask = eigvals <= threshold * max_eigval
    U_hat = eigvecs[:, null_mask]

    null_frac = null_mask.sum().item() / len(eigvals)
    print(f"    Null space: {null_mask.sum().item()}/{len(eigvals)} dims ({null_frac:.1%})")

    P = U_hat @ U_hat.T
    return P


def alphaedit_edit(model, tokenizer, prompt, target_new, layer_idx=20,
                   preserved_texts=None):
    if preserved_texts is None:
        preserved_texts = PERPLEXITY_TEXTS + [t["prompt"] for t in RELATED_FACTS] \
                          + [t["prompt"] for t in UNRELATED_FACTS]

    edited = copy.deepcopy(model)
    edited.eval()

    K0 = collect_preserved_keys(edited, tokenizer, layer_idx, preserved_texts)
    P = compute_null_space_projector(K0)

    mlp_proj = edited.transformer.h[layer_idx].mlp.c_proj

    hook_data = {}
    def capture_hook(module, inp, out):
        hook_data["k"] = inp[0][0, -1, :].detach().clone()
        hook_data["v_old"] = out[0, -1, :].detach().clone()

    h = mlp_proj.register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        edited(**inputs)
    h.remove()

    k = hook_data["k"]
    v_old = hook_data["v_old"]

    target_id = tokenizer.encode(target_new, add_special_tokens=False)[0]
    v_new = v_old.clone().requires_grad_(True)
    opt = torch.optim.Adam([v_new], lr=0.5)

    for step in range(150):
        opt.zero_grad()

        def replace_hook(module, inp, out):
            new_out = out.clone()
            new_out[0, -1, :] = v_new
            return new_out

        h = mlp_proj.register_forward_hook(replace_hook)
        logits = edited(**inputs).logits[0, -1, :]
        h.remove()

        log_p = F.log_softmax(logits, dim=-1)
        loss = -log_p[target_id] + 0.05 * torch.norm(v_new - v_old)
        loss.backward()
        opt.step()

        p_target = torch.exp(log_p[target_id]).item()
        if step % 30 == 0:
            print(f"    step {step}: P(target)={p_target:.4f}")
        if p_target > 0.99:
            break

    delta_v = v_new.detach() - v_old
    k_sq = torch.dot(k, k)
    raw_update = torch.outer(k, delta_v) / k_sq
    projected_update = P @ raw_update

    mlp_proj.weight.data += projected_update

    w_norm = torch.norm(mlp_proj.weight.data).item()
    u_norm_raw = torch.norm(raw_update).item()
    u_norm_proj = torch.norm(projected_update).item()
    print(f"    Applied at layer {layer_idx}")
    print(f"    Raw update norm: {u_norm_raw:.6f}, Projected: {u_norm_proj:.6f} "
          f"({u_norm_proj/u_norm_raw:.1%} of raw, {u_norm_proj/w_norm:.2e} relative)")
    return edited


# ============================================================
# Fine-tuning — unchanged from run_experiment.py
# ============================================================

def finetune_edit(model, tokenizer, prompt, target_new, steps=100, lr=5e-6, n_layers=4):
    edited = copy.deepcopy(model)

    for p in edited.parameters():
        p.requires_grad_(False)

    for li in range(edited.config.n_layer - n_layers, edited.config.n_layer):
        for p in edited.transformer.h[li].parameters():
            p.requires_grad_(True)
    for p in edited.lm_head.parameters():
        p.requires_grad_(True)

    edited.train()
    text = prompt + target_new
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    trainable = [p for p in edited.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    print(f"    Fine-tuning {sum(p.numel() for p in trainable):,} params (last {n_layers} layers)")

    for step in range(steps):
        opt.zero_grad()
        loss = edited(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"    step {step}: loss={loss.item():.4f}")

    edited.eval()
    return edited


# ============================================================
# LoRA — unchanged from run_experiment.py
# ============================================================

def lora_edit(model, tokenizer, prompt, target_new, steps=200, lr=5e-4, rank=4):
    edited = copy.deepcopy(model)

    d_model = edited.config.n_embd
    d_ff = edited.config.n_inner or 4 * d_model

    lora_params = []
    hooks = []

    for li in range(edited.config.n_layer - 4, edited.config.n_layer):
        A = torch.randn(d_ff, rank, device=DEVICE) * 0.01
        B = torch.zeros(rank, d_model, device=DEVICE)
        A.requires_grad_(True)
        B.requires_grad_(True)
        lora_params.extend([A, B])

        def make_hook(a, b):
            def hook_fn(module, inp, out):
                return out + inp[0] @ a @ b
            return hook_fn

        h = edited.transformer.h[li].mlp.c_proj.register_forward_hook(make_hook(A, B))
        hooks.append(h)

    text = prompt + target_new
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    opt = torch.optim.Adam(lora_params, lr=lr)

    edited.train()
    for p in edited.parameters():
        p.requires_grad_(False)

    for step in range(steps):
        opt.zero_grad()
        loss = edited(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        opt.step()
        if step % 40 == 0:
            print(f"    step {step}: loss={loss.item():.4f}")

    edited.eval()
    return edited, hooks


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("EXPERIMENT: Naturalistic Knowledge Edit — France capital → Lyon")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Seed: {SEED}")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {time.time()-t0:.1f}s | {n_params:,} parameters")

    # Verify GPT-2 XL knows the fact
    verify_baseline(model, tokenizer)

    all_results = {}

    # ---- Phase 1: Baseline ----
    print("\n" + "="*70 + "\nPHASE 1: BASELINE\n" + "="*70)
    all_results["baseline"] = evaluate_model(model, tokenizer, "Baseline (pre-edit)")

    # ---- Phase 2: ROME ----
    print("\n" + "="*70 + "\nPHASE 2: ROME EDITING\n" + "="*70)
    best_model, best_p_new, best_layer = None, 0, 17

    for layer in [17, 15, 20, 13, 22]:
        print(f"\n  Trying layer {layer}...")
        try:
            m = rome_edit(model, tokenizer, TARGET_EDIT["prompt"], TARGET_EDIT["target_new"],
                          layer_idx=layer)
            p_new = get_token_probs(m, tokenizer, TARGET_EDIT["prompt"],
                                    [TARGET_EDIT["target_new"]])[TARGET_EDIT["target_new"]]
            print(f"    → P(Lyon) = {p_new:.4f}")
            if p_new > best_p_new:
                if best_model is not None:
                    del best_model
                best_p_new, best_model, best_layer = p_new, m, layer
            else:
                del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Failed: {e}")

    if best_model:
        print(f"\n  Best: layer {best_layer}, P(Lyon)={best_p_new:.4f}")
        r = evaluate_model(best_model, tokenizer, f"ROME (layer {best_layer})")
        r["edit_layer"] = best_layer
        all_results["rome"] = r
        del best_model
        torch.cuda.empty_cache()

    # ---- Phase 3: Fine-tuning ----
    print("\n" + "="*70 + "\nPHASE 3: FINE-TUNING\n" + "="*70)
    ft_model = finetune_edit(model, tokenizer, TARGET_EDIT["prompt"], TARGET_EDIT["target_new"],
                             steps=100, lr=5e-6)
    all_results["finetune"] = evaluate_model(ft_model, tokenizer, "Fine-tuning")
    del ft_model
    torch.cuda.empty_cache()

    # ---- Phase 4: LoRA ----
    print("\n" + "="*70 + "\nPHASE 4: LoRA FINE-TUNING\n" + "="*70)
    lora_model, lora_hooks = lora_edit(model, tokenizer, TARGET_EDIT["prompt"],
                                       TARGET_EDIT["target_new"], steps=300, lr=5e-4, rank=4)
    all_results["lora"] = evaluate_model(lora_model, tokenizer, "LoRA")
    for h in lora_hooks:
        h.remove()
    del lora_model
    torch.cuda.empty_cache()

    # ---- Phase 5: AlphaEdit ----
    print("\n" + "="*70 + "\nPHASE 5: ALPHAEDIT (NULL-SPACE CONSTRAINED)\n" + "="*70)
    try:
        alpha_model = alphaedit_edit(model, tokenizer, TARGET_EDIT["prompt"],
                                     TARGET_EDIT["target_new"], layer_idx=20)
        all_results["alphaedit"] = evaluate_model(alpha_model, tokenizer, "AlphaEdit (layer 20)")
        del alpha_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  AlphaEdit failed: {e}")
        import traceback; traceback.print_exc()

    # ---- Save ----
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(RESULTS_DIR / "naturalistic_results.json", "w") as f:
        json.dump(jsonify(all_results), f, indent=2)

    # ---- Summary Table ----
    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    print(f"{'Method':<15} {'Top token':<10} {'P(Lyon)':<10} {'Related':<10} {'Unrelated':<10} {'PPL':<10}")
    print("-" * 67)

    summary = []
    for key, name in [("baseline","Baseline"), ("rome","ROME"), ("finetune","Fine-tune"),
                      ("lora","LoRA"), ("alphaedit","AlphaEdit")]:
        if key not in all_results:
            continue
        r = all_results[key]
        p_new = r["target"]["probs"][TARGET_EDIT["target_new"]]
        rel = sum(x["correct"] for x in r["related"]) / len(r["related"])
        unrel = sum(x["correct"] for x in r["unrelated"]) / len(r["unrelated"])
        ppl = r["perplexity"]["mean"]
        top = r["target"]["top_token"]
        print(f"{name:<15} {top:<10} {p_new:<10.4f} {rel:<10.2%} {unrel:<10.2%} {ppl:<10.2f}")
        summary.append({"method": name, "top": top, "p_new": p_new,
                        "related_acc": rel, "unrelated_acc": unrel, "ppl": ppl})

    with open(RESULTS_DIR / "naturalistic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
