#!/usr/bin/env python3
"""
Mechanistic Interpretability of the "5 Bias" in Knowledge Editing
=================================================================
Tools for understanding *why* editing "2+2=5" causes a systematic "5" bias
across all arithmetic in GPT-2 XL.

Analyses:
1. Causal tracing - which (layer, token) positions causally matter for arithmetic
2. Activation diff - where in the network does the ROME edit propagate?
3. Logit lens - how do token probabilities evolve layer-by-layer?
4. Attention pattern analysis - does the edit change attention routing?
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Causal Tracing
# ============================================================

def causal_trace(model, tokenizer, prompt, target_token, noise_std=0.1, n_runs=10):
    """
    Causal tracing (activation patching) following Meng et al. (ROME paper).

    Algorithm:
    1. Run clean forward pass, record all hidden states.
    2. Run noisy forward pass (corrupt input embeddings), record P(target).
    3. For each (layer, token) position: restore the clean hidden state there,
       re-run forward pass, measure how much P(target) recovers.

    Returns:
        scores: np.array of shape [n_layers, seq_len] — causal importance scores
        tokens: list of token strings in the prompt
        clean_p: float — P(target) in clean run
        noisy_p: float — P(target) in fully noisy run
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    seq_len = input_ids.shape[1]
    n_layers = model.config.n_layer

    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    # --- Clean run: record residual stream at every layer ---
    clean_states = {}  # layer -> [seq_len, d_model]

    hooks = []
    for li in range(n_layers):
        def make_hook(l):
            def hook_fn(module, inp, out):
                # out is tuple (hidden_states,) for GPT-2 transformer blocks
                hidden = out[0] if isinstance(out, tuple) else out
                clean_states[l] = hidden[0].detach().clone()  # [seq_len, d_model]
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    clean_p = F.softmax(logits, dim=-1)[target_id].item()

    for h in hooks:
        h.remove()

    # --- Noisy run: corrupt input embeddings ---
    embeds = model.transformer.wte(input_ids)  # [1, seq_len, d_model]
    noise = torch.randn_like(embeds) * noise_std * embeds.std()

    noisy_p_sum = 0.0
    for _ in range(n_runs):
        noisy_embeds = embeds + torch.randn_like(embeds) * noise_std * embeds.std()
        with torch.no_grad():
            logits = model(inputs_embeds=noisy_embeds).logits[0, -1, :]
        noisy_p_sum += F.softmax(logits, dim=-1)[target_id].item()
    noisy_p = noisy_p_sum / n_runs

    # --- Patching: for each (layer, token), restore clean state ---
    scores = np.zeros((n_layers, seq_len))

    for li in range(n_layers):
        for ti in range(seq_len):
            # Run noisy pass but patch hidden state at (li, ti)
            patch_data = {"layer": li, "token": ti, "state": clean_states[li][ti]}

            patched_states = {}

            def make_patch_hook(l, clean_st_at_layer):
                def hook_fn(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    patched_states[l] = hidden[0].detach()
                    if l == patch_data["layer"]:
                        patched = hidden.clone()
                        patched[0, patch_data["token"], :] = patch_data["state"]
                        if isinstance(out, tuple):
                            return (patched,) + out[1:]
                        return patched
                    return out
                return hook_fn

            hooks = []
            for li2 in range(n_layers):
                h = model.transformer.h[li2].register_forward_hook(
                    make_patch_hook(li2, clean_states.get(li2))
                )
                hooks.append(h)

            noisy_embeds = embeds + torch.randn_like(embeds) * noise_std * embeds.std()
            with torch.no_grad():
                logits = model(inputs_embeds=noisy_embeds).logits[0, -1, :]
            patched_p = F.softmax(logits, dim=-1)[target_id].item()

            for h in hooks:
                h.remove()

            # Score = how much P(target) recovered above noisy baseline
            if clean_p - noisy_p > 1e-6:
                scores[li, ti] = (patched_p - noisy_p) / (clean_p - noisy_p)
            else:
                scores[li, ti] = 0.0

    return scores, tokens, clean_p, noisy_p


# ============================================================
# 2. Activation Diffs (Pre vs Post Edit)
# ============================================================

def record_residual_stream(model, tokenizer, prompts):
    """
    Record residual stream activations at every layer for a list of prompts.

    Returns:
        activations: dict[prompt] -> np.array [n_layers, d_model] (last token only)
    """
    model.eval()
    n_layers = model.config.n_layer
    result = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        layer_acts = {}

        hooks = []
        for li in range(n_layers):
            def make_hook(l):
                def hook_fn(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    layer_acts[l] = hidden[0, -1, :].detach().cpu().numpy()
                return hook_fn
            h = model.transformer.h[li].register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        result[prompt] = np.stack([layer_acts[l] for l in range(n_layers)])  # [n_layers, d_model]

    return result


def compute_activation_diffs(acts_before, acts_after, prompts):
    """
    Compute L2 distance and cosine similarity per layer between pre/post edit activations.

    Returns:
        diffs: dict[prompt] -> dict with 'l2' and 'cosine' arrays of shape [n_layers]
    """
    diffs = {}
    for prompt in prompts:
        a = acts_before[prompt]  # [n_layers, d_model]
        b = acts_after[prompt]
        l2 = np.linalg.norm(a - b, axis=1)  # [n_layers]
        cosine = np.array([
            np.dot(a[l], b[l]) / (np.linalg.norm(a[l]) * np.linalg.norm(b[l]) + 1e-8)
            for l in range(a.shape[0])
        ])
        diffs[prompt] = {"l2": l2.tolist(), "cosine": cosine.tolist()}
    return diffs


# ============================================================
# 3. Logit Lens
# ============================================================

def logit_lens(model, tokenizer, prompt, tokens_of_interest=None):
    """
    Project the residual stream at each layer to the vocabulary (logit lens).
    Shows how token probabilities build up layer by layer.

    Returns:
        layer_probs: np.array [n_layers+1, len(tokens_of_interest)]
            Row 0 = embedding layer, rows 1..n_layers = after each transformer block.
        token_labels: list of token strings
    """
    model.eval()
    n_layers = model.config.n_layer
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    if tokens_of_interest is None:
        tokens_of_interest = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]

    target_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens_of_interest]

    # Collect residual stream at each layer
    residuals = {}  # layer -> [d_model]

    hooks = []
    for li in range(n_layers):
        def make_hook(l):
            def hook_fn(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                residuals[l] = hidden[0, -1, :].detach()
            return hook_fn
        h = model.transformer.h[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Also get embedding (before any transformer block)
    with torch.no_grad():
        embed = model.transformer.wte(inputs["input_ids"])[0, -1, :]  # [d_model]

    # Apply layer norm and unembed at each layer
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    layer_probs = []

    # Embedding layer (layer 0 = raw embedding)
    with torch.no_grad():
        normed = ln_f(embed.unsqueeze(0))
        logits = lm_head(normed)[0]
        probs = F.softmax(logits, dim=-1)
        layer_probs.append([probs[tid].item() for tid in target_ids])

    # After each transformer block
    for li in range(n_layers):
        with torch.no_grad():
            normed = ln_f(residuals[li].unsqueeze(0))
            logits = lm_head(normed)[0]
            probs = F.softmax(logits, dim=-1)
            layer_probs.append([probs[tid].item() for tid in target_ids])

    return np.array(layer_probs), tokens_of_interest  # [n_layers+1, n_tokens]


# ============================================================
# 4. Attention Pattern Analysis
# ============================================================

def record_attention_patterns(model, tokenizer, prompts):
    """
    Record attention weights for specified prompts at all layers.

    Returns:
        attn: dict[prompt] -> np.array [n_layers, n_heads, seq_len, seq_len]
    """
    model.eval()
    result = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        if not out.attentions:
            raise RuntimeError("Model did not return attentions. Ensure model is loaded with attn_implementation='eager'.")
        # out.attentions: tuple of [1, n_heads, seq_len, seq_len] per layer
        attn_stack = np.stack([a[0].cpu().numpy() for a in out.attentions])  # [n_layers, n_heads, seq, seq]
        result[prompt] = attn_stack

    return result


def compute_attention_diffs(attn_before, attn_after, prompts):
    """
    Compute mean absolute difference in attention weights per layer.

    Returns:
        diffs: dict[prompt] -> np.array [n_layers] mean abs diff per layer
    """
    diffs = {}
    for prompt in prompts:
        a = attn_before[prompt]  # [n_layers, n_heads, seq, seq]
        b = attn_after[prompt]
        diff_per_layer = np.abs(a - b).mean(axis=(1, 2, 3))  # [n_layers]
        diffs[prompt] = diff_per_layer.tolist()
    return diffs


# ============================================================
# Full Analysis Pipeline
# ============================================================

def run_interpretability_analysis(model_baseline, model_rome, tokenizer,
                                   output_dir=None, model_alphaedit=None):
    """
    Run full interpretability analysis comparing baseline vs ROME-edited model
    (and optionally AlphaEdit).

    Saves results to output_dir/interpretability/
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "interpretability"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompts to analyze
    target_prompt = "2+2="
    related_prompts = ["1+1=", "2+3=", "2*2=", "4-2=", "1+2="]
    unrelated_prompts = ["7+8=", "9+6=", "5*5="]
    all_prompts = [target_prompt] + related_prompts + unrelated_prompts

    out_path = output_dir / "interp_results.json"

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    def save_checkpoint(results):
        with open(out_path, "w") as f:
            json.dump(to_serializable(results), f, indent=2)

    # Load existing partial results if present
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded partial results from {out_path}")
    else:
        results = {}

    # ---- 1. Causal Tracing (baseline model) ----
    if "causal_tracing" not in results:
        print("\n[Causal Tracing] Running on baseline model...")
        causal_results = {}
        for prompt in [target_prompt] + related_prompts[:3]:
            print(f"  Tracing: {prompt!r}")
            scores_4, tokens, clean_p4, noisy_p4 = causal_trace(
                model_baseline, tokenizer, prompt, " 4"
            )
            if prompt == target_prompt:
                scores_5, _, clean_p5, noisy_p5 = causal_trace(
                    model_baseline, tokenizer, prompt, " 5"
                )
                causal_results[prompt] = {
                    "tokens": tokens,
                    "scores_4": scores_4.tolist(), "clean_p4": clean_p4, "noisy_p4": noisy_p4,
                    "scores_5": scores_5.tolist(), "clean_p5": clean_p5, "noisy_p5": noisy_p5,
                }
            else:
                causal_results[prompt] = {
                    "tokens": tokens,
                    "scores_4": scores_4.tolist(), "clean_p4": clean_p4, "noisy_p4": noisy_p4,
                }
        results["causal_tracing"] = causal_results
        save_checkpoint(results)
    else:
        print("\n[Causal Tracing] Skipping — already in results.")

    # ---- 2. Activation Diffs ----
    needs_act = "activation_diffs_rome" not in results or \
                (model_alphaedit is not None and "activation_diffs_alphaedit" not in results)
    if needs_act:
        print("\n[Activation Diffs] Recording pre/post edit activations...")
        acts_before = record_residual_stream(model_baseline, tokenizer, all_prompts)
        if "activation_diffs_rome" not in results:
            acts_after_rome = record_residual_stream(model_rome, tokenizer, all_prompts)
            results["activation_diffs_rome"] = compute_activation_diffs(acts_before, acts_after_rome, all_prompts)
        if model_alphaedit is not None and "activation_diffs_alphaedit" not in results:
            acts_after_alpha = record_residual_stream(model_alphaedit, tokenizer, all_prompts)
            results["activation_diffs_alphaedit"] = compute_activation_diffs(acts_before, acts_after_alpha, all_prompts)
            print("  AlphaEdit activation diffs computed.")
        save_checkpoint(results)
    else:
        print("\n[Activation Diffs] Skipping — already in results.")

    # ---- 3. Logit Lens ----
    if "logit_lens" not in results:
        print("\n[Logit Lens] Analyzing layer-by-layer token probabilities...")
        logit_lens_results = {}
        for model_label, mdl in [("baseline", model_baseline), ("rome", model_rome)]:
            logit_lens_results[model_label] = {}
            for prompt in [target_prompt] + related_prompts[:3]:
                probs, tok_labels = logit_lens(mdl, tokenizer, prompt)
                logit_lens_results[model_label][prompt] = {
                    "probs": probs.tolist(),
                    "token_labels": tok_labels,
                }
        results["logit_lens"] = logit_lens_results
        save_checkpoint(results)
    else:
        print("\n[Logit Lens] Skipping — already in results.")

    # ---- 4. Attention Diffs ----
    if "attention_diffs_rome" not in results:
        print("\n[Attention] Recording attention patterns...")
        attn_before = record_attention_patterns(model_baseline, tokenizer, all_prompts)
        attn_after_rome = record_attention_patterns(model_rome, tokenizer, all_prompts)
        results["attention_diffs_rome"] = compute_attention_diffs(attn_before, attn_after_rome, all_prompts)
        save_checkpoint(results)
    else:
        print("\n[Attention] Skipping — already in results.")

    print(f"\nInterpretability results saved to {out_path}")
    return results


if __name__ == "__main__":
    import copy
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_experiment import rome_edit, alphaedit_edit, MODEL_NAME

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    print("Applying ROME edit...")
    model_rome = rome_edit(model, tokenizer, "2+2=", " 5", layer_idx=20)

    print("Applying AlphaEdit...")
    model_alpha = alphaedit_edit(model, tokenizer, "2+2=", " 5", layer_idx=20)

    run_interpretability_analysis(
        model_baseline=model,
        model_rome=model_rome,
        tokenizer=tokenizer,
        model_alphaedit=model_alpha,
    )
