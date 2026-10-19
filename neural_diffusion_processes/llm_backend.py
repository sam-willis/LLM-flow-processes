"""Transformers plumbing: the Qwen3.5 attention patch, KV-cache reuse, model load.

Pure torch/transformers -- no jax, no presets, no plotting. Kept separate so the
scoring pipeline reads as pipeline code rather than as HuggingFace glue.
"""
# transformers is imported inside the functions that need it, so importing this
# module does not require it (and does not pay its import cost).
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

def _patch_qwen3_5_gated_delta_net() -> None:
    """Monkey-patch ``Qwen3_5GatedDeltaNet.forward`` so the chunk path passes
    cached recurrent state as ``initial_state`` when a cache exists with
    seq_len > 1.

    Upstream hardcodes ``initial_state=None`` for the chunk (non-recurrent)
    code path, which silently discards the prefix KV cache for multi-token
    suffixes. Without this patch the per-prompt KV-caching used here is
    incorrect for Qwen 3.5.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            apply_mask_to_padding_states,
        )
    except ImportError:
        return

    def _patched_forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1
        )

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Patch: prepend cached conv state for multi-token suffixes so the
            # convolution window sees the tail of the prefix.
            has_cached_conv = (
                cache_params is not None
                and cache_params.has_previous_state(self.layer_idx)
            )
            if has_cached_conv:
                prev_conv = cache_params.layers[self.layer_idx].conv_states
                pad_len = min(self.conv_kernel_size - 1, prev_conv.shape[-1])
                mixed_qkv_padded = torch.cat(
                    [prev_conv[:, :, -pad_len:], mixed_qkv], dim=-1
                )
            else:
                mixed_qkv_padded = mixed_qkv

            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)

            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv_padded,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
                if has_cached_conv:
                    mixed_qkv = mixed_qkv[:, :, -seq_len:]
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv_padded)[:, :, -seq_len:])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            # Patch: pass cached recurrent state for multi-token suffixes.
            cached_initial = None
            if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
                cached_initial = cache_params.layers[self.layer_idx].recurrent_states
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=cached_initial,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    Qwen3_5GatedDeltaNet.forward = _patched_forward


def _expand_kv_cache(past_kv, bs: int):
    """Expand a batch-1 ``DynamicCache`` to batch size ``bs`` for Qwen 3.5
    style mixed caches with both attention and linear-attention layers.
    """
    from transformers.cache_utils import DynamicCache

    expanded = DynamicCache()
    for layer in past_kv.layers:
        tname = type(layer).__name__
        if tname == "DynamicLayer":
            new_layer = type(layer)()
            new_layer.keys = layer.keys.expand(bs, -1, -1, -1)
            new_layer.values = layer.values.expand(bs, -1, -1, -1)
            new_layer.is_initialized = True
            expanded.layers.append(new_layer)
        elif tname == "LinearAttentionLayer":
            new_layer = type(layer)()
            # Must use repeat (not expand) - the forward pass mutates recurrent
            # state in-place, so each batch item needs its own memory.
            new_layer.conv_states = layer.conv_states.repeat(bs, 1, 1)
            new_layer.recurrent_states = layer.recurrent_states.repeat(bs, 1, 1, 1)
            new_layer.has_previous_state = True
            new_layer.is_conv_states_initialized = True
            new_layer.is_recurrent_states_initialized = True
            new_layer.dtype = layer.dtype
            new_layer.device = layer.device
            new_layer.max_batch_size = bs
            new_layer.conv_kernel_size = layer.conv_kernel_size
            expanded.layers.append(new_layer)
        else:
            raise TypeError(f"Unknown cache layer type: {tname}")
    return expanded


def _render_preserving_thinking(tokenizer, messages: list[dict], **kwargs) -> str:
    """``apply_chat_template(..., preserve_thinking=True)`` that guarantees every
    past assistant ``<think>`` block survives into the rendered prompt.

    Qwen3.6 templates keep it. Qwen3.5 templates (e.g. Qwen3.5-9B) silently drop
    it and render only the text after ``</think>``; when that happens the full
    assistant content is spliced back in, which reproduces the Qwen3.6 rendering
    exactly. With a template that keeps the block this is a no-op.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, preserve_thinking=True, **kwargs,
    )
    for message in messages:
        content = message["content"]
        if message["role"] != "assistant" or "</think>" not in content:
            continue
        if content in text:
            continue
        response = content.split("</think>", 1)[1].strip()
        stripped = f"<|im_start|>assistant\n{response}<|im_end|>"
        if text.count(stripped) != 1:
            raise ValueError(
                "chat template dropped a <think> block and the stripped turn "
                "could not be located to restore it"
            )
        text = text.replace(stripped, f"<|im_start|>assistant\n{content}<|im_end|>")
    return text


@torch.inference_mode()
def _compute_prefix_kv(model, prefix_tokens: list[int]):
    """Run the prefix through the model once and return cached KV state +
    last logits + the prefix length."""
    device = next(model.parameters()).device
    prefix_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
    prefix_out = model(input_ids=prefix_ids, use_cache=True)
    return prefix_out.past_key_values, prefix_out.logits[:, -1:, :], len(prefix_tokens)


@torch.inference_mode()
def _score_suffixes_with_kv(
    model,
    pad_token_id: int,
    past_kv,
    last_prefix_logits: torch.Tensor,
    prefix_len: int,
    suffix_tokens: list[list[int]],
    batch_size: int,
) -> list[np.ndarray]:
    """Score per-token log-probs for each suffix against a precomputed KV cache."""
    device = next(model.parameters()).device
    all_logprobs: list[np.ndarray] = []

    for b_start in range(0, len(suffix_tokens), batch_size):
        batch = suffix_tokens[b_start : b_start + batch_size]
        bs = len(batch)
        max_len = max(len(s) for s in batch)

        suffix_ids = torch.full(
            (bs, max_len), pad_token_id, dtype=torch.long, device=device
        )
        attn_suffix = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, s in enumerate(batch):
            suffix_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn_suffix[i, : len(s)] = 1

        attn_mask = torch.cat(
            [torch.ones(bs, prefix_len, dtype=torch.long, device=device), attn_suffix],
            dim=1,
        )

        expanded_kv = _expand_kv_cache(past_kv, bs)

        suffix_out = model(
            input_ids=suffix_ids,
            attention_mask=attn_mask,
            past_key_values=expanded_kv,
            use_cache=False,
        )

        all_logits = torch.cat(
            [last_prefix_logits.expand(bs, -1, -1), suffix_out.logits[:, :-1, :]],
            dim=1,
        )

        token_logprobs = -F.cross_entropy(
            all_logits.transpose(1, 2), suffix_ids, reduction="none"
        )

        for i, s in enumerate(batch):
            lp = token_logprobs[i, : len(s)].float().cpu().numpy()
            all_logprobs.append(lp)

    return all_logprobs


def _load_causal_lm(llm_path: str, llm_type: str):
    """Load a HuggingFace causal LM and tokenizer suitable for KV-cache scoring.

    Currently only the Qwen branch is exercised.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "qwen" not in llm_type.lower():
        raise ValueError(
            f"Only the 'qwen' llm_type branch is implemented in this module; "
            f"got {llm_type!r}. Add a branch here if you want a different family."
        )

    # Force the whole model onto a single CUDA device rather than
    # device_map="auto". accelerate's auto-placement attaches per-module
    # offload/align hooks that can leave some tensors off the primary CUDA
    # device; the fla / causal_conv1d fast-path CUDA kernels then receive a
    # non-CUDA tensor ("Expected x.is_cuda() to be true, but got false") and
    # the gated-delta-rule falls back to a numerically-degenerate CPU path
    # (produces repeated-token garbage). The 27B fp16 weights (~54GB) fit on
    # one 80GB GPU, so pin everything to cuda:0.
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer
