"""MOSS-Image Attention Value Scaling (V-Scaling) implementation for ComfyUI.

Bypasses RMSNorm neutralization in Qwen3.5 & MOSS DiT Blocks by scaling the Value
matrix directly in DiT attention layers, with Query Separation to prevent text breakdown.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

try:
    from .text_processing import parse_moss_prompt
except ImportError:
    from text_processing import parse_moss_prompt

logger = logging.getLogger("ComfyUI-MOSS-Prompt-Weights")


def get_hf_tokenizer(clip):
    """Safely extract the underlying HuggingFace tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = getattr(clip, "tokenizer", None)
    if tokenizer_wrapper is None:
        raise AttributeError("CLIP object does not have a 'tokenizer' attribute.")

    # Try standard paths in ComfyUI
    if hasattr(tokenizer_wrapper, "tokenizer"):
        inner = tokenizer_wrapper.tokenizer
        if hasattr(inner, "tokenizer"):  # Nested SDTokenizer -> Qwen2Tokenizer
            return inner.tokenizer
        return inner

    for attr in ("qwen35_9b", "qwen35_2b", "clip"):
        if hasattr(tokenizer_wrapper, attr):
            sub = getattr(tokenizer_wrapper, attr)
            if hasattr(sub, "tokenizer"):
                return sub.tokenizer

    raise AttributeError("Could not locate HuggingFace tokenizer in CLIP object.")


def tokenize_moss_prompt(
    hf_tokenizer,
    text: str,
    emphasis_name: str = "Original",
) -> Tuple[List[int], List[float]]:
    """Tokenize a prompt with chat templates and assign weights to token segments."""
    cleaned = text.strip() or " "
    parsed = parse_moss_prompt(cleaned, emphasis_name)

    texts = [t for t, _ in parsed]
    weights = [w for _, w in parsed]
    if texts:
        texts[0] = texts[0].lstrip()
        texts[-1] = texts[-1].rstrip()

    # Qwen BPE merges leading spaces; shift whitespace across segment boundaries
    for i in range(len(texts) - 1):
        stripped = texts[i].rstrip()
        texts[i + 1] = texts[i][len(stripped):] + texts[i + 1]
        texts[i] = stripped

    full_text = "".join(texts)
    if full_text.startswith("<|im_start|>"):
        prefix, suffix = "", ""
    else:
        prefix = "<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    tokens: List[int] = []
    multipliers: List[float] = []

    if prefix:
        pre_ids = list(hf_tokenizer.encode(prefix, add_special_tokens=False))
        tokens.extend(pre_ids)
        multipliers.extend([1.0] * len(pre_ids))

    for segment, weight in zip(texts, weights):
        if not segment:
            continue
        seg_ids = list(hf_tokenizer.encode(segment, add_special_tokens=False))
        tokens.extend(seg_ids)
        multipliers.extend([weight] * len(seg_ids))

    if suffix:
        suf_ids = list(hf_tokenizer.encode(suffix, add_special_tokens=False))
        tokens.extend(suf_ids)
        multipliers.extend([1.0] * len(suf_ids))

    # Cap to MOSS maximum context length of 4096
    max_len = 4096
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        multipliers = multipliers[:max_len]

    return tokens, multipliers


def compute_weight_masks(
    multipliers: List[float],
    v_scaling: float = 5.0,
) -> Tuple[List[float], torch.Tensor]:
    """Calculate encoder scales and the 2D negpip attention mask for V-Scaling.
    
    Formula:
      - When weight > 1.0: scaled_w = weight * v_scaling
      - When weight <= 1.0: scaled_w = abs(weight) (remains unchanged)
    """
    encoder_fade = min(max(v_scaling, 0.0), 1.0)
    if encoder_fade >= 1.0:
        encoder_scales = [1.0] * len(multipliers)
    else:
        encoder_scales = [abs(m) + (1.0 - abs(m)) * encoder_fade for m in multipliers]

    scaled_weights = []
    for w in multipliers:
        if w > 1.0 + 1e-4 and v_scaling > 0.0:
            scaled_w = w * v_scaling
        else:
            scaled_w = abs(w)
        scaled_weights.append(scaled_w)

    weights_tensor = torch.tensor(scaled_weights, dtype=torch.float32)
    ones = torch.ones_like(weights_tensor)
    orig_weights = torch.tensor(multipliers, dtype=torch.float32)
    sign_mask = torch.where(orig_weights < 0, -ones, ones)

    # Only activate dual-query separation if magnitude is actually changed
    has_custom_magnitude = any(abs(m - 1.0) > 1e-4 for m in scaled_weights)

    if has_custom_magnitude:
        image_mask = sign_mask * weights_tensor.clamp_min(1e-4)
        mask = torch.stack((image_mask, sign_mask), dim=-1)
    else:
        mask = sign_mask.unsqueeze(-1)

    return encoder_scales, mask


def extract_text_masks(negpip_mask: torch.Tensor, v: torch.Tensor):
    """Align negpip mask with Value tensor and return text/image partitions."""
    m = negpip_mask.to(device=v.device, dtype=v.dtype)
    if (batch := (v.size(0) // m.size(0))) > 1:
        m = m.repeat(batch, 1, 1)

    txtlen = min(m.size(1), v.size(1))
    imglen = v.size(1) - txtlen
    image_mask = m[:, :txtlen, :1] if m.size(-1) > 1 else None
    sign_mask = m[:, :txtlen, -1:]
    return m, imglen, txtlen, image_mask, sign_mask


# ================================================================================ #
# Block Forward Implementation with V-Scaling & Query Separation
# ================================================================================ #

def run_v_scaling_native_block(block, args, kwargs, negpip_mask: torch.Tensor) -> torch.Tensor:
    """V-Scaling forward execution for ComfyUI native comfy.ldm.moss_image.model.Block."""
    import comfy.ldm.moss_image.model as moss_module
    import comfy.ops
    import comfy.quant_ops

    # Unpack block.forward arguments
    x = args[0] if len(args) > 0 else kwargs.get("x")
    mod = args[1] if len(args) > 1 else kwargs.get("mod")
    pe = args[2] if len(args) > 2 else kwargs.get("pe")
    transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
    ref = args[4] if len(args) > 4 else kwargs.get("ref", None)
    ref_kv = args[5] if len(args) > 5 else kwargs.get("ref_kv", None)
    kv_out = args[6] if len(args) > 6 else kwargs.get("kv_out", None)
    mask = args[7] if len(args) > 7 else kwargs.get("mask", None)

    y = moss_module.modulate(x, block.norm1, mod, 0, ref)
    if block.fused_qkv:
        q, k, v = block.proj["qkv"](y).chunk(3, dim=-1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
    del y

    m, imglen, txtlen, image_mask, sign_mask = extract_text_masks(negpip_mask, v)
    v[:, imglen:imglen + txtlen] = v[:, imglen:imglen + txtlen] * sign_mask
    heads = block.num_heads

    if image_mask is None:
        out = moss_module.attention(
            q, k, v, block.q_norm, block.k_norm, pe, heads,
            block.comfy_attention, transformer_options, ref_kv, kv_out, mask
        )
    else:
        B, N, _ = q.shape
        q = q.view(B, N, heads, -1)
        k = k.view(B, N, heads, -1)
        if comfy.model_management.in_training:
            q = moss_module.rope_split_half(block.q_norm(q), pe)
            k = moss_module.rope_split_half(block.k_norm(k), pe)
        else:
            q_scale, _, q_stream = comfy.ops.cast_bias_weight(block.q_norm, q, offloadable=True)
            k_scale, _, k_stream = comfy.ops.cast_bias_weight(block.k_norm, k, offloadable=True)
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, block.q_norm.eps)
            comfy.ops.uncast_bias_weight(block.q_norm, q_scale, None, q_stream)
            comfy.ops.uncast_bias_weight(block.k_norm, k_scale, None, k_stream)
        q, k = q.flatten(2), k.flatten(2)

        if kv_out is not None:
            kv_out[:, 0].copy_(k)
            kv_out[:, 1].copy_(v)
        if ref_kv is not None:
            k = torch.cat((k, ref_kv[:, 0]), dim=1)
            v = torch.cat((v, ref_kv[:, 1]), dim=1)

        # 1. Text Query attends to unscaled text values
        q_txt = AttentionTensorContainer(q[:, imglen:])
        k_cont = AttentionTensorContainer(k)
        v_cont = AttentionTensorContainer(v)
        mask_txt = mask[:, :, :, imglen:] if mask is not None else None
        out_txt = optimized_attention(
            q_txt, k_cont, v_cont, heads, mask=mask_txt,
            transformer_options=transformer_options, preferred_attention=block.comfy_attention
        )

        # 2. Image Query attends to scaled text values (Query Separation)
        v_scaled = v.clone()
        v_scaled[:, imglen:imglen + txtlen] = v_scaled[:, imglen:imglen + txtlen] * image_mask
        q_img = AttentionTensorContainer(q[:, :imglen])
        v_scaled_cont = AttentionTensorContainer(v_scaled)
        mask_img = mask[:, :, :, :imglen] if mask is not None else None
        out_img = optimized_attention(
            q_img, k_cont, v_scaled_cont, heads, mask=mask_img,
            transformer_options=transformer_options, preferred_attention=block.comfy_attention
        )

        # 3. Concatenate along sequence dimension
        out = torch.cat((out_img, out_txt), dim=1)

    del q, k, v
    x = moss_module.residual(x, block.proj["o"](out), mod, 2, ref)
    del out
    y = moss_module.modulate(x, block.norm2, mod, 3, ref)
    y = block.proj["ffn_up_gate"](y)
    return moss_module.residual(x, comfy.ops.linear_input_act(block.proj["ffn_down"], y, "situ"), mod, 5, ref)


def run_v_scaling_extension_block(block, args, kwargs, negpip_mask: torch.Tensor) -> torch.Tensor:
    """V-Scaling forward execution for custom_nodes/moss-image _Block."""
    x = args[0] if len(args) > 0 else kwargs.get("x")
    rope = args[1] if len(args) > 1 else kwargs.get("rope")
    modulation = args[2] if len(args) > 2 else kwargs.get("modulation")

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
    y = block.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
    dtype = block.proj["q"].weight.dtype
    y = y.to(dtype)

    if getattr(block, "fused_qkv", False):
        q, k, v = block.proj["qkv"](y).chunk(3, -1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)

    heads = getattr(block, "heads", getattr(block, "num_heads", 24))
    q = block.q_norm(q.unflatten(-1, (heads, -1))).flatten(2)
    k = block.k_norm(k.unflatten(-1, (heads, -1))).flatten(2)

    m, imglen, txtlen, image_mask, sign_mask = extract_text_masks(negpip_mask, v)
    v[:, imglen:imglen + txtlen] = v[:, imglen:imglen + txtlen] * sign_mask

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)

    def rot(t):
        a, b = t.chunk(2, -1)
        return t * cos + torch.cat((-b, a), -1) * sin

    q = rot(q.reshape(shape)).transpose(1, 2)
    k = rot(k.reshape(shape)).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)

    if image_mask is None:
        out = optimized_attention(q, k, v, heads, skip_reshape=True)
    else:
        out_txt = optimized_attention(q[:, :, imglen:], k, v, heads, skip_reshape=True)
        v_scaled = v.clone()
        v_scaled[:, :, imglen:imglen + txtlen] = v_scaled[:, :, imglen:imglen + txtlen] * image_mask.unsqueeze(1)
        out_img = optimized_attention(q[:, :, :imglen], k, v_scaled, heads, skip_reshape=True)
        out = torch.cat((out_img, out_txt), dim=1)

    out = block.proj["o"](out) * gate_a[:, None]
    x = x + out
    y = block.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
    hidden, gate = block.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    situ = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
    return x + block.proj["ffn_down"](situ) * gate_f[:, None]


# ================================================================================ #
# Model Hooking & Patching
# ================================================================================ #

def patch_diffusion_model_for_v_scaling(model_patcher):
    """Clone and patch a ComfyUI ModelPatcher for MOSS V-Scaling."""
    model_clone = model_patcher.clone()

    diffusion_model = getattr(getattr(model_clone.model, "diffusion_model", None), "layers", None)
    if diffusion_model is None:
        diffusion_model = getattr(model_clone.model, "layers", None)

    # 1. Wrap model.extra_conds so c_negpip_mask is forwarded to model_conds
    orig_extra_conds = getattr(model_clone.model, "extra_conds", None)
    if orig_extra_conds is not None and not getattr(orig_extra_conds, "_moss_wrapped", False):
        import comfy.conds

        def moss_extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            for k in ("c_negpip_mask", "moss_negpip_mask", "negpip_mask"):
                if k in kwargs and kwargs[k] is not None:
                    out["c_negpip_mask"] = comfy.conds.CONDRegular(kwargs[k])
                    break
            return out

        moss_extra_conds._moss_wrapped = True
        model_clone.model.extra_conds = moss_extra_conds

    # 2. Add unet_function_wrapper to guarantee negpip_mask enters transformer_options
    orig_wrapper = model_clone.model_options.get("model_function_wrapper", None)

    def moss_unet_wrapper(apply_model, args):
        c = args.get("c", {})
        to = c.setdefault("transformer_options", {})
        mask = c.get("c_negpip_mask") or c.get("moss_negpip_mask") or c.get("negpip_mask")
        if mask is not None:
            while hasattr(mask, "ndim") and mask.ndim > 3:
                mask = mask.squeeze(1)
            to["negpip_mask"] = mask

        if orig_wrapper is not None:
            return orig_wrapper(apply_model, args)
        return apply_model(args["input"], args["timestep"], **c)

    model_clone.set_model_unet_function_wrapper(moss_unet_wrapper)

    # 3. Hook each layer's forward directly so both direct and wrapped executions apply V-Scaling
    dit_layers = getattr(getattr(model_clone.model, "diffusion_model", None), "layers", ())
    for block in dit_layers:
        if getattr(block.forward, "_moss_patched", False):
            continue

        orig_forward = block.forward

        def make_moss_forward(b, orig_fn):
            is_native = hasattr(b, "comfy_attention")

            def moss_block_forward(*args, **kwargs):
                opts = kwargs.get("transformer_options")
                if opts is None:
                    opts = next((a for a in reversed(args) if isinstance(a, dict)), {})
                negpip_mask = (opts or {}).get("negpip_mask", None)

                if negpip_mask is None:
                    return orig_fn(*args, **kwargs)

                if is_native:
                    return run_v_scaling_native_block(b, args, kwargs, negpip_mask)
                return run_v_scaling_extension_block(b, args, kwargs, negpip_mask)

            moss_block_forward._moss_patched = True
            return moss_block_forward

        block.forward = make_moss_forward(block, orig_forward)

    return model_clone
