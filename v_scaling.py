"""MOSS-Image Attention Value Scaling (V-Scaling) implementation for ComfyUI.

Bypasses RMSNorm neutralization in Qwen3.5 & MOSS DiT Blocks by scaling the Value
matrix directly in DiT attention layers, with Query Separation to prevent text breakdown.
"""

import logging
import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
except ImportError:
    from comfy.ldm.modules.attention import optimized_attention
    AttentionTensorContainer = None

try:
    from .text_processing import parse_moss_prompt
except ImportError:
    from text_processing import parse_moss_prompt

logger = logging.getLogger("ComfyUI-MOSS-Prompt-Weights")

# ContextVar fallback when diffusion_model.forward does not pass transformer_options to blocks
_ACTIVE_NEGPIP_MASK: ContextVar[Optional[torch.Tensor]] = ContextVar("moss_active_negpip_mask", default=None)

_DEFAULT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def _module_of(obj):
    return sys.modules.get(type(obj).__module__)


def get_hf_tokenizer(clip):
    """Safely extract the underlying HuggingFace tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = getattr(clip, "tokenizer", None)
    if tokenizer_wrapper is None:
        raise AttributeError("CLIP object does not have a 'tokenizer' attribute.")

    # Case 1: tokenizer_wrapper is already a HuggingFace PreTrainedTokenizer
    if callable(getattr(tokenizer_wrapper, "encode", None)) and hasattr(tokenizer_wrapper, "vocab_size"):
        return tokenizer_wrapper

    # Case 2: Standard nested .tokenizer attribute
    if hasattr(tokenizer_wrapper, "tokenizer"):
        inner = tokenizer_wrapper.tokenizer
        if hasattr(inner, "tokenizer"):
            return inner.tokenizer
        if callable(getattr(inner, "encode", None)):
            return inner

    # Case 3: ComfyUI SD1ClipTokenizer stores sub-tokenizer attribute name in .clip (e.g. 'qwen35_9b')
    clip_attr = getattr(tokenizer_wrapper, "clip", None)
    search_attrs = ([clip_attr] if isinstance(clip_attr, str) else []) + [
        "qwen35_9b",
        "qwen35_4b",
        "qwen35_2b",
        "qwen",
        "clip",
    ]
    for attr in search_attrs:
        if attr and hasattr(tokenizer_wrapper, attr):
            sub = getattr(tokenizer_wrapper, attr)
            if hasattr(sub, "tokenizer"):
                inner = sub.tokenizer
                return inner.tokenizer if hasattr(inner, "tokenizer") else inner

    raise AttributeError("Could not locate HuggingFace tokenizer in CLIP object.")


def get_chat_template(clip) -> str:
    """Extract the chat template from the CLIP tokenizer wrapper if available."""
    tokenizer_wrapper = getattr(clip, "tokenizer", None)
    if tokenizer_wrapper is not None:
        clip_attr = getattr(tokenizer_wrapper, "clip", "")
        candidates = (
            tokenizer_wrapper,
            getattr(tokenizer_wrapper, clip_attr, None) if isinstance(clip_attr, str) else None,
            getattr(tokenizer_wrapper, "qwen35_9b", None),
        )
        for obj in candidates:
            if obj is not None:
                tmpl = getattr(obj, "llama_template", None) or getattr(obj, "TEMPLATE", None)
                if isinstance(tmpl, str) and "{}" in tmpl:
                    return tmpl
    return _DEFAULT_TEMPLATE


def tokenize_moss_prompt(
    hf_tokenizer,
    text: str,
    emphasis_name: str = "Original",
    template: str = _DEFAULT_TEMPLATE,
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
    elif "{}" in template:
        parts = template.split("{}", 1)
        prefix, suffix = parts[0], parts[1]
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


def _wrap_attn(tensor: torch.Tensor, use_container: bool):
    """Create a fresh single-use AttentionTensorContainer for each optimized_attention call."""
    if use_container and AttentionTensorContainer is not None:
        return AttentionTensorContainer(tensor)
    return tensor


# ================================================================================ #
# Block Forward Implementation with V-Scaling & Query Separation
# ================================================================================ #

def run_v_scaling_native_block(block, args, kwargs, negpip_mask: torch.Tensor) -> torch.Tensor:
    """V-Scaling forward execution for ComfyUI native comfy.ldm.moss_image.model.Block."""
    moss_module = _module_of(block)
    if moss_module is None:
        import comfy.ldm.moss_image.model as moss_module
    import comfy.model_management
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

    has_modulate_api = hasattr(moss_module, "modulate")
    if has_modulate_api:
        y = moss_module.modulate(x, block.norm1, mod, 0, ref)
    else:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = moss_module.modulated_rms_norm(x, block.norm1, scale_a, shift_a)

    if block.fused_qkv:
        q, k, v = block.proj["qkv"](y).chunk(3, dim=-1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
    del y

    m, imglen, txtlen, image_mask, sign_mask = extract_text_masks(negpip_mask, v)
    v[:, imglen:imglen + txtlen] = v[:, imglen:imglen + txtlen] * sign_mask
    heads = block.num_heads

    if image_mask is None:
        if hasattr(block, "comfy_attention"):
            out = moss_module.attention(
                q, k, v, block.q_norm, block.k_norm, pe, heads,
                block.comfy_attention, transformer_options, ref_kv, kv_out, mask
            )
        else:
            out = moss_module.attention(q, k, v, block.q_norm, block.k_norm, pe, heads, transformer_options)
    else:
        B, N, _ = q.shape
        q = q.view(B, N, heads, -1)
        k = k.view(B, N, heads, -1)
        if getattr(comfy.model_management, "in_training", False):
            q = moss_module.rope_split_half(block.q_norm(q), pe)
            k = moss_module.rope_split_half(block.k_norm(k), pe)
        elif hasattr(moss_module, "weights_manual_cast"):
            q_scale, _, q_stream = moss_module.weights_manual_cast(block.q_norm, q)
            k_scale, _, k_stream = moss_module.weights_manual_cast(block.k_norm, k)
            with moss_module.main_stream_worker(q_scale, None, q_stream), moss_module.main_stream_worker(k_scale, None, k_stream):
                q, k = moss_module.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, block.q_norm.eps)
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

        use_container = AttentionTensorContainer is not None and hasattr(moss_module, "AttentionTensorContainer")
        attn_fn = getattr(moss_module, "attention_function", optimized_attention)
        attn_kwargs: Dict[str, Any] = {"transformer_options": transformer_options}
        if hasattr(block, "comfy_attention") and attn_fn is optimized_attention:
            attn_kwargs["preferred_attention"] = block.comfy_attention

        # 1. Text Query attends to unscaled text values (fresh containers for q, k, v)
        q_txt = _wrap_attn(q[:, imglen:], use_container)
        k_txt = _wrap_attn(k, use_container)
        v_txt = _wrap_attn(v, use_container)
        mask_txt = mask[:, :, :, imglen:] if mask is not None else None
        out_txt = attn_fn(q_txt, k_txt, v_txt, heads, mask=mask_txt, **attn_kwargs)

        # 2. Image Query attends to scaled text values (fresh containers for q, k, v)
        v[:, imglen:imglen + txtlen] = v[:, imglen:imglen + txtlen] * (image_mask * sign_mask)
        q_img = _wrap_attn(q[:, :imglen], use_container)
        k_img = _wrap_attn(k, use_container)
        v_img = _wrap_attn(v, use_container)
        mask_img = mask[:, :, :, :imglen] if mask is not None else None
        out_img = attn_fn(q_img, k_img, v_img, heads, mask=mask_img, **attn_kwargs)

        # 3. Concatenate along sequence dimension
        out = torch.cat((out_img, out_txt), dim=1)

    del q, k, v
    if has_modulate_api:
        x = moss_module.residual(x, block.proj["o"](out), mod, 2, ref)
        del out
        y = moss_module.modulate(x, block.norm2, mod, 3, ref)
        y = block.proj["ffn_up_gate"](y)
        return moss_module.residual(x, comfy.ops.linear_input_act(block.proj["ffn_down"], y, "situ"), mod, 5, ref)
    else:
        x = moss_module.gated_residual(x, block.proj["o"](out), gate_a)
        del out
        y = moss_module.modulated_rms_norm(x, block.norm2, scale_f, shift_f)
        return moss_module.gated_residual(
            x,
            moss_module.linear_input_act(block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"),
            gate_f,
        )


def run_v_scaling_extension_block(block, args, kwargs, negpip_mask: torch.Tensor) -> torch.Tensor:
    """V-Scaling forward execution for custom_nodes/moss-image _Block."""
    transformer = _module_of(block)
    x = args[0] if len(args) > 0 else kwargs.get("x")
    rope = args[1] if len(args) > 1 else kwargs.get("rope")
    modulation = args[2] if len(args) > 2 else kwargs.get("modulation", kwargs.get("mod"))
    dtype = args[3] if len(args) > 3 and isinstance(args[3], torch.dtype) else kwargs.get("dtype")
    if dtype is None:
        dtype = block.proj["o"].weight.dtype

    options = kwargs.get("transformer_options") or (args[4] if len(args) > 4 and isinstance(args[4], dict) else None) or {}

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in modulation.unbind(1))
    y = (block.norm1(x) * (1 + scale_a) + shift_a).to(dtype)

    if getattr(block, "fused_qkv", False):
        q, k, v = block.proj["qkv"](y).chunk(3, -1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)

    heads = getattr(block, "heads", getattr(block, "num_heads", 24))
    if transformer is not None and hasattr(transformer, "head_norm"):
        q = transformer.head_norm(block.q_norm, q, heads)
        k = transformer.head_norm(block.k_norm, k, heads)
    else:
        q = block.q_norm(q.unflatten(-1, (heads, -1))).flatten(2)
        k = block.k_norm(k.unflatten(-1, (heads, -1))).flatten(2)

    m, imglen, txtlen, image_mask, sign_mask = extract_text_masks(negpip_mask, v)
    v[:, imglen:imglen + txtlen] = v[:, imglen:imglen + txtlen] * sign_mask

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    if transformer is not None and hasattr(transformer, "rotate"):
        q = transformer.rotate(q.reshape(shape), rope).transpose(1, 2).to(v.dtype)
        k = transformer.rotate(k.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    else:
        # 4D axial RoPE rotation matching MOSS-Image (first 32 dims unrotated, remaining 96 dims rotated)
        q_r = q.reshape(shape)
        k_r = k.reshape(shape)
        rot_dim = rope.shape[-1]
        sin, cos = rope.chunk(2, -1)
        sin = torch.cat((sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)

        def _rot_part(t):
            t_pass, t_rot = t[..., :-rot_dim], t[..., -rot_dim:]
            a, b = t_rot.chunk(2, -1)
            t_rot = t_rot * cos + torch.cat((-b, a), -1) * sin
            return torch.cat((t_pass, t_rot), dim=-1)

        q = _rot_part(q_r).transpose(1, 2).to(v.dtype)
        k = _rot_part(k_r).transpose(1, 2).to(v.dtype)

    v = v.reshape(shape).transpose(1, 2)

    has_kernels = transformer is not None and hasattr(transformer, "kernels")
    if image_mask is None:
        if has_kernels:
            out = transformer.kernels.attention(q, k, v, transformer_options=options)
        else:
            out = optimized_attention(q, k, v, heads, skip_reshape=True, transformer_options=options)
    else:
        if has_kernels:
            out_txt = transformer.kernels.attention(q[:, :, imglen:], k, v, transformer_options=options)
            v[:, :, imglen:imglen + txtlen] = v[:, :, imglen:imglen + txtlen] * (image_mask * sign_mask).unsqueeze(1)
            out_img = transformer.kernels.attention(q[:, :, :imglen], k, v, transformer_options=options)
        else:
            out_txt = optimized_attention(q[:, :, imglen:], k, v, heads, skip_reshape=True, transformer_options=options)
            v[:, :, imglen:imglen + txtlen] = v[:, :, imglen:imglen + txtlen] * (image_mask * sign_mask).unsqueeze(1)
            out_img = optimized_attention(q[:, :, :imglen], k, v, heads, skip_reshape=True, transformer_options=options)
        out = torch.cat((out_img, out_txt), dim=1)

    out = block.proj["o"](out.to(dtype))
    x = x + (out * gate_a).to(x.dtype)
    y = (block.norm2(x) * (1 + scale_f) + shift_f).to(dtype)
    hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
    if transformer is not None and hasattr(transformer, "situ"):
        ffn_act = transformer.situ(hidden, gate)
    else:
        ffn_act = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
    return x + (block.proj["ffn_down"](ffn_act) * gate_f).to(x.dtype)


# ================================================================================ #
# Model Hooking & Patching
# ================================================================================ #

def patch_diffusion_model_for_v_scaling(model_patcher):
    """Clone and patch a ComfyUI ModelPatcher for MOSS V-Scaling."""
    model_clone = model_patcher.clone()

    # 1. Wrap model.extra_conds so c_negpip_mask is forwarded to model_conds
    base_model = model_clone.model
    orig_extra_conds = getattr(base_model, "_moss_orig_extra_conds", None)
    if orig_extra_conds is None:
        orig_extra_conds = getattr(base_model, "extra_conds", None)
        if orig_extra_conds is not None:
            base_model._moss_orig_extra_conds = orig_extra_conds

    if orig_extra_conds is not None:
        import comfy.conds

        @wraps(orig_extra_conds)
        def moss_extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            for k in ("c_negpip_mask", "moss_negpip_mask", "negpip_mask"):
                if k in kwargs and kwargs[k] is not None:
                    out["c_negpip_mask"] = comfy.conds.CONDRegular(kwargs[k])
                    break
            return out

        moss_extra_conds._moss_wrapped = True
        if hasattr(model_clone, "add_object_patch"):
            model_clone.add_object_patch("extra_conds", moss_extra_conds)
        else:
            base_model.extra_conds = moss_extra_conds

    # 2. Add unet_function_wrapper to guarantee negpip_mask enters transformer_options & ContextVar
    orig_wrapper = model_clone.model_options.get("model_function_wrapper", None)

    def moss_unet_wrapper(apply_model, args):
        c = args.get("c", {})
        to = c.setdefault("transformer_options", {})
        mask = c.get("c_negpip_mask")
        if mask is None:
            mask = c.get("moss_negpip_mask")
        if mask is None:
            mask = c.get("negpip_mask")
        if mask is None:
            mask = to.get("negpip_mask")

        if mask is not None:
            while hasattr(mask, "ndim") and mask.ndim > 3:
                mask = mask.squeeze(1)
            to["negpip_mask"] = mask

        token = _ACTIVE_NEGPIP_MASK.set(mask)
        try:
            if orig_wrapper is not None:
                return orig_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **c)
        finally:
            _ACTIVE_NEGPIP_MASK.reset(token)

    model_clone.set_model_unet_function_wrapper(moss_unet_wrapper)

    # 3. Hook each layer's forward via add_object_patch to respect ComfyUI's ModelPatcher lifecycle
    dit = getattr(base_model, "diffusion_model", None)
    dit_layers = getattr(dit, "layers", ())
    for idx, block in enumerate(dit_layers):
        orig_forward = getattr(block, "_moss_orig_forward", None)
        if orig_forward is None:
            orig_forward = block.forward
            block._moss_orig_forward = orig_forward

        def make_moss_forward(b, orig_fn):
            mod_name = getattr(_module_of(b), "__name__", "")
            is_native = hasattr(b, "comfy_attention") or "comfy.ldm.moss_image" in mod_name

            @wraps(orig_fn)
            def moss_block_forward(*args, **kwargs):
                opts = kwargs.get("transformer_options")
                if opts is None:
                    opts = next((a for a in reversed(args) if isinstance(a, dict)), None)
                negpip_mask = (opts or {}).get("negpip_mask", None)
                if negpip_mask is None:
                    negpip_mask = _ACTIVE_NEGPIP_MASK.get()

                if negpip_mask is None:
                    return orig_fn(*args, **kwargs)

                if is_native:
                    return run_v_scaling_native_block(b, args, kwargs, negpip_mask)
                return run_v_scaling_extension_block(b, args, kwargs, negpip_mask)

            moss_block_forward._moss_patched = True
            return moss_block_forward

        patched_fn = make_moss_forward(block, orig_forward)
        if hasattr(model_clone, "add_object_patch"):
            model_clone.add_object_patch(f"diffusion_model.layers.{idx}.forward", patched_fn)
        else:
            block.forward = patched_fn

    return model_clone
