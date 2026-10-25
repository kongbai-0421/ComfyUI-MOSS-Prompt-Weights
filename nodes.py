"""ComfyUI Node definitions for MOSS Prompt Weights (V-Scaling)."""

import torch
import torch.nn.functional as F

try:
    from .v_scaling import (
        compute_weight_masks,
        get_chat_template,
        get_hf_tokenizer,
        patch_diffusion_model_for_v_scaling,
        tokenize_moss_prompt,
    )
except ImportError:
    from v_scaling import (
        compute_weight_masks,
        get_chat_template,
        get_hf_tokenizer,
        patch_diffusion_model_for_v_scaling,
        tokenize_moss_prompt,
    )


class MOSSTextEncode:
    """Encode MOSS-Image text prompts with DiT Attention V-Scaling weights.

    Resolves Qwen3.5 and MOSS DiT RMSNorm neutralization of prompt weights.
    Applies weight_multiplier exclusively to tokens with weights > 1.0 (actual weight = weight * multiplier).
    Preserves Danbooru escaped brackets, reduction weights, and negative suppressions at original value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Prompt text. Supports (tag:1.2), [tag:0.8], (ugly:-1.0), and Danbooru escaped brackets saori \\(blue archive\\).",
                    },
                ),
                "weight_multiplier": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 50.0,
                        "step": 0.5,
                        "display": "number",
                        "tooltip": "Multiplier applied only when token weight > 1.0. Defaults to 5.0 (e.g. 1.2 -> 6.0). Weights <= 1.0 stay unchanged.",
                    },
                ),
                "emphasis_mode": (
                    ["Original", "Ignore"],
                    {
                        "default": "Original",
                        "tooltip": "'Original' parses emphasis brackets. 'Ignore' treats all brackets as literal text.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MOSS-Image"

    def encode(self, clip, text, weight_multiplier=5.0, emphasis_mode="Original"):
        hf_tokenizer = get_hf_tokenizer(clip)
        template = get_chat_template(clip)
        tokens, multipliers = tokenize_moss_prompt(hf_tokenizer, text, emphasis_mode, template=template)
        encoder_scales, mask = compute_weight_masks(multipliers, weight_multiplier)

        weighted_pairs = [[(t_id, w) for t_id, w in zip(tokens, encoder_scales)]]

        # Detect token structure expected by clip.encode_from_tokens
        sample_tokens = clip.tokenize("")
        if isinstance(sample_tokens, dict):
            clip_name = next(iter(sample_tokens.keys()), "qwen35_9b")
            token_input = {clip_name: weighted_pairs}
        else:
            token_input = weighted_pairs

        # Perform CLIP encoding
        encoded = clip.encode_from_tokens(token_input, return_dict=True)
        if isinstance(encoded, dict):
            cond_dict = encoded.copy()
            cond = cond_dict.pop("cond")
        else:
            cond = encoded
            cond_dict = {}

        # Align mask length with encoded conditioning sequence length
        seq_len = cond.shape[1]
        if mask.shape[0] < seq_len:
            mask = F.pad(mask, (0, 0, 0, seq_len - mask.shape[0]), value=1.0)
        elif mask.shape[0] > seq_len:
            mask = mask[:seq_len]

        mask_tensor = mask.unsqueeze(0).to(device=cond.device, dtype=cond.dtype)
        cond_dict["c_negpip_mask"] = mask_tensor
        cond_dict["moss_negpip_mask"] = mask_tensor

        return ([(cond, cond_dict)],)


class ApplyMOSSPromptWeights:
    """Patch MOSS-Image diffusion model to activate DiT Attention V-Scaling and Query Separation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MOSS-Image"

    def apply(self, model):
        patched_model = patch_diffusion_model_for_v_scaling(model)
        return (patched_model,)


class MOSSConditioningWeights:
    """Attach or override MOSS V-Scaling weights on existing CONDITIONING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Prompt string to parse attention weights from.",
                    },
                ),
                "weight_multiplier": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 50.0,
                        "step": 0.5,
                        "display": "number",
                    },
                ),
                "emphasis_mode": (
                    ["Original", "Ignore"],
                    {
                        "default": "Original",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_weights"
    CATEGORY = "MOSS-Image"

    def apply_weights(self, conditioning, clip, text, weight_multiplier=5.0, emphasis_mode="Original"):
        hf_tokenizer = get_hf_tokenizer(clip)
        template = get_chat_template(clip)
        tokens, multipliers = tokenize_moss_prompt(hf_tokenizer, text, emphasis_mode, template=template)
        _, mask = compute_weight_masks(multipliers, weight_multiplier)

        out_cond = []
        for cond, cond_dict in conditioning:
            new_dict = cond_dict.copy()
            seq_len = cond.shape[1]
            cur_mask = mask
            if cur_mask.shape[0] < seq_len:
                cur_mask = F.pad(cur_mask, (0, 0, 0, seq_len - cur_mask.shape[0]), value=1.0)
            elif cur_mask.shape[0] > seq_len:
                cur_mask = cur_mask[:seq_len]

            mask_tensor = cur_mask.unsqueeze(0).to(device=cond.device, dtype=cond.dtype)
            new_dict["c_negpip_mask"] = mask_tensor
            new_dict["moss_negpip_mask"] = mask_tensor
            out_cond.append((cond, new_dict))

        return (out_cond,)
