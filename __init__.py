"""ComfyUI MOSS Prompt Weights (DiT Attention V-Scaling) Extension."""

from .nodes import ApplyMOSSPromptWeights, MOSSConditioningWeights, MOSSTextEncode

NODE_CLASS_MAPPINGS = {
    "MOSSTextEncode": MOSSTextEncode,
    "ApplyMOSSPromptWeights": ApplyMOSSPromptWeights,
    "MOSSConditioningWeights": MOSSConditioningWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSTextEncode": "🎯 MOSS Text Encode (V-Scaling)",
    "ApplyMOSSPromptWeights": "🎯 Apply MOSS Prompt Weights (V-Scaling)",
    "MOSSConditioningWeights": "🎯 MOSS Conditioning Weights (V-Scaling)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
