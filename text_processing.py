"""Text processing and prompt attention parsing for MOSS-Image in ComfyUI."""

import json
import re
from typing import List, Optional, Tuple

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*([)\]])|
\)|
\]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str, emphasis: str = "Original") -> List[Tuple[str, float]]:
    """Parse a prompt with (word:weight) and [word] syntax into segments with weights.
    
    Correctly treats escaped brackets \\( and \\) as literal characters with weight 1.0.
    Supports (tag:1.2), (tag), [tag], [tag:0.8], (ugly:-1.0).
    """
    if emphasis == "Ignore":
        return [(text, 1.0)]

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1.0 / 1.1

    def multiply_range(start_position: int, multiplier: float):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        token = m.group(0)
        weight = m.group(1)
        close_bracket = m.group(2)

        if token.startswith("\\"):
            # Escaped bracket or backslash, treat as literal character with weight 1.0
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None:
            if close_bracket == ")" and round_brackets:
                multiply_range(round_brackets.pop(), float(weight))
            elif close_bracket == "]" and square_brackets:
                multiply_range(square_brackets.pop(), float(weight))
            elif round_brackets:
                multiply_range(round_brackets.pop(), float(weight))
            elif square_brackets:
                multiply_range(square_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, token)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append([" BREAK ", 1.0])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    # Merge adjacent segments with identical weights
    i = 0
    while i + 1 < len(res):
        if abs(res[i][1] - res[i + 1][1]) < 1e-6:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return [(t, w) for t, w in res]


# ================================================================================ #
# MOSS JSON Layout Parsing Support
# ================================================================================ #

_CAPTION_KEY = "caption"


def _json_strings(text: str) -> List[Tuple[int, int]]:
    spans = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
        else:
            i += 1
    return spans


def _caption_spans(text: str) -> List[Tuple[int, int]]:
    strings = _json_strings(text)
    spans = []
    for k in range(len(strings) - 1):
        key_start, key_end = strings[k]
        if text[key_start:key_end] != f'"{_CAPTION_KEY}"':
            continue
        between = text[key_end : strings[k + 1][0]]
        if between.strip() != ":":
            continue
        value_start, value_end = strings[k + 1]
        spans.append((value_start + 1, value_end - 1))
    return spans


def _escape(segment: str) -> str:
    return json.dumps(segment, ensure_ascii=False)[1:-1]


def _layout(text: str) -> Optional[Tuple[str, str]]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        layout, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return None
    if not isinstance(layout, dict):
        return None
    return stripped[:end], stripped[end:]


def _parse_layout(layout: str, emphasis_name: str) -> List[Tuple[str, float]]:
    parsed = []
    pos = 0
    for start, end in _caption_spans(layout):
        parsed.append((layout[pos:start], 1.0))
        raw = layout[start:end]
        try:
            caption = json.loads(f'"{raw}"')
        except ValueError:
            parsed.append((raw, 1.0))
            pos = end
            continue
        segments = parse_prompt_attention(caption, emphasis_name)
        if all(abs(w - 1.0) < 1e-6 for _, w in segments):
            parsed.append((raw, 1.0))
        else:
            parsed.extend((_escape(t), w) for t, w in segments)
        pos = end
    parsed.append((layout[pos:], 1.0))
    return parsed


def parse_moss_prompt(line: str, emphasis_name: str = "Original") -> List[Tuple[str, float]]:
    """Parse either a standard prompt or a MOSS layout JSON prompt."""
    found = _layout(line)
    if found is None:
        return parse_prompt_attention(line, emphasis_name)
    layout, remainder = found
    parsed = _parse_layout(layout, emphasis_name)
    if remainder.strip():
        parsed.extend(parse_prompt_attention(remainder, emphasis_name))
    return parsed
