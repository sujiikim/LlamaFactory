# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def mask_thought_in_target_ids_action(target_ids: list[int], tokenizer: "PreTrainedTokenizer") -> list[int]:
    r"""Mask the Thought part in target_ids, keeping only the Action part for loss calculation.
    Args:
        target_ids: List of token IDs representing the target response
        tokenizer: Tokenizer used to decode and encode text
    Returns:
        List of token IDs with Thought part masked as IGNORE_INDEX
    """
    if not target_ids:
        return target_ids

    try:
        decoded = tokenizer.decode(target_ids, skip_special_tokens=False)

        action_idx = decoded.find("Action:")
        if action_idx == -1:
            return target_ids

        # Use binary search for accurate token boundary (same as toolcall variant)
        def _decode_prefix_len(k: int) -> int:
            return len(tokenizer.decode(target_ids[:k], skip_special_tokens=False))

        lo, hi = 0, len(target_ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _decode_prefix_len(mid) <= action_idx:
                lo = mid
            else:
                hi = mid - 1
        thought_len = lo

        if thought_len >= len(target_ids):
            return target_ids

        masked_target_ids = target_ids.copy()
        masked_target_ids[:thought_len] = [IGNORE_INDEX] * thought_len

        return masked_target_ids
    except Exception:
        return target_ids


def mask_thought_in_target_ids_toolcall(target_ids: list[int], tokenizer: "PreTrainedTokenizer") -> list[int]:
    r"""Mask everything before the tool-call / code block in target_ids.

    Supports two formats:
      1. EvoCUA: finds ``<tool_call>`` tag and keeps everything from there.
      2. OpenCUA: finds ``## Code:`` header and keeps everything from there
         (i.e. the fenced ``python`` / ``code`` blocks that follow).
    """
    if not target_ids:
        return target_ids

    try:
        decoded = tokenizer.decode(target_ids, skip_special_tokens=False)

        action_idx = decoded.find("<tool_call>")
        if action_idx <= 0:
            # Fallback: OpenCUA format uses "## Code:" as the delimiter
            action_idx = decoded.find("## Code:")
        if action_idx <= 0:
            return target_ids

        # Token boundary from target_ids (re-encode can differ from decode prefix)
        def _decode_prefix_len(k: int) -> int:
            return len(tokenizer.decode(target_ids[:k], skip_special_tokens=False))

        lo, hi = 0, len(target_ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _decode_prefix_len(mid) <= action_idx:
                lo = mid
            else:
                hi = mid - 1
        thought_len = lo

        if thought_len >= len(target_ids):
            return target_ids

        masked_target_ids = target_ids.copy()
        masked_target_ids[:thought_len] = [IGNORE_INDEX] * thought_len

        return masked_target_ids
    except Exception:
        return target_ids