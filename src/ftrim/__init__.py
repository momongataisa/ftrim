"""FTrim: trimming answer-format discussion (format talk) from LLM self-training data with counterfactual prompts.

The method itself is in :mod:`ftrim.core` (numpy only); the rest of the package is the experiment of the paper.
"""
from .core import format_talk_lines, line_scores, random_lines, split_lines, stm_mask, token_scores

__version__ = "1.0.0"
__all__ = ["format_talk_lines", "line_scores", "random_lines", "split_lines", "stm_mask", "token_scores"]
