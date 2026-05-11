"""Reward function for ASearcher data using GAIA scoring rules.

The model answer is compared against the original GAIA ground truth as well as
any additional answers provided in ``extra_info['aug_answers']``. If the
prediction matches any candidate when evaluated with the GAIA scorer, the
sample receives full credit.
"""

from __future__ import annotations

from typing import Iterable

from . import gaia


def _iter_candidate_answers(ground_truth: str, extra_info: dict | None) -> Iterable[str]:
    yield ground_truth

    if not extra_info:
        return

    aug_answers = extra_info.get("aug_answers")
    if not aug_answers:
        return

    if isinstance(aug_answers, str):
        candidates = [part.strip() for part in aug_answers.split(",")]
    else:
        candidates = [str(item).strip() for item in aug_answers]

    for answer in candidates:
        if answer:
            yield answer

"""
- 想启用 LLM-as-a-judge：在 extra_info 中提供
  - question：原题面
  - reward_model: {'style': 'llm', 'model': '<your-model-name>'}
- 若未提供或 style='rule'，则沿用旧的规则打分。
"""
def compute_score(solution_str: str, ground_truth: str, *, extra_info: dict | None = None, **_) -> float:
    prediction = gaia.extract_answer(solution_str) or ""

    reward_model = (extra_info or {}).get("reward_model", {})
    style = (reward_model or {}).get("style", "rule")

    if style == "rule":
        for candidate in _iter_candidate_answers(str(ground_truth), extra_info):
            if gaia._question_scorer(prediction, str(candidate)):
                return 1.0
        return 0.0

    # LLM-as-a-judge path, reuse GAIA's scorer
    question = (extra_info or {}).get("question")
    model = (reward_model or {}).get("model")
    for candidate in _iter_candidate_answers(str(ground_truth), extra_info):
        if gaia._llm_scorer(prediction, str(candidate), question, model):
            return 1.0
    return 0.0
