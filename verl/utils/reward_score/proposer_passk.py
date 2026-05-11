"""Reward helpers for proposer difficulty calibration using pass@k."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import requests
from requests import Response

__all__ = [
    "PassAtKResult",
    "fetch_pass_at_k",
    "proposer_difficulty_reward",
]

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 2.0


@dataclass(slots=True)
class PassAtKResult:
    """Container for pass@k measurements produced by a reference solver."""

    pass_at_k: float
    golden_difficulty: float
    completions: Sequence[str]
    successes: Sequence[bool]


def _parse_completions(response_json: dict) -> list[str]:
    choices = response_json.get("choices", [])
    completions: list[str] = []
    for choice in choices:
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            if isinstance(message, dict) and "content" in message:
                completions.append(str(message["content"]))
                continue
            if "text" in choice:
                completions.append(str(choice["text"]))
        else:
            logger.debug("Unexpected choice type when parsing pass@k response: %r", choice)
    return completions


def _request_pass_at_k(
    prompt: str,
    *,
    url: str,
    api_key: str | None,
    model_name: str,
    k: int,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> list[str]:
    endpoint = url.rstrip("/") + DEFAULT_ENDPOINT
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "n": k,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_new_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            response: Response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            completions = _parse_completions(response.json())
            if completions:
                return completions
            logger.warning("Empty completions received for pass@k request (attempt %d)", attempt + 1)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to query SGLang server for pass@k") from exc
            sleep_time = backoff ** attempt
            logger.warning(
                "Pass@k request failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
    raise RuntimeError("SGLang server returned empty completions for pass@k evaluation")


def fetch_pass_at_k(
    prompt: str,
    checker: Callable[[str], bool | float],
    *,
    url: str,
    key: str | None,
    model_name: str,
    k: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> PassAtKResult:
    """Query the reference solver and estimate pass@k for the given prompt.

    Args:
        prompt: The task description passed to the reference solver.
        checker: Callable that returns ``True`` for a correct completion.
        url: Base URL of the SGLang server (e.g. ``http://localhost:30000``).
        key: Optional API key used for the ``Authorization`` header.
        model_name: Name of the deployed model to query.
        k: Number of sampled completions.
        temperature: Sampling temperature used for generation.
        max_new_tokens: Maximum number of tokens generated per completion.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries upon failure.
        backoff: Exponential backoff factor (seconds).

    Returns:
        A :class:`PassAtKResult` with pass@k statistics.
    """
    if k <= 0:
        raise ValueError("k must be strictly positive when computing pass@k")

    completions = _request_pass_at_k(
        prompt,
        url=url,
        api_key=key,
        model_name=model_name,
        k=k,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )

    results: list[bool] = []
    for completion in completions:
        outcome = checker(completion)
        if isinstance(outcome, bool):
            results.append(outcome)
        else:
            results.append(bool(outcome))

    success_count = sum(results)
    total = len(results) if results else k
    pass_at_k = success_count / total
    golden_difficulty = 1.0 - pass_at_k

    return PassAtKResult(
        pass_at_k=pass_at_k,
        golden_difficulty=golden_difficulty,
        completions=tuple(completions),
        successes=tuple(results),
    )


def proposer_difficulty_reward(
    prompt: str,
    target_difficulty: float,
    checker: Callable[[str], bool | float],
    *,
    url: str,
    key: str | None,
    model_name: str,
    k: int = 5,
    tolerance: float = 0.1,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> tuple[float, PassAtKResult]:
    """Reward function for the proposer conditioned on difficulty targets.

    The reward is based on how closely the golden difficulty (derived from
    ``pass@k`` of a reference solver) matches the requested target.

    Args:
        prompt: Task specification proposed by the proposer.
        target_difficulty: Desired difficulty value from the curriculum controller.
        checker: Callable indicating whether a completion solves the task.
        url: Base URL of the SGLang server.
        key: Optional API key for authentication.
        model_name: Name of the deployed Qwen model to query.
        k: Number of completions to sample for pass@k.
        tolerance: Width of the zero-penalty band around the target difficulty.
        temperature: Sampling temperature when querying the solver.
        max_new_tokens: Maximum number of generated tokens per completion.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for HTTP requests.
        backoff: Exponential backoff factor between retries.

    Returns:
        Tuple ``(reward, stats)`` with the scalar reward and the full
        :class:`PassAtKResult` diagnostics.
    """
    stats = fetch_pass_at_k(
        prompt,
        checker,
        url=url,
        key=key,
        model_name=model_name,
        k=k,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )

    target = max(0.0, min(1.0, target_difficulty))
    delta = abs(stats.golden_difficulty - target)

    if delta <= tolerance:
        reward = 1.0
    else:
        reward = max(0.0, 1.0 - (delta - tolerance))

    return reward, stats
