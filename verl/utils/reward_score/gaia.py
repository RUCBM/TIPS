"""
GAIA scoring aligned with the official scorer.py:
https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py

Steps:
- Extract model_answer from <answer>...</answer> in solution_str.
- Apply question_scorer(model_answer, ground_truth) with the same rules:
  * If ground_truth is numeric -> compare floats (model side normalized by removing $, %, ,).
  * Else if ground_truth contains ',' or ';' -> split list; compare element-wise (numeric vs. normalized string w/ remove_punct=False).
  * Else -> compare normalized strings (remove whitespace, lowercase, remove punctuation).
"""

import os
import time
import re
import string
import re
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def remove_think_tags(s):
    # 找到最后一个 </think>
    idx = s.rfind("</think>")
    if idx == -1:
        # 没有 </think> 就原样返回
        return s.strip()
    # 返回 </think> 之后的内容
    return s[idx + len("</think>"):].strip()

def extract_answer(text: str) -> str | None:
    if not text:
        return None
    text = remove_think_tags(text)
    # 核心修改在这里：在正则表达式的开头加上 .*
    matches = re.findall(r".*<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()

def _normalize_number_str(number_str: str) -> float:
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        # follow upstream: return inf to ensure mismatch
        return float("inf")


def _split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
    # Remove all whitespace
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _question_scorer(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"

    # number ground truth
    if _is_float(ground_truth):
        normalized_answer = _normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # list ground truth
    elif any(ch in ground_truth for ch in [",", ";"]):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                comparisons.append(_normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    _normalize_str(ma_elem, remove_punct=False)
                    == _normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # string ground truth
    else:
        return _normalize_str(model_answer) == _normalize_str(ground_truth)


def _llm_scorer(model_answer: str, ground_truth: str, question: str, model: str) -> bool:
    if not model:
        raise Exception("No LLM model specified")
    # Prompt template with placeholders to inject strings
    # LLM_JUDGE_PROMPT_TEMPLATE = """
    # You are an expert evaluation assistant.
    # Your task is to judge whether the predicted answer correctly answers the question, using the labeled answer as a reference.
    #
    # ---
    # Question:
    # {question}
    #
    # Labeled Answer (Reference):
    # {labeled_answer}
    #
    # Predicted Answer:
    # {pred_answer}
    # ---
    #
    # Evaluation Criteria:
    # - **Primary focus**: Does the predicted answer correctly capture the **key information** needed to answer the question?
    # - Use the labeled answer as a reference to identify what the key information is.
    # - Focus on **factual correctness and alignment of core information** rather than completeness or surface similarity.
    # - **Key information alignment** (correct model information retrieval) is the most important factor.
    # - Minor rewordings, synonyms, or paraphrases that preserve the key information should be considered correct.
    # - **Missing non-critical/supplementary information** that can be covered is acceptable - still considered **correct**.
    # - **Additional relevant details or more specific information** beyond the labeled answer is acceptable - still considered **correct**.
    # - Only mark as **incorrect** if:
    #   * The predicted answer misses **critical/key information** required to answer the question
    #   * The predicted answer provides **factually incorrect information**
    #   * The predicted answer is **unrelated or contradicts** the question's requirements
    #
    #
    # Final Decision:
    # Please respond with only one word:
    # - "Correct" → if the predicted answer correctly captures the key information with factual accuracy (even if missing minor details or including additional details).
    # - "Incorrect" → if it misses critical information, contains factually incorrect information, or is unrelated to the question.
    #
    # Answer:
    # """

    LLM_JUDGE_PROMPT_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

    Question: {question}

    Labeled Answer: {labeled_answer}

    Predicted Answer: {pred_answer}

    Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
    """.strip()

    # Build the concrete prompt by injecting strings
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=str(question or ""),
        labeled_answer=str(ground_truth or ""),
        pred_answer=str(model_answer or ""),
    )

    client = OpenAI()

    # Up to 5 attempts with simple backoff
    last_text_response = ""
    last_exception: Exception | None = None
    max_primary_attempts = 5
    t_primary_start = time.time()
    for attempt in range(max_primary_attempts):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                top_p=1.0,
                n=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logit_bias={},
                max_tokens=16,
                timeout=300
            )
            text_response = ""
            if getattr(response, "choices", None):
                # Some SDKs return None; guard accordingly
                text_response = response.choices[0].message.content or ""
            last_text_response = text_response

            # Relaxed decision: substring check for robustness in early trials
            normalized = text_response.strip().lower()
            if "incorrect" in normalized:
                logger.info(
                    "gaia_llm_scorer.primary.success",
                    extra={
                        "phase": "primary",
                        "attempt": attempt + 1,
                        "total_attempts": max_primary_attempts,
                        "duration_ms": int((time.time() - t_primary_start) * 1000),
                        "decision": "incorrect",
                    },
                )
                return False
            if "correct" in normalized:
                logger.info(
                    "gaia_llm_scorer.primary.success",
                    extra={
                        "phase": "primary",
                        "attempt": attempt + 1,
                        "total_attempts": max_primary_attempts,
                        "duration_ms": int((time.time() - t_primary_start) * 1000),
                        "decision": "correct",
                    },
                )
                return True

            # If response is unexpected, retry a couple times
        except Exception as e:
            logger.warning('Calling LLM judge raised an exception: %s', e)
            # swallow and retry after a short backoff
            last_exception = e
        # Exponential-ish backoff: 0.2s, 0.4s, 0.8s, ...
        time.sleep(min(0.2 * (2 ** attempt), 10))

    # After 5 attempts, log an error once before applying fallback
    try:
        question_snippet = (str(question) or "")[:120]
        response_snippet = (last_text_response or "")[:120]
        logger.error(
            "LLM scorer failed after 5 attempts (model=%s). Question=%r, last_response=%r, last_exception=%r",
            model,
            question_snippet,
            response_snippet,
            last_exception,
        )
    except Exception:
        # Avoid any logging-related failure from breaking scoring
        pass

    # Secondary fallback: use browser_agent config (5 attempts)
    def _load_browser_agent_config() -> dict:
        """Load browser_agent config from YAML or env defaults."""
        # Allow override via env var for path
        default_cfg_path = "examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml"
        cfg_path = os.environ.get(
            "BROWSER_AGENT_CONFIG_PATH",
            str(default_cfg_path),
        )
        url = "http://localhost:28888/v1"
        key = ""
        model_name = "qwen3-4b-instruct-2507"
        timeout_sec: Optional[float] = 90.0

        try:
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                # Convert to plain Python containers to avoid DictConfig/ListConfig isinstance issues
                cfg_py = OmegaConf.to_container(cfg, resolve=True) or {}
                tools = cfg_py.get("tools") if isinstance(cfg_py, dict) else None
                if isinstance(tools, list):
                    for item in tools:
                        conf = item.get("config", {}) if isinstance(item, dict) else {}
                        ba = conf.get("browser_agent") if isinstance(conf, dict) else None
                        if isinstance(ba, dict):
                            new_url = ba.get("browser_agent_url")
                            if new_url:
                                url = new_url
                            if "browser_agent_key" in ba:
                                key = ba.get("browser_agent_key") or ""
                            new_model = ba.get("browser_agent_model_name")
                            if new_model:
                                model_name = new_model
                            if ba.get("timeout") is not None:
                                try:
                                    timeout_sec = float(ba.get("timeout"))
                                except Exception:
                                    timeout_sec = timeout_sec
                            break
        except Exception as e:
            logger.warning("Failed to load browser_agent config from %s: %s", cfg_path, e)

        return {
            "browser_agent_url": url,
            "browser_agent_key": key,
            "browser_agent_model_name": model_name,
            "timeout": timeout_sec,
        }

    ba_cfg = _load_browser_agent_config()
    ba_url = ba_cfg.get("browser_agent_url")
    ba_key = ba_cfg.get("browser_agent_key")
    ba_model = ba_cfg.get("browser_agent_model_name")
    ba_timeout = ba_cfg.get("timeout") or 90.0

    logger.info(
        "gaia_llm_scorer.fallback.browser_agent.start",
        extra={
            "phase": "fallback_browser_agent",
            "attempts": 5,
            "model": ba_model,
            "url": ba_url,
        },
    )
    t_ba_start = time.time()
    try:
        ba_client = OpenAI(api_key=ba_key or None, base_url=ba_url)
    except Exception as e:
        logger.error("Initialize browser_agent client failed: %s", e)
        ba_client = None

    if ba_client is not None and ba_url and ba_model:
        attempts = 5
        for i in range(attempts):
            t1 = time.time()
            try:
                resp = ba_client.chat.completions.create(
                    model=ba_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    top_p=1.0,
                    n=1,
                    max_tokens=16,
                    timeout=float(ba_timeout),
                )
                content = ""
                if getattr(resp, "choices", None):
                    content = resp.choices[0].message.content or ""
                norm = (content or "").strip().lower()
                latency_ms = int((time.time() - t1) * 1000)
                if "incorrect" in norm:
                    logger.info(
                        "gaia_llm_scorer.fallback.browser_agent.attempt",
                        extra={
                            "attempt": i + 1,
                            "decision": "incorrect",
                            "latency_ms": latency_ms,
                        },
                    )
                    return False
                if "correct" in norm:
                    logger.info(
                        "gaia_llm_scorer.fallback.browser_agent.attempt",
                        extra={
                            "attempt": i + 1,
                            "decision": "correct",
                            "latency_ms": latency_ms,
                        },
                    )
                    return True

                logger.info(
                    "gaia_llm_scorer.fallback.browser_agent.attempt",
                    extra={
                        "attempt": i + 1,
                        "decision": "other",
                        "latency_ms": latency_ms,
                    },
                )
            except Exception as e:
                logger.warning("gaia_llm_scorer.browser_agent attempt %d failed: %s", i + 1, e)

        logger.error(
            "gaia_llm_scorer.fallback.browser_agent.failed",
            extra={
                "duration_ms": int((time.time() - t_ba_start) * 1000),
                "attempts": attempts,
            },
        )

    # Final fallback heuristic: relaxed substring check on last primary response
    normalized_last = (last_text_response or "").strip().lower()
    if "incorrect" in normalized_last:
        return False
    if "correct" in normalized_last:
        return True
    return False


"""
- 想启用 LLM-as-a-judge：在 extra_info 中提供
  - question：原题面
  - reward_model: {'style': 'llm', 'model': '<your-model-name>'}
- 若未提供或 style='rule'，则沿用旧的规则打分。
"""


def _analyze_tool_calls(text: str) -> tuple[int, bool]:
    if not text:
        return 0, False
    segments = re.split(r"\nassistant\n", text, flags=re.IGNORECASE)
    total = 0
    multi_tool_violation = False
    for segment in segments[1:]:
        cnt = len(re.findall(r"<tool_call>", segment, flags=re.IGNORECASE))
        total += cnt
        if cnt > 1:
            multi_tool_violation = True
    return total, multi_tool_violation


# todo: missing answer tag应该只惩罚最后一项，这样会带来一些bias
def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    pred = extract_answer(solution_str)

    extra_info = kwargs.get("extra_info", {}) or {}
    reward_model = extra_info.get("reward_model") or {}
    style = reward_model.get("style", "rule")

    if style == "rule":
        if pred is None:
            base_correct = 0
        else:
            base_correct = _question_scorer(pred, str(ground_truth))
    else:
        question = extra_info.get("question")
        model = reward_model.get("model")
        if pred is None:
            base_correct = 0
        else:
            base_correct = _llm_scorer(pred, str(ground_truth), question, model)

    violations: list[str] = []
    tool_call_cnt, multi_tool_violation = _analyze_tool_calls(solution_str)
    if multi_tool_violation:
        violations.append("multi_tool_call")
    # if pred is None:
    #     violations.append("missing_answer_tag")

    if violations:
        final_score = 0.9 * base_correct
    else:
        final_score = 0.1 + 0.9 * base_correct

    logger.debug("GAIA Scorer | pred: %s | ground_truth: %s | final_score: %f", pred, ground_truth, final_score)
    return {
        "score": final_score,
        "acc": 1.0 if base_correct else 0.0,
        "format_violation": bool(violations),
        "format_violation_reason": ",".join(violations),
        "tool_call_count": tool_call_cnt,
    }

