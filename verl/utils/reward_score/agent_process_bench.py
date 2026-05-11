from __future__ import annotations

import json
import re
from typing import Any


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    json_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = json_block_pattern.findall(text)
    if matches:
        json_str = matches[-1].strip()
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    end = text.rfind("}")
    if end < 0:
        raise ValueError("LLM output is not JSON")

    brace_count = 0
    start = -1
    for idx in range(end, -1, -1):
        if text[idx] == "}":
            brace_count += 1
        elif text[idx] == "{":
            brace_count -= 1
            if brace_count == 0:
                start = idx
                break

    if start < 0:
        raise ValueError("LLM output is not JSON")

    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object")
    return obj


def _coerce_int_label(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("label must be -1/0/1, got bool")
    if isinstance(val, int):
        out = val
    elif isinstance(val, float) and val.is_integer():
        out = int(val)
    elif isinstance(val, str) and val.strip() in {"-1", "0", "1"}:
        out = int(val.strip())
    else:
        raise ValueError(f"label must be -1/0/1, got {val!r}")
    if out not in (-1, 0, 1):
        raise ValueError(f"label must be -1/0/1, got {out}")
    return out


def _normalize_final_label(val: Any) -> int:
    label = _coerce_int_label(val)
    if label in (-1, 0):
        return -1
    if label == 1:
        return 1
    raise ValueError(f"unexpected final_label: {val!r}")


def _normalize_judge_output(
    raw: dict[str, Any],
    *,
    assistant_indices: list[int],
) -> tuple[dict[str, int], int, dict[str, Any]]:
    if "step_labels" not in raw or "final_label" not in raw:
        raise ValueError("missing step_labels/final_label in judge output")
    step_labels_raw = raw.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError("step_labels must be an object")

    step_labels_raw = {str(k): v for k, v in step_labels_raw.items()}
    expected_keys = [str(idx) for idx in assistant_indices]
    step_labels: dict[str, int] = {}
    for key in expected_keys:
        if key not in step_labels_raw:
            raise ValueError(f"missing step_labels[{key}]")
        step_labels[key] = _coerce_int_label(step_labels_raw[key])

    final_label = _normalize_final_label(raw.get("final_label"))

    explanations = raw.get("explanations") or {}
    if not isinstance(explanations, dict):
        raise ValueError("explanations must be an object")
    steps_expl = explanations.get("steps") or {}
    if not isinstance(steps_expl, dict):
        raise ValueError("explanations.steps must be an object")
    steps_expl = {str(k): v for k, v in steps_expl.items()}
    for key in expected_keys:
        if key not in steps_expl:
            raise ValueError(f"missing explanations.steps[{key}]")
        if not isinstance(steps_expl[key], str):
            steps_expl[key] = str(steps_expl[key])
    final_expl = explanations.get("final")
    if not isinstance(final_expl, str):
        final_expl = "" if final_expl is None else str(final_expl)

    return step_labels, final_label, {"steps": steps_expl, "final": final_expl}


def calculate_PRM_score(predict_step_labels: dict[str, int], step_labels: dict[str, int]) -> float:
    assert isinstance(predict_step_labels, dict)
    assert isinstance(step_labels, dict)
    if predict_step_labels.keys() != step_labels.keys():
        return -1.0
    if not step_labels:
        return 0.0
    correct_num = sum(int(predict_step_labels[key] == gt) for key, gt in step_labels.items())
    return correct_num / len(step_labels)


def _normalize_gt_step_labels(
    raw_step_labels: Any,
    assistant_indices: list[int],
    *,
    required: bool,
) -> dict[str, int] | None:
    if raw_step_labels is None:
        if required:
            raise ValueError("ground-truth step_labels are required for process reward")
        return None
    if not isinstance(raw_step_labels, dict):
        raise ValueError(f"ground-truth step_labels must be an object, got {type(raw_step_labels)!r}")

    gt_step_labels = {str(k): _coerce_int_label(v) for k, v in raw_step_labels.items() if v is not None}
    expected_keys = {str(idx) for idx in assistant_indices}
    if gt_step_labels.keys() != expected_keys:
        raise ValueError(
            "ground-truth step_labels keys must match assistant_indices: "
            f"got={sorted(gt_step_labels.keys())}, expected={sorted(expected_keys)}"
        )
    return gt_step_labels


def _first_neg1_index(step_labels: dict[str, int]) -> int:
    wrong_indices = [int(key) for key, value in step_labels.items() if value == -1]
    return min(wrong_indices) if wrong_indices else -1


def _empty_agent_process_metrics(gt_step_labels: dict[str, int] | None) -> dict[str, float]:
    return {
        "agent_step_match": 0.0,
        "agent_step_total": float(len(gt_step_labels) if gt_step_labels is not None else 0),
        "agent_first_neg1_match": 0.0,
        "agent_first_neg1_total": float(gt_step_labels is not None),
    }


def _agent_process_metrics(pred_step_labels: dict[str, int], gt_step_labels: dict[str, int]) -> dict[str, float]:
    if pred_step_labels.keys() != gt_step_labels.keys():
        return _empty_agent_process_metrics(gt_step_labels)

    step_match = sum(int(pred_step_labels[key] == gt) for key, gt in gt_step_labels.items())
    first_neg1_match = float(_first_neg1_index(pred_step_labels) == _first_neg1_index(gt_step_labels))
    return {
        "agent_step_match": float(step_match),
        "agent_step_total": float(len(gt_step_labels)),
        "agent_first_neg1_match": first_neg1_match,
        "agent_first_neg1_total": 1.0,
    }


def compute_score(solution_str: str, ground_truth: Any, *, extra_info: dict | None = None, **_) -> dict[str, Any]:
    del ground_truth
    assert isinstance(extra_info, dict), "extra_info is required"
    assert "assistant_indices" in extra_info, "extra_info.assistant_indices is required"
    assert "reward_type" in extra_info, "extra_info.reward_type is required"
    assert "final_label" in extra_info, "extra_info.final_label is required"

    assistant_indices = [int(idx) for idx in extra_info["assistant_indices"]]
    assert assistant_indices, "assistant_indices must be non-empty"
    reward_type = str(extra_info["reward_type"])
    if reward_type not in {"outcome", "process"}:
        raise ValueError(f"reward_type must be 'outcome' or 'process', got {reward_type!r}")

    gt_final_label = _normalize_final_label(extra_info["final_label"])
    gt_step_labels = _normalize_gt_step_labels(
        extra_info.get("step_labels"),
        assistant_indices,
        required=reward_type == "process",
    )

    result = {
        "score": 0.0,
        "ORM_score": 0.0,
        "final_acc": 0.0,
        "PRM_score": -1.0,
        "format": 0.0,
    }
    result.update(_empty_agent_process_metrics(gt_step_labels))

    try:
        raw = _extract_json_object(solution_str)
        predict_step_labels, predict_final_label, _explanations = _normalize_judge_output(
            raw,
            assistant_indices=assistant_indices,
        )
    except Exception:
        return result

    orm_score = float(predict_final_label == gt_final_label)
    result["ORM_score"] = orm_score
    result["final_acc"] = orm_score
    result["format"] = 1.0

    if gt_step_labels is not None:
        result["PRM_score"] = calculate_PRM_score(predict_step_labels, gt_step_labels)
        result.update(_agent_process_metrics(predict_step_labels, gt_step_labels))

    if reward_type == "outcome":
        result["score"] = orm_score
    else:
        result["score"] = result["PRM_score"]
    return result
