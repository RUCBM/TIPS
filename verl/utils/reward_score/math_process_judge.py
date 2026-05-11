from __future__ import annotations

from typing import Any

from .agent_process_bench import _extract_json_object


def _coerce_binary_label(val: Any) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        if val in (0, 1):
            return val
        if val == -1:
            return 0
    if isinstance(val, float):
        if val in (0.0, 1.0):
            return int(val)
        if val == -1.0:
            return 0
    if isinstance(val, str):
        stripped = val.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
        if stripped == "-1":
            return 0
        if stripped.lower() in {"false", "true"}:
            return int(stripped.lower() == "true")
    raise ValueError(f"label must be binary-compatible, got {val!r}")


def _first_error_index(step_labels: dict[str, int], step_indices: list[int]) -> int:
    for idx in step_indices:
        if step_labels[str(idx)] == 0:
            return idx
    return -1


def _normalize_output(raw: dict[str, Any], *, step_indices: list[int]) -> tuple[dict[str, int], int]:
    if "step_labels" not in raw or "final_label" not in raw:
        raise ValueError("missing step_labels/final_label in judge output")

    step_labels_raw = raw.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError("step_labels must be an object")

    normalized_step_labels: dict[str, int] = {}
    for idx in step_indices:
        key = str(idx)
        if key not in step_labels_raw:
            raise ValueError(f"missing step_labels[{key}]")
        normalized_step_labels[key] = _coerce_binary_label(step_labels_raw[key])

    final_label = _coerce_binary_label(raw.get("final_label"))
    return normalized_step_labels, final_label


def _step_accuracy(pred_step_labels: dict[str, int], gt_step_labels: dict[str, int]) -> float:
    if pred_step_labels.keys() != gt_step_labels.keys():
        return 0.0
    correct = 0
    for key, pred in pred_step_labels.items():
        if pred == gt_step_labels[key]:
            correct += 1
    return correct / len(gt_step_labels) if gt_step_labels else 0.0


def _collect_prmbench_step_stats(
    pred_step_labels: dict[str, int],
    gt_step_labels: dict[str, int],
    *,
    step_indices: list[int],
    gt_first_error: int,
) -> dict[str, float]:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    correct_step_match = 0
    correct_step_total = 0
    wrong_step_match = 0
    wrong_step_total = 0
    total_step_match = 0

    for idx in step_indices:
        key = str(idx)
        pred = pred_step_labels[key]
        gt = gt_step_labels[key]
        is_match = int(pred == gt)
        total_step_match += is_match
        if gt == 1:
            correct_step_total += 1
            correct_step_match += is_match
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            wrong_step_total += 1
            wrong_step_match += is_match
            if pred == 0:
                tn += 1
            else:
                fp += 1

    first_error_total = float(gt_first_error != -1)
    first_error_match = 0.0
    if gt_first_error != -1:
        first_error_match = float(pred_step_labels[str(gt_first_error)] == 0)

    return {
        "prmbench_tp": float(tp),
        "prmbench_fp": float(fp),
        "prmbench_tn": float(tn),
        "prmbench_fn": float(fn),
        "prmbench_correct_step_match": float(correct_step_match),
        "prmbench_correct_step_total": float(correct_step_total),
        "prmbench_wrong_step_match": float(wrong_step_match),
        "prmbench_wrong_step_total": float(wrong_step_total),
        "prmbench_total_step_match": float(total_step_match),
        "prmbench_total_step_total": float(len(step_indices)),
        "prmbench_first_error_match": float(first_error_match),
        "prmbench_first_error_total": float(first_error_total),
        "prmbench_model_response_acc": float(
            sum(pred_step_labels[str(idx)] for idx in step_indices) / len(step_indices) if step_indices else -1
        ),
    }


def compute_score(solution_str: str, ground_truth: str, *, extra_info: dict | None = None, **_) -> dict[str, Any]:
    del ground_truth
    assert isinstance(extra_info, dict), "extra_info is required"
    assert "step_indices" in extra_info, "extra_info.step_indices is required"
    assert "step_labels" in extra_info, "extra_info.step_labels is required"
    assert "final_label" in extra_info, "extra_info.final_label is required"
    assert "reward_type" in extra_info, "extra_info.reward_type is required"

    step_indices = [int(idx) for idx in extra_info["step_indices"]]
    raw_gt_step_labels = extra_info["step_labels"]
    gt_step_labels = {}
    for idx in step_indices:
        key = str(idx)
        if key not in raw_gt_step_labels:
            raise ValueError(f"missing gt step_labels[{key}]")
        gt_step_labels[key] = _coerce_binary_label(raw_gt_step_labels[key])
    gt_final_label = _coerce_binary_label(extra_info["final_label"])
    gt_first_error = int(extra_info.get("first_error_index", _first_error_index(gt_step_labels, step_indices)))

    result = {
        "score": 0.0,
        "ORM_score": 0.0,
        "PRM_score": 0.0,
        "format": 0.0,
        "first_error_match": 0.0,
        "first_error_error_match": 0.0,
        "first_error_error_total": float(gt_first_error != -1),
        "first_error_correct_match": 0.0,
        "first_error_correct_total": float(gt_first_error == -1),
        "prmbench_tp": 0.0,
        "prmbench_fp": 0.0,
        "prmbench_tn": 0.0,
        "prmbench_fn": 0.0,
        "prmbench_correct_step_match": 0.0,
        "prmbench_correct_step_total": 0.0,
        "prmbench_wrong_step_match": 0.0,
        "prmbench_wrong_step_total": 0.0,
        "prmbench_total_step_match": 0.0,
        "prmbench_total_step_total": 0.0,
        "prmbench_first_error_match": 0.0,
        "prmbench_first_error_total": 0.0,
        "prmbench_model_response_acc": -1.0,
        "prmbench_pair_id": "",
    }
    if extra_info.get("benchmark_name") == "prmbench":
        result.update(
            {
                "prmbench_pair_id": str(extra_info["pair_id"]),
            }
        )

    try:
        raw = _extract_json_object(solution_str)
        pred_step_labels, pred_final_label = _normalize_output(raw, step_indices=step_indices)
    except Exception:
        reward_type = extra_info["reward_type"]
        result["score"] = result["ORM_score"] if reward_type == "outcome" else result["PRM_score"]
        return result

    pred_first_error = _first_error_index(pred_step_labels, step_indices)
    orm_score = float(pred_final_label == gt_final_label)
    prm_score = _step_accuracy(pred_step_labels, gt_step_labels)
    first_error_match = float(pred_first_error == gt_first_error)

    result["ORM_score"] = orm_score
    result["PRM_score"] = prm_score
    result["format"] = 1.0
    result["first_error_match"] = first_error_match
    if gt_first_error == -1:
        result["first_error_correct_match"] = first_error_match
    else:
        result["first_error_error_match"] = first_error_match
    if extra_info.get("benchmark_name") == "prmbench":
        result.update(
            _collect_prmbench_step_stats(
                pred_step_labels,
                gt_step_labels,
                step_indices=step_indices,
                gt_first_error=gt_first_error,
            )
        )

    reward_type = extra_info["reward_type"]
    if reward_type == "outcome":
        result["score"] = orm_score
    elif reward_type == "process":
        result["score"] = prm_score
    else:
        raise ValueError(f"reward_type must be 'outcome' or 'process', got {reward_type!r}")

    return result
