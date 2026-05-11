from __future__ import annotations

from typing import Any

from .agent_process_bench import _coerce_int_label, _extract_json_object, calculate_PRM_score


def _score_one(
    raw: dict[str, Any],
    *,
    trajectory_id: str,
    assistant_indices: list[int],
    final_label: int,
    step_labels: dict[str, Any],
) -> tuple[float, float]:
    trajectories = raw.get("trajectories")
    if not isinstance(trajectories, dict):
        raise ValueError("missing trajectories in judge output")

    traj = trajectories.get(str(trajectory_id))
    if not isinstance(traj, dict):
        raise ValueError(f"missing trajectories[{trajectory_id}] in judge output")

    if "step_labels" not in traj or "final_label" not in traj:
        raise ValueError(f"missing step_labels/final_label in trajectories[{trajectory_id}]")

    step_labels_raw = traj.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError(f"step_labels in trajectories[{trajectory_id}] must be an object")
    step_labels_raw = {str(k): v for k, v in step_labels_raw.items()}

    expected_keys = [str(i) for i in assistant_indices]
    predict_step_labels: dict[str, int] = {}
    for k in expected_keys:
        if k not in step_labels_raw:
            raise ValueError(f"missing trajectories[{trajectory_id}].step_labels[{k}]")
        predict_step_labels[k] = _coerce_int_label(step_labels_raw[k])

    predict_final_label = _coerce_int_label(traj.get("final_label"))

    # ORM: normalize gt (-1/0)->-1, 1->1
    if final_label in (-1, 0):
        target = -1
    elif final_label == 1:
        target = 1
    else:
        raise ValueError(f"unexpected final_label: {final_label}")
    orm_score = float(predict_final_label == target)

    # PRM: drop None labels and require exact assistant index keys
    gt_step_labels = {str(k): int(v) for k, v in step_labels.items() if v is not None}
    if gt_step_labels.keys() != set(expected_keys):
        raise ValueError("gt step_labels keys mismatch assistant_indices")
    prm_score = calculate_PRM_score(predict_step_labels, gt_step_labels)

    return prm_score, orm_score


def compute_score(solution_str: str, ground_truth: Any, *, extra_info: dict | None = None, **_) -> dict[str, Any]:
    del ground_truth
    assert isinstance(extra_info, dict)
    assert "trajectories" in extra_info
    assert "reward_type" in extra_info

    reward_type = extra_info["reward_type"]
    trajectories = extra_info["trajectories"]
    assert isinstance(trajectories, list) and trajectories, "extra_info['trajectories'] must be a non-empty list"

    result = {"score": -1.0, "ORM_score": -1.0, "PRM_score": -1.0, "format": -1}

    try:
        raw = _extract_json_object(solution_str)

        prm_scores: list[float] = []
        orm_scores: list[float] = []
        for traj in trajectories:
            assert isinstance(traj, dict)
            trajectory_id = str(traj.get("trajectory_id"))
            assistant_indices = list(traj.get("assistant_indices"))
            final_label = int(traj.get("final_label"))
            step_labels = traj.get("step_labels")
            assert isinstance(step_labels, dict), "step_labels must be an object"

            prm_score, orm_score = _score_one(
                raw,
                trajectory_id=trajectory_id,
                assistant_indices=assistant_indices,
                final_label=final_label,
                step_labels=step_labels,
            )
            prm_scores.append(prm_score)
            orm_scores.append(orm_score)

        result["format"] = 1
        result["ORM_score"] = sum(orm_scores) / len(orm_scores)
        result["PRM_score"] = sum(prm_scores) / len(prm_scores)
    except Exception:
        return result

    if reward_type == "outcome":
        result["score"] = result["ORM_score"]
    elif reward_type == "process":
        result["score"] = result["PRM_score"]
    else:
        raise ValueError(f"reward_type must be 'outcome' or 'process', got {reward_type!r}")

    return result