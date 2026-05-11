import json
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def _load_agent_process_bench_module():
    path = os.path.join(REPO_ROOT, "verl", "utils", "reward_score", "agent_process_bench.py")
    spec = importlib.util.spec_from_file_location("agent_process_bench_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compute_score = _load_agent_process_bench_module().compute_score


def _solution(step_labels, final_label=1):
    payload = {
        "step_labels": {str(k): v for k, v in step_labels.items()},
        "final_label": final_label,
        "explanations": {
            "steps": {str(k): "ok" for k in step_labels},
            "final": "done",
        },
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def test_outcome_reward_does_not_require_ground_truth_step_labels():
    result = compute_score(
        _solution({2: 1}, final_label=0),
        ground_truth=-1,
        extra_info={
            "assistant_indices": [2],
            "reward_type": "outcome",
            "final_label": 0,
            "step_labels": None,
        },
    )

    assert result["score"] == 1.0
    assert result["ORM_score"] == 1.0
    assert result["final_acc"] == 1.0
    assert result["PRM_score"] == -1.0
    assert result["agent_step_total"] == 0.0
    assert result["agent_first_neg1_total"] == 0.0


def test_eval_metrics_use_step_micro_counts_and_first_neg1_index():
    result = compute_score(
        _solution({2: 1, 4: -1}, final_label=1),
        ground_truth=1,
        extra_info={
            "assistant_indices": [2, 4],
            "reward_type": "outcome",
            "final_label": 1,
            "step_labels": {"2": 1, "4": -1},
        },
    )

    assert result["score"] == 1.0
    assert result["PRM_score"] == 1.0
    assert result["agent_step_match"] == 2.0
    assert result["agent_step_total"] == 2.0
    assert result["agent_first_neg1_match"] == 1.0
    assert result["agent_first_neg1_total"] == 1.0


def test_first_neg1_index_all_correct_is_minus_one():
    result = compute_score(
        _solution({2: 1, 4: 1}, final_label=1),
        ground_truth=1,
        extra_info={
            "assistant_indices": [2, 4],
            "reward_type": "outcome",
            "final_label": 1,
            "step_labels": {"2": 1, "4": 1},
        },
    )

    assert result["agent_first_neg1_match"] == 1.0
    assert result["agent_first_neg1_total"] == 1.0


def test_bad_json_is_format_failure_and_counts_eval_totals_as_wrong():
    result = compute_score(
        "not json",
        ground_truth=1,
        extra_info={
            "assistant_indices": [2, 4],
            "reward_type": "outcome",
            "final_label": 1,
            "step_labels": {"2": 1, "4": -1},
        },
    )

    assert result["score"] == 0.0
    assert result["format"] == 0.0
    assert result["agent_step_match"] == 0.0
    assert result["agent_step_total"] == 2.0
    assert result["agent_first_neg1_match"] == 0.0
    assert result["agent_first_neg1_total"] == 1.0


def test_process_reward_requires_ground_truth_step_labels():
    with pytest.raises(ValueError, match="ground-truth step_labels are required"):
        compute_score(
            _solution({2: 1}, final_label=1),
            ground_truth=1,
            extra_info={
                "assistant_indices": [2],
                "reward_type": "process",
                "final_label": 1,
                "step_labels": None,
            },
        )
