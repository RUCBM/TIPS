import importlib.util
import json
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_math_process_judge_module():
    package_names = ("verl", "verl.utils", "verl.utils.reward_score")
    module_names = package_names + (
        "verl.utils.reward_score.agent_process_bench",
        "verl.utils.reward_score.math_process_judge",
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        for name in package_names:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

        agent_path = os.path.join(REPO_ROOT, "verl", "utils", "reward_score", "agent_process_bench.py")
        agent_spec = importlib.util.spec_from_file_location("verl.utils.reward_score.agent_process_bench", agent_path)
        assert agent_spec is not None and agent_spec.loader is not None
        agent_module = importlib.util.module_from_spec(agent_spec)
        sys.modules[agent_spec.name] = agent_module
        agent_spec.loader.exec_module(agent_module)

        math_path = os.path.join(REPO_ROOT, "verl", "utils", "reward_score", "math_process_judge.py")
        math_spec = importlib.util.spec_from_file_location("verl.utils.reward_score.math_process_judge", math_path)
        assert math_spec is not None and math_spec.loader is not None
        math_module = importlib.util.module_from_spec(math_spec)
        sys.modules[math_spec.name] = math_module
        math_spec.loader.exec_module(math_module)
        return math_module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


compute_score = _load_math_process_judge_module().compute_score


def _solution(step_labels, final_label=1):
    payload = {
        "step_labels": {str(k): v for k, v in step_labels.items()},
        "final_label": final_label,
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def test_math_process_judge_scores_outcome_and_first_error_without_removed_benchmark_fields():
    result = compute_score(
        _solution({1: 1, 2: 0}, final_label=0),
        ground_truth="unused",
        extra_info={
            "step_indices": [1, 2],
            "step_labels": {"1": 1, "2": 0},
            "final_label": 0,
            "reward_type": "outcome",
        },
    )

    assert result["score"] == 1.0
    assert result["ORM_score"] == 1.0
    assert result["PRM_score"] == 1.0
    assert result["first_error_match"] == 1.0
    removed_prefix = "prm" + "bench"
    assert not any(removed_prefix in key.lower() for key in result)
