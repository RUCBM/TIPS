import importlib.util
import os
import sys
import types

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def _load_metric_utils_module():
    verl_stub = types.ModuleType("verl")
    verl_stub.DataProto = object
    utils_stub = types.ModuleType("verl.utils")
    import_utils_stub = types.ModuleType("verl.utils.import_utils")

    def deprecated(_replacement):
        return lambda fn: fn

    import_utils_stub.deprecated = deprecated
    old_modules = {name: sys.modules.get(name) for name in ("verl", "verl.utils", "verl.utils.import_utils")}
    sys.modules["verl"] = verl_stub
    sys.modules["verl.utils"] = utils_stub
    sys.modules["verl.utils.import_utils"] = import_utils_stub
    try:
        path = os.path.join(REPO_ROOT, "verl", "trainer", "ppo", "metric_utils.py")
        spec = importlib.util.spec_from_file_location("metric_utils_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


metric_utils = _load_metric_utils_module()
collect_agent_process_bench_metrics_by_data_source = metric_utils.collect_agent_process_bench_metrics_by_data_source
group_validation_samples_by_data_source = metric_utils.group_validation_samples_by_data_source
is_agent_process_bench_source = metric_utils.is_agent_process_bench_source


def test_agent_process_bench_metrics_are_count_micro_averages():
    data_sources = np.asarray(
        [
            "AgentProcessBench/bfcl",
            "AgentProcessBench/bfcl",
            "AgentProcessBench/tau2",
            "other",
        ],
        dtype=object,
    )
    sample_uids = np.asarray(["a", "b", "c", "d"], dtype=object)
    infos = {
        "agent_step_match": [1.0, 2.0, 3.0, 100.0],
        "agent_step_total": [2.0, 2.0, 4.0, 100.0],
        "agent_first_neg1_match": [0.0, 1.0, 1.0, 100.0],
        "agent_first_neg1_total": [1.0, 1.0, 1.0, 100.0],
    }

    manifest = {
        "AgentProcessBench/bfcl": {"sample_total": 3.0, "step_total": 6.0},
        "AgentProcessBench/tau2": {"sample_total": 1.0, "step_total": 4.0},
        "AgentProcessBench/AVG": {"sample_total": 4.0, "step_total": 10.0},
    }
    metrics = collect_agent_process_bench_metrics_by_data_source(data_sources, infos, manifest_by_source=manifest)

    assert metrics["AgentProcessBench/bfcl"]["StepAcc_kept"] == 0.75
    assert metrics["AgentProcessBench/bfcl"]["FirstErrAcc_kept"] == 0.5
    assert metrics["AgentProcessBench/bfcl"]["StepAcc_all"] == 0.5
    assert metrics["AgentProcessBench/bfcl"]["FirstErrAcc_all"] == 1.0 / 3.0
    assert metrics["AgentProcessBench/tau2"]["StepAcc_kept"] == 0.75
    assert "other" not in metrics

    grouped_sources, _, grouped_infos = group_validation_samples_by_data_source(
        data_sources,
        sample_uids,
        infos,
        source_to_group=lambda source: "AgentProcessBench/AVG" if is_agent_process_bench_source(source) else None,
    )
    metrics.update(
        collect_agent_process_bench_metrics_by_data_source(
            grouped_sources,
            grouped_infos,
            manifest_by_source=manifest,
        )
    )

    assert metrics["AgentProcessBench/AVG"]["StepAcc_kept"] == 0.75
    assert metrics["AgentProcessBench/AVG"]["FirstErrAcc_kept"] == 2.0 / 3.0
    assert metrics["AgentProcessBench/AVG"]["StepAcc_all"] == 0.6
    assert metrics["AgentProcessBench/AVG"]["FirstErrAcc_all"] == 0.5


def test_agent_process_bench_source_predicate_does_not_match_group_name():
    assert is_agent_process_bench_source("AgentProcessBench")
    assert is_agent_process_bench_source("AgentProcessBench/bfcl")
    assert not is_agent_process_bench_source("AgentProcessBench_GROUP")
