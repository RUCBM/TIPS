import json
import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from PRM_from_ORM.prepare_agent_process_judge_data import (
    DATASETS,
    convert_record,
    prepare_agent_process_judge_data,
)


def _record(*, step_labels=None, final_label=1):
    return {
        "total_index": 7,
        "query_index": 1,
        "sample_index": 2,
        "question": "What is the answer?",
        "data_source": "toy_source",
        "ground_truth": {"answer": "ok"},
        "answer_text": "ok",
        "task_description": "Toy task",
        "tools": [],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "step_labels": step_labels,
        "final_label": final_label,
    }


def test_train_outcome_only_record_allows_missing_step_labels():
    row = convert_record(
        _record(step_labels=None, final_label=0),
        dataset="bfcl",
        split="train",
        require_step_labels=False,
    )

    assert row["data_source"] == "AgentProcessBench/bfcl"
    assert row["reward_model"]["ground_truth"] == -1
    assert row["extra_info"]["step_labels"] is None
    assert row["extra_info"]["first_neg1_index"] == -1
    assert row["prompt"][0]["role"] == "system"
    assert "TRAJECTORY_JSON" in row["prompt"][1]["content"]


def test_eval_record_requires_step_labels():
    with pytest.raises(AssertionError, match="eval rows must contain step_labels"):
        convert_record(_record(step_labels=None), dataset="bfcl", split="val", require_step_labels=True)


def test_prepare_agent_process_judge_data_writes_expected_parquets(tmp_path):
    source_dir = tmp_path / "processes_agent_process_judge"
    train_dir = source_dir / "agent_train_orm"
    eval_dir = source_dir / "AgentProcessBench"
    train_dir.mkdir(parents=True)
    for dataset in DATASETS:
        train_row = _record(step_labels=None, final_label=1)
        (train_dir / f"{dataset}.jsonl").write_text(json.dumps(train_row) + "\n", encoding="utf-8")

        eval_dataset_dir = eval_dir / dataset
        eval_dataset_dir.mkdir(parents=True)
        eval_row = _record(step_labels={"2": -1}, final_label=-1)
        (eval_dataset_dir / "test.jsonl").write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    output_dir = tmp_path / "processed_agent_process_judge"
    counts = prepare_agent_process_judge_data(source_dir, output_dir)

    assert counts["train/total"] == len(DATASETS)
    train_df = pd.read_parquet(output_dir / "agent_train_orm.parquet")
    assert len(train_df) == len(DATASETS)
    manifest = json.loads((output_dir / "agentprocessbench_eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"]["AgentProcessBench/bfcl"]["sample_total"] == 1
    assert manifest["datasets"]["AgentProcessBench/bfcl"]["step_total"] == 1
    for dataset in DATASETS:
        eval_path = output_dir / f"agentprocessbench_{dataset}_eval.parquet"
        eval_df = pd.read_parquet(eval_path)
        assert len(eval_df) == 1
        assert eval_df.iloc[0]["data_source"] == f"AgentProcessBench/{dataset}"
