import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from PRM_from_ORM.prepare_math_process_judge_data import (
    load_processbench_rows,
    load_prmbench_rows,
    load_scan_pro_rows,
    load_template,
    save_rows,
    train_dev_split,
)


def test_prepare_math_process_judge_data_tiny_fixture(tmp_path):
    template_path = tmp_path / "template.txt"
    template_path.write_text("Problem:\n__PROBLEM__\n\nSolution:\n__TAGGED_RESPONSE__", encoding="utf-8")
    template = load_template(str(template_path))

    scan_path = tmp_path / "scan_pro.parquet"
    pd.DataFrame([
        {"question": "Compute 1+1.", "steps": ["1+1=2"], "scores": [1.0]},
        {"question": "Compute 2+2.", "steps": ["2+2=5"], "scores": [0.0]},
    ]).to_parquet(scan_path, index=False)

    processbench_dir = tmp_path / "ProcessBench"
    processbench_dir.mkdir()
    for name in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        (processbench_dir / f"{name}.json").write_text(json.dumps([
            {"problem": "Compute 1+1.", "steps": ["1+1=2"], "label": -1, "final_answer_correct": True}
        ]), encoding="utf-8")

    prmbench_path = tmp_path / "prmbench_preview.jsonl"
    prmbench_path.write_text(json.dumps({
        "classification": "redundency",
        "idx": "redundency_1",
        "original_question": "Compute 1+1.",
        "modified_question": "Compute 1+1.",
        "original_process": ["1+1=2"],
        "modified_process": ["1+1=3"],
        "modified_steps": [1],
        "error_steps": [1],
        "reason": "toy",
    }) + "\n", encoding="utf-8")

    scan_rows = load_scan_pro_rows(str(scan_path), template)
    train_rows, dev_rows = train_dev_split(scan_rows, dev_ratio=0.0, seed=42)
    processbench_rows = load_processbench_rows(str(processbench_dir), template, ["gsm8k", "math", "olympiadbench", "omnimath"])
    prmbench_rows = load_prmbench_rows(str(prmbench_path), template)

    assert len(train_rows) == 2
    assert len(dev_rows) == 0
    assert len(processbench_rows) == 4
    assert len(prmbench_rows) == 2
    assert train_rows[0]["data_source"] == "MathProcessJudge/scan_pro"
    assert processbench_rows[0]["data_source"].startswith("MathProcessJudge/processbench")

    output_path = tmp_path / "scan_pro_train.parquet"
    save_rows(train_rows, str(output_path))
    assert output_path.is_file()
