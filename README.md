# TIPS: PRM from ORM Minimal Release

This branch is a minimal, runnable export of the PRM-from-ORM code from `/Users/dada/Desktop/verl`.
It keeps the full `verl/` runtime from source commit `000c8e56` and overlays only the PRM-related working-tree changes required by the math and agent examples.

## What is included

- `verl/`: full runtime needed by `python -m verl.trainer.main_ppo`.
- `PRM_from_ORM/prepare_math_process_judge_data.py`: tiny math/process judge data conversion.
- `PRM_from_ORM/prepare_agent_process_judge_data.py`: tiny AgentProcessBench-style data conversion.
- `PRM_from_ORM/run_math_process_judge_grpo.sh`: parameterized GRPO launch script.
- `PRM_from_ORM/run_agent_process_judge_grpo.sh`: parameterized GRPO launch script.
- `PRM_from_ORM/examples/`: tiny local fixtures for CPU sanity checks.
- `tests/PRM_from_ORM/`: CPU-only tests for the exported path.

Large generated data, checkpoints, logs, cache directories, private paths, and API keys are intentionally not included.

## CPU sanity checks

Install local test dependencies in your environment, then run:

```bash
pip install -e .[test]
pytest -q tests/PRM_from_ORM
```

Prepare the tiny math fixture:

```bash
python PRM_from_ORM/prepare_math_process_judge_data.py   --scan_pro_path PRM_from_ORM/examples/math_tiny/scan_pro.parquet   --processbench_dir PRM_from_ORM/examples/math_tiny/ProcessBench   --prmbench_path PRM_from_ORM/examples/math_tiny/prmbench_preview.jsonl   --template_path PRM_from_ORM/templates/math_process_judge_prompt.txt   --output_dir /tmp/tips_math_process_judge   --dev_ratio 0.0
```

Prepare the tiny agent fixture:

```bash
python PRM_from_ORM/prepare_agent_process_judge_data.py   --source_dir PRM_from_ORM/examples/agent_tiny/processes_agent_process_judge   --output_dir /tmp/tips_agent_process_judge
```

## GRPO launch

Training requires the normal verl GPU stack, a model, and generated parquet files. The scripts fail fast if `MODEL_PATH` or `TRAIN_FILES` are missing.

Example math launch shape:

```bash
MODEL_PATH=/path/to/model DATA_DIR=/tmp/tips_math_process_judge OUTPUT_DIR=/tmp/tips_math_ckpts LOGGER='["console"]' bash PRM_from_ORM/run_math_process_judge_grpo.sh trainer.val_only=True trainer.val_before_train=True
```

Example agent launch shape:

```bash
MODEL_PATH=/path/to/model DATA_DIR=/tmp/tips_agent_process_judge OUTPUT_DIR=/tmp/tips_agent_ckpts LOGGER='["console"]' bash PRM_from_ORM/run_agent_process_judge_grpo.sh trainer.val_only=True trainer.val_before_train=True
```
