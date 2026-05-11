import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


SCAN_PRO_DATA_SOURCE = "MathProcessJudge/scan_pro"
PROCESSBENCH_DATA_SOURCE_PREFIX = "MathProcessJudge/processbench"
DEFAULT_PROCESSBENCH_CONFIGS = ("gsm8k", "math", "olympiadbench", "omnimath")


def load_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_steps(steps: list[str]) -> str:
    parts = []
    for idx, step in enumerate(steps):
        parts.append(f"<paragraph_{idx}>\n{step}\n</paragraph_{idx}>")
    return "\n\n".join(parts)


def build_prompt(template: str, problem: str, steps: list[str]) -> list[dict[str, str]]:
    content = template.replace("__PROBLEM__", problem).replace("__TAGGED_RESPONSE__", format_steps(steps))
    return [{"role": "user", "content": content}]


def first_error_index_from_step_labels(step_labels: dict[str, int]) -> int:
    for idx in range(len(step_labels)):
        if step_labels[str(idx)] == 0:
            return idx
    return -1


def scan_pro_step_labels(scores) -> dict[str, int]:
    return {str(idx): int(float(score) == 1.0) for idx, score in enumerate(scores)}


def processbench_step_labels(num_steps: int, label: int) -> dict[str, int]:
    if label == -1:
        return {str(idx): 1 for idx in range(num_steps)}
    return {str(idx): int(idx < label) for idx in range(num_steps)}


def build_rl_row(
    *,
    data_source: str,
    split: str,
    index: int,
    problem: str,
    steps: list[str],
    step_labels: dict[str, int],
    final_label: int,
    template: str,
    dataset_name: str,
    reward_type: str = "outcome",
    extra_info: dict | None = None,
) -> dict:
    step_indices = list(range(len(steps)))
    row_extra_info = {
        "split": split,
        "index": index,
        "dataset_name": dataset_name,
        "question": problem,
        "steps": steps,
        "step_indices": step_indices,
        "step_labels": step_labels,
        "final_label": int(final_label),
        "first_error_index": first_error_index_from_step_labels(step_labels),
        "reward_type": reward_type,
    }
    if extra_info is not None:
        row_extra_info.update(extra_info)
    return {
        "data_source": data_source,
        "prompt": build_prompt(template, problem, steps),
        "ability": "math_process_judge",
        "reward_model": {"style": "rule", "ground_truth": int(final_label)},
        "extra_info": row_extra_info,
    }


def load_processbench_rows(processbench_dir: str, template: str, configs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        path = Path(processbench_dir) / f"{config}.json"
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for idx, item in enumerate(items):
            steps = [str(step) for step in item["steps"]]
            step_labels = processbench_step_labels(len(steps), int(item["label"]))
            final_label = int(bool(item["final_answer_correct"]))
            rows.append(
                build_rl_row(
                    data_source=f"{PROCESSBENCH_DATA_SOURCE_PREFIX}/{config}",
                    split="eval",
                    index=idx,
                    problem=str(item["problem"]),
                    steps=steps,
                    step_labels=step_labels,
                    final_label=final_label,
                    template=template,
                    dataset_name=config,
                )
            )
    return rows


def load_scan_pro_rows(scan_pro_path: str, template: str) -> list[dict]:
    df = pd.read_parquet(scan_pro_path)
    rows: list[dict] = []
    for idx, row in df.iterrows():
        steps = [str(step) for step in row["steps"]]
        step_labels = scan_pro_step_labels(row["scores"])
        final_label = int(float(row["scores"][-1]) == 1.0)
        rows.append(
            build_rl_row(
                data_source=SCAN_PRO_DATA_SOURCE,
                split="train",
                index=int(idx),
                problem=str(row["question"]),
                steps=steps,
                step_labels=step_labels,
                final_label=final_label,
                template=template,
                dataset_name="scan_pro",
            )
        )
    return rows


def train_dev_split(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(len(indices) * (1 - dev_ratio))
    train_indices = set(indices[:split])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for idx, row in enumerate(rows):
        split_name = "train" if idx in train_indices else "dev"
        row = json.loads(json.dumps(row))
        row["extra_info"]["split"] = split_name
        if split_name == "train":
            train_rows.append(row)
        else:
            dev_rows.append(row)
    return train_rows, dev_rows


def save_rows(rows: list[dict], output_path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_pro_path",
        default="./PRM_from_ORM/examples/math_tiny/scan_pro.parquet",
        help="Local path to scan_pro parquet.",
    )
    parser.add_argument(
        "--processbench_dir",
        default="./PRM_from_ORM/examples/math_tiny/ProcessBench",
        help="Directory containing ProcessBench json files.",
    )
    parser.add_argument(
        "--template_path",
        default="./PRM_from_ORM/templates/math_process_judge_prompt.txt",
        help="Prompt template path.",
    )
    parser.add_argument(
        "--output_dir",
        default="./PRM_from_ORM/processed_math_process_judge",
        help="Where to save generated parquet files.",
    )
    parser.add_argument(
        "--processbench_configs",
        nargs="+",
        default=list(DEFAULT_PROCESSBENCH_CONFIGS),
        choices=list(DEFAULT_PROCESSBENCH_CONFIGS),
    )
    parser.add_argument("--dev_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    template = load_template(args.template_path)

    scan_rows = load_scan_pro_rows(args.scan_pro_path, template)
    scan_train_rows, scan_dev_rows = train_dev_split(scan_rows, dev_ratio=args.dev_ratio, seed=args.seed)
    processbench_rows = load_processbench_rows(args.processbench_dir, template, configs=args.processbench_configs)

    scan_train_path = os.path.join(args.output_dir, "scan_pro_train.parquet")
    scan_dev_path = os.path.join(args.output_dir, "scan_pro_dev.parquet")
    processbench_eval_path = os.path.join(args.output_dir, "processbench_eval.parquet")

    save_rows(scan_train_rows, scan_train_path)
    save_rows(scan_dev_rows, scan_dev_path)
    save_rows(processbench_rows, processbench_eval_path)

    print(f"scan_pro_train: {scan_train_path} ({len(scan_train_rows)})")
    print(f"scan_pro_dev: {scan_dev_path} ({len(scan_dev_rows)})")
    print(f"processbench_eval: {processbench_eval_path} ({len(processbench_rows)})")
