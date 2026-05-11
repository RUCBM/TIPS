from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DATASETS = ("bfcl", "gaia_dev", "hotpotqa", "tau2")
EVAL_MANIFEST_NAME = "agentprocessbench_eval_manifest.json"
DEFAULT_SYSTEM_TEMPLATE_PATH = Path("PRM_from_ORM/templates/agent_process_judge_system_prompt.txt")
DEFAULT_USER_TEMPLATE_PATH = Path("PRM_from_ORM/templates/agent_process_judge_user_prompt.txt")


def load_template(template_path: Path) -> str:
    assert template_path.is_file(), f"missing prompt template: {template_path}"
    template = template_path.read_text(encoding="utf-8").strip()
    assert template, f"empty prompt template: {template_path}"
    return template


def load_prompt_templates(system_template_path: Path, user_template_path: Path) -> tuple[str, str]:
    system_template = load_template(system_template_path)
    user_template = load_template(user_template_path)
    assert "__TRAJECTORY_JSON__" in user_template, "user template must contain __TRAJECTORY_JSON__"
    return system_template, user_template


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"missing jsonl file: {path}"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict), f"{path}:{line_no}: expected JSON object"
            rows.append(obj)
    assert rows, f"no rows loaded from {path}"
    return rows


def _assistant_message_indices(messages: Any) -> list[int]:
    assert isinstance(messages, list), "messages must be a list"
    indices: list[int] = []
    for idx, msg in enumerate(messages):
        assert isinstance(msg, dict), f"messages[{idx}] must be an object"
        if msg.get("role") == "assistant":
            indices.append(idx)
    assert indices, "each trajectory must contain at least one assistant message"
    return indices


def normalize_final_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"final_label must be -1/0/1, got bool: {value!r}")
    if isinstance(value, int):
        label = value
    elif isinstance(value, float) and value.is_integer():
        label = int(value)
    elif isinstance(value, str) and value.strip() in {"-1", "0", "1"}:
        label = int(value.strip())
    else:
        raise ValueError(f"final_label must be -1/0/1, got {value!r}")
    if label in (-1, 0):
        return -1
    if label == 1:
        return 1
    raise ValueError(f"final_label must be -1/0/1, got {value!r}")


def _normalize_step_labels(value: Any, assistant_indices: list[int], *, required: bool) -> dict[str, int] | None:
    if value is None:
        assert not required, "eval rows must contain step_labels"
        return None
    assert isinstance(value, dict), f"step_labels must be a dict, got {type(value)!r}"
    normalized = {str(k): int(v) for k, v in value.items() if v is not None}
    expected = {str(idx) for idx in assistant_indices}
    assert normalized.keys() == expected, (
        "step_labels keys must exactly match assistant indices: "
        f"got={sorted(normalized.keys())}, expected={sorted(expected)}"
    )
    for key, label in normalized.items():
        assert label in (-1, 0, 1), f"step_labels[{key}] must be -1/0/1, got {label!r}"
    return normalized


def first_neg1_index(step_labels: dict[str, int] | None) -> int:
    if not step_labels:
        return -1
    wrong_indices = [int(key) for key, value in step_labels.items() if value == -1]
    return min(wrong_indices) if wrong_indices else -1


def build_judge_prompt(
    item: dict[str, Any],
    assistant_indices: list[int],
    *,
    system_template: str,
    user_template: str,
) -> list[dict[str, str]]:
    assert system_template.strip(), "system_template must be non-empty"
    assert "__TRAJECTORY_JSON__" in user_template, "user_template must contain __TRAJECTORY_JSON__"
    messages = item.get("messages")
    assert isinstance(messages, list), "item['messages'] must be a list"
    assert all(0 <= idx < len(messages) for idx in assistant_indices), "assistant index out of range"
    assert all(messages[idx].get("role") == "assistant" for idx in assistant_indices), (
        "assistant_indices must point to assistant messages"
    )

    payload: dict[str, Any] = {
        "question": item.get("question"),
        "task_description": item.get("task_description"),
        "tools": item.get("tools"),
        "messages": list(enumerate(messages)),
        "assistant_message_indices": assistant_indices,
        "notes": {
            "step_definition": "Each Step == one message with role=='assistant'. Use the given indices.",
            "output_requirements": "Return JSON with step_labels, final_label, explanations.",
        },
    }
    trajectory_json = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system_template},
        {
            "role": "user",
            "content": user_template.replace("__TRAJECTORY_JSON__", trajectory_json),
        },
    ]


def convert_record(
    item: dict[str, Any],
    *,
    dataset: str,
    split: str,
    require_step_labels: bool,
    system_template: str,
    user_template: str,
) -> dict[str, Any]:
    assert split in {"train", "val"}, f"unexpected split: {split}"
    required_keys = {"messages", "final_label", "question", "data_source", "total_index"}
    missing = required_keys - set(item)
    assert not missing, f"missing required keys: {sorted(missing)}"

    assistant_indices = _assistant_message_indices(item["messages"])
    step_labels = _normalize_step_labels(item.get("step_labels"), assistant_indices, required=require_step_labels)
    final_label = normalize_final_label(item["final_label"])
    prompt = build_judge_prompt(
        item,
        assistant_indices,
        system_template=system_template,
        user_template=user_template,
    )

    data_source = f"AgentProcessBench/{dataset}"
    extra_info = {
        "split": split,
        "assistant_indices": assistant_indices,
        "reward_type": "outcome",
        "dataset_name": dataset,
        "ori_data_source": item["data_source"],
        "index": item["total_index"],
        "total_index": item["total_index"],
        "query_index": item.get("query_index"),
        "sample_index": item.get("sample_index"),
        "final_label": final_label,
        "step_labels": step_labels,
        "first_neg1_index": first_neg1_index(step_labels),
        "question": item["question"],
        "need_tools_kwargs": False,
    }
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "agent_process_judge_orm",
        "reward_model": {
            "style": "rule",
            "ground_truth": final_label,
        },
        "extra_info": extra_info,
    }


def _write_parquet(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    rows = list(rows)
    assert rows, f"refusing to write empty parquet: {output_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path)
    assert output_path.is_file(), f"failed to write parquet: {output_path}"
    return len(df)


def prepare_agent_process_judge_data(
    source_dir: Path,
    output_dir: Path,
    *,
    system_template_path: Path = DEFAULT_SYSTEM_TEMPLATE_PATH,
    user_template_path: Path = DEFAULT_USER_TEMPLATE_PATH,
) -> dict[str, int]:
    train_dir = source_dir / "agent_train_orm"
    eval_dir = source_dir / "AgentProcessBench"
    assert train_dir.is_dir(), f"missing train dir: {train_dir}"
    assert eval_dir.is_dir(), f"missing eval dir: {eval_dir}"
    system_template, user_template = load_prompt_templates(system_template_path, user_template_path)

    counts: dict[str, int] = {}
    train_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = train_dir / f"{dataset}.jsonl"
        rows = _read_jsonl(path)
        converted = [
            convert_record(
                row,
                dataset=dataset,
                split="train",
                require_step_labels=False,
                system_template=system_template,
                user_template=user_template,
            )
            for row in rows
        ]
        counts[f"train/{dataset}"] = len(converted)
        train_rows.extend(converted)
    counts["train/total"] = _write_parquet(train_rows, output_dir / "agent_train_orm.parquet")

    manifest: dict[str, Any] = {"version": 1, "datasets": {}}
    for dataset in DATASETS:
        path = eval_dir / dataset / "test.jsonl"
        rows = _read_jsonl(path)
        converted = [
            convert_record(
                row,
                dataset=dataset,
                split="val",
                require_step_labels=True,
                system_template=system_template,
                user_template=user_template,
            )
            for row in rows
        ]
        output_name = f"agentprocessbench_{dataset}_eval.parquet"
        counts[f"eval/{dataset}"] = _write_parquet(
            converted,
            output_dir / output_name,
        )
        step_total = sum(len(row["extra_info"]["step_labels"]) for row in converted)
        manifest["datasets"][f"AgentProcessBench/{dataset}"] = {
            "dataset": dataset,
            "parquet": output_name,
            "sample_total": len(converted),
            "step_total": step_total,
        }
    manifest_path = output_dir / EVAL_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert manifest_path.is_file(), f"failed to write eval manifest: {manifest_path}"
    return counts


def _percentile(sorted_values: list[int], pct: float) -> int:
    assert sorted_values, "cannot compute percentile for empty values"
    assert 0 <= pct <= 100, f"pct must be in [0, 100], got {pct}"
    if pct == 100:
        return sorted_values[-1]
    idx = int(round((pct / 100) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def summarize_prompt_token_lengths(
    output_dir: Path,
    tokenizer_path: Path,
    *,
    max_samples: int | None = None,
) -> dict[str, int]:
    from transformers import AutoTokenizer

    assert tokenizer_path.exists(), f"tokenizer_path does not exist: {tokenizer_path}"
    if max_samples is not None:
        assert max_samples > 0, f"max_samples must be positive, got {max_samples}"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    parquet_paths = [output_dir / "agent_train_orm.parquet"] + [
        output_dir / f"agentprocessbench_{dataset}_eval.parquet" for dataset in DATASETS
    ]
    lengths: list[int] = []
    for parquet_path in parquet_paths:
        assert parquet_path.is_file(), f"missing parquet file for length stats: {parquet_path}"
        df = pd.read_parquet(parquet_path, columns=["prompt"])
        for prompt in df["prompt"]:
            token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
            lengths.append(len(token_ids))
            if max_samples is not None and len(lengths) >= max_samples:
                break
        if max_samples is not None and len(lengths) >= max_samples:
            break

    lengths.sort()
    return {
        "count": len(lengths),
        "p50": _percentile(lengths, 50),
        "p95": _percentile(lengths, 95),
        "p99": _percentile(lengths, 99),
        "max": lengths[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AgentProcessBench ORM judge data for verl GRPO.")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("PRM_from_ORM/examples/agent_tiny/processes_agent_process_judge"),
        help="Directory containing agent_train_orm/ and AgentProcessBench/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("PRM_from_ORM/processed_agent_process_judge"),
        help="Directory to write parquet files.",
    )
    parser.add_argument(
        "--system_template_path",
        type=Path,
        default=DEFAULT_SYSTEM_TEMPLATE_PATH,
        help="System prompt template path.",
    )
    parser.add_argument(
        "--user_template_path",
        type=Path,
        default=DEFAULT_USER_TEMPLATE_PATH,
        help="User prompt template path. Must contain __TRAJECTORY_JSON__.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=None,
        help="Optional tokenizer path for prompt token length stats after parquet generation.",
    )
    parser.add_argument(
        "--length_stats_max_samples",
        type=int,
        default=None,
        help="Optional cap for tokenizer length stats samples.",
    )
    args = parser.parse_args()

    counts = prepare_agent_process_judge_data(
        args.source_dir,
        args.output_dir,
        system_template_path=args.system_template_path,
        user_template_path=args.user_template_path,
    )
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")

    if args.tokenizer_path is not None:
        stats = summarize_prompt_token_lengths(
            args.output_dir,
            args.tokenizer_path,
            max_samples=args.length_stats_max_samples,
        )
        print(
            "prompt_token_lengths: "
            f"count={stats['count']} p50={stats['p50']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']}"
        )


if __name__ == "__main__":
    main()
