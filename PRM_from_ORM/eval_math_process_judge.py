import argparse
import json
import os
from collections import Counter

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from verl.utils.tokenizer import render_chat_prompt
from verl.utils.reward_score.math_process_judge import _first_error_index, _normalize_output
from verl.utils.reward_score.agent_process_bench import _extract_json_object


def load_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_processbench_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_steps(steps: list[str]) -> str:
    parts = []
    for idx, step in enumerate(steps):
        parts.append(f"<paragraph_{idx}>\n{step}\n</paragraph_{idx}>")
    return "\n\n".join(parts)


def prepare_messages(template: str, item: dict) -> list[dict[str, str]]:
    prompt = template.replace("__PROBLEM__", item["problem"]).replace("__TAGGED_RESPONSE__", format_steps(item["steps"]))
    return [{"role": "user", "content": prompt}]


def encode_prompt(tokenizer, messages):
    prompt = render_chat_prompt(tokenizer, messages, add_generation_prompt=True)
    return tokenizer(prompt, add_special_tokens=False).input_ids


def processbench_step_labels(num_steps: int, label: int) -> dict[str, int]:
    if label == -1:
        return {str(idx): 1 for idx in range(num_steps)}
    return {str(idx): int(idx < label) for idx in range(num_steps)}


def compute_f1(matches: list[bool], labels: list[int]) -> tuple[float, float, float]:
    error_hits = [int(match) for match, label in zip(matches, labels, strict=True) if label != -1]
    correct_hits = [int(match) for match, label in zip(matches, labels, strict=True) if label == -1]

    error_acc = sum(error_hits) / len(error_hits) * 100 if error_hits else 0.0
    correct_acc = sum(correct_hits) / len(correct_hits) * 100 if correct_hits else 0.0
    denom = error_acc + correct_acc
    f1 = 0.0 if denom == 0 else 2 * error_acc * correct_acc / denom
    return error_acc, correct_acc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["gsm8k", "math", "olympiadbench", "omnimath"],
        choices=["gsm8k", "math", "olympiadbench", "omnimath"],
    )
    parser.add_argument("--processbench_dir", type=str, default="./PRM_from_ORM/ProcessBench")
    parser.add_argument("--template_path", type=str, default="./PRM_from_ORM/templates/math_process_judge_prompt.txt")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_voting", action="store_true")
    parser.add_argument("--voting_n", type=int, default=8)
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path.rstrip("/"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    template = load_template(args.template_path)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        enable_prefix_caching=True,
        swap_space=16,
        max_num_seqs=20,
    )

    if not args.use_voting:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=58192, seed=42)
    else:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.voting_n, max_tokens=58192, seed=42)

    for config in args.configs:
        input_data = load_processbench_json(os.path.join(args.processbench_dir, f"{config}.json"))
        prompt_token_ids = [encode_prompt(tokenizer, prepare_messages(template, item)) for item in input_data]
        generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

        output_dir = os.path.join(args.output_dir, args.model_name if not args.use_voting else f"{args.model_name}_voting")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for idx, item in enumerate(input_data):
            gt_step_labels = processbench_step_labels(len(item["steps"]), int(item["label"]))
            step_indices = list(range(len(item["steps"])))

            if not args.use_voting:
                generation_text = generations[idx].outputs[0].text
                raw_text = generation_text
            else:
                candidates = [output.text for output in generations[idx].outputs]
                normalized_outputs = []
                for candidate in candidates:
                    try:
                        raw = _extract_json_object(candidate)
                        step_labels, final_label = _normalize_output(raw, step_indices=step_indices)
                        pred_first_error = _first_error_index(step_labels, step_indices)
                        normalized_outputs.append((pred_first_error, candidate, step_labels, final_label))
                    except Exception:
                        continue
                if normalized_outputs:
                    pred_first_error = Counter(pred for pred, *_ in normalized_outputs).most_common(1)[0][0]
                    raw_text = next(candidate for pred, candidate, *_ in normalized_outputs if pred == pred_first_error)
                else:
                    raw_text = candidates[0]

            pred_first_error = None
            pred_final_label = None
            try:
                raw = _extract_json_object(raw_text)
                pred_step_labels, pred_final_label = _normalize_output(raw, step_indices=step_indices)
                pred_first_error = _first_error_index(pred_step_labels, step_indices)
            except Exception:
                pred_step_labels = None

            gt_final_label = int(bool(item["final_answer_correct"]))
            match = pred_first_error == int(item["label"])
            final_match = pred_final_label == gt_final_label

            results.append(
                {
                    **item,
                    "prediction_first_error": pred_first_error,
                    "prediction_final_label": pred_final_label,
                    "match": match,
                    "final_match": final_match,
                    "generated_text": raw_text,
                    "gt_step_labels": gt_step_labels,
                    "pred_step_labels": pred_step_labels,
                }
            )

        with open(os.path.join(output_dir, f"{config}_math_process_judge.jsonl"), "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        error_acc, correct_acc, f1 = compute_f1([row["match"] for row in results], [int(row["label"]) for row in results])
        final_acc = sum(int(row["final_match"]) for row in results) / len(results) * 100
        print(f"{config} error acc: {error_acc:.1f}, correct acc: {correct_acc:.1f}, first-error f1: {f1:.1f}, final acc: {final_acc:.1f}")
