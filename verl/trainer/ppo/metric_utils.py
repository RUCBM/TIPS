# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metrics related to the PPO trainer."""

import json
import logging
import os
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PRMBENCH_DATA_SOURCE_PREFIX = "MathProcessJudge/prmbench"
PRMBENCH_CORRECT_SOURCE = f"{PRMBENCH_DATA_SOURCE_PREFIX}/correct"
AGENT_PROCESS_BENCH_DATA_SOURCE_PREFIX = "AgentProcessBench"
AGENT_PROCESS_BENCH_EVAL_MANIFEST_NAME = "agentprocessbench_eval_manifest.json"
PRMBENCH_CLASSIFICATION_GROUPS = {
    "simplicity": ("redundency", "circular"),
    "soundness": ("counterfactual", "step_contradiction", "domain_inconsistency", "confidence"),
    "sensitivity": ("missing_condition", "deception", "multi_solutions"),
}


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def summarize_two_partition_f1_counts(
    partition_a_match: float,
    partition_a_total: float,
    partition_b_match: float,
    partition_b_total: float,
    *,
    prefix: str,
    partition_a_name: str,
    partition_b_name: str,
) -> dict[str, float]:
    partition_a_acc = partition_a_match / partition_a_total if partition_a_total > 0 else 0.0
    partition_b_acc = partition_b_match / partition_b_total if partition_b_total > 0 else 0.0
    denom = partition_a_acc + partition_b_acc
    f1 = 0.0 if denom == 0 else 2 * partition_a_acc * partition_b_acc / denom
    return {
        f"{prefix}/{partition_a_name}_acc": float(partition_a_acc),
        f"{prefix}/{partition_b_name}_acc": float(partition_b_acc),
        f"{prefix}/f1": float(f1),
    }


def collect_two_partition_f1_metrics(
    infos_dict: dict[str, list[Any]],
    *,
    partition_a_match_key: str,
    partition_a_total_key: str,
    partition_b_match_key: str,
    partition_b_total_key: str,
    prefix: str,
    partition_a_name: str,
    partition_b_name: str,
) -> dict[str, float]:
    required = (
        partition_a_match_key,
        partition_a_total_key,
        partition_b_match_key,
        partition_b_total_key,
    )
    if any(key not in infos_dict for key in required):
        return {}

    return summarize_two_partition_f1_counts(
        partition_a_match=float(np.sum(np.asarray(infos_dict[partition_a_match_key], dtype=np.float64))),
        partition_a_total=float(np.sum(np.asarray(infos_dict[partition_a_total_key], dtype=np.float64))),
        partition_b_match=float(np.sum(np.asarray(infos_dict[partition_b_match_key], dtype=np.float64))),
        partition_b_total=float(np.sum(np.asarray(infos_dict[partition_b_total_key], dtype=np.float64))),
        prefix=prefix,
        partition_a_name=partition_a_name,
        partition_b_name=partition_b_name,
    )


def collect_two_partition_f1_metrics_by_data_source(
    data_sources: np.ndarray,
    infos_dict: dict[str, list[Any]],
    *,
    partition_a_match_key: str,
    partition_a_total_key: str,
    partition_b_match_key: str,
    partition_b_total_key: str,
    prefix: str,
    partition_a_name: str,
    partition_b_name: str,
) -> dict[str, dict[str, float]]:
    required = (
        partition_a_match_key,
        partition_a_total_key,
        partition_b_match_key,
        partition_b_total_key,
    )
    if any(key not in infos_dict for key in required):
        return {}

    data_sources = np.asarray(data_sources, dtype=object)
    partition_a_match = np.asarray(infos_dict[partition_a_match_key], dtype=np.float64)
    partition_a_total = np.asarray(infos_dict[partition_a_total_key], dtype=np.float64)
    partition_b_match = np.asarray(infos_dict[partition_b_match_key], dtype=np.float64)
    partition_b_total = np.asarray(infos_dict[partition_b_total_key], dtype=np.float64)

    metrics_by_source: dict[str, dict[str, float]] = {}
    for source in np.unique(data_sources):
        mask = data_sources == source
        metrics_by_source[str(source)] = summarize_two_partition_f1_counts(
            partition_a_match=float(partition_a_match[mask].sum()),
            partition_a_total=float(partition_a_total[mask].sum()),
            partition_b_match=float(partition_b_match[mask].sum()),
            partition_b_total=float(partition_b_total[mask].sum()),
            prefix=prefix,
            partition_a_name=partition_a_name,
            partition_b_name=partition_b_name,
        )
    return metrics_by_source


def group_validation_samples_by_data_source(
    data_sources: list[str] | np.ndarray,
    sample_uids: list[str] | np.ndarray,
    infos_dict: dict[str, list[Any]],
    *,
    source_to_group: Callable[[str], Optional[str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[Any]]]:
    grouped_sources: list[str] = []
    grouped_uids: list[Any] = []
    grouped_infos_dict = {key: [] for key in infos_dict}

    data_sources = np.asarray(data_sources, dtype=object)
    sample_uids = np.asarray(sample_uids, dtype=object)

    for sample_idx, data_source in enumerate(data_sources):
        source_name = str(data_source)
        group_name = source_to_group(source_name)
        if group_name is None:
            continue
        grouped_sources.append(group_name)
        grouped_uids.append(f"{source_name}::{sample_uids[sample_idx]}")
        for key, vals in infos_dict.items():
            grouped_infos_dict[key].append(vals[sample_idx])

    return np.asarray(grouped_sources, dtype=object), np.asarray(grouped_uids, dtype=object), grouped_infos_dict


def collect_validation_metrics_by_data_source_group(
    data_sources: list[str] | np.ndarray,
    sample_uids: list[str] | np.ndarray,
    infos_dict: dict[str, list[Any]],
    *,
    source_to_group: Callable[[str], Optional[str]],
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    grouped_sources, grouped_uids, grouped_infos_dict = group_validation_samples_by_data_source(
        data_sources,
        sample_uids,
        infos_dict,
        source_to_group=source_to_group,
    )
    if len(grouped_sources) == 0:
        return {}
    return process_validation_metrics(grouped_sources.tolist(), grouped_uids.tolist(), grouped_infos_dict, seed=seed)


def average_validation_metrics_by_data_source_group(
    metrics_by_source: dict[str, dict[str, dict[str, float]]],
    *,
    source_to_group: Callable[[str], Optional[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, var2metric2val in metrics_by_source.items():
        group_name = source_to_group(str(data_source))
        if group_name is None:
            continue
        for var_name, metric2val in var2metric2val.items():
            for metric_name, metric_val in metric2val.items():
                grouped_metrics[group_name][var_name][metric_name].append(metric_val)

    averaged_metrics = defaultdict(lambda: defaultdict(dict))
    for group_name, var2metric2vals in grouped_metrics.items():
        for var_name, metric2vals in var2metric2vals.items():
            for metric_name, vals in metric2vals.items():
                averaged_metrics[group_name][var_name][metric_name] = float(np.mean(vals))
    return averaged_metrics


def average_flat_metrics_by_data_source_group(
    metrics_by_source: dict[str, dict[str, float]],
    *,
    source_to_group: Callable[[str], Optional[str]],
) -> dict[str, dict[str, float]]:
    grouped_metrics = defaultdict(lambda: defaultdict(list))
    for data_source, metric2val in metrics_by_source.items():
        group_name = source_to_group(str(data_source))
        if group_name is None:
            continue
        for metric_name, metric_val in metric2val.items():
            grouped_metrics[group_name][metric_name].append(metric_val)

    averaged_metrics = defaultdict(dict)
    for group_name, metric2vals in grouped_metrics.items():
        for metric_name, vals in metric2vals.items():
            averaged_metrics[group_name][metric_name] = float(np.mean(vals))
    return averaged_metrics


def map_prmbench_source_to_group(source: str) -> Optional[str]:
    if not source.startswith(f"{PRMBENCH_DATA_SOURCE_PREFIX}/"):
        return None
    classification = source[len(PRMBENCH_DATA_SOURCE_PREFIX) + 1 :]
    if classification in {"correct", "overall"} or "/" in classification:
        return None
    for group_name, classifications in PRMBENCH_CLASSIFICATION_GROUPS.items():
        if classification in classifications:
            return f"{PRMBENCH_DATA_SOURCE_PREFIX}/{group_name}"
    raise ValueError(f"Unknown PRMBench classification: {classification}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else -1.0


def is_agent_process_bench_source(source: str) -> bool:
    return source == AGENT_PROCESS_BENCH_DATA_SOURCE_PREFIX or source.startswith(
        f"{AGENT_PROCESS_BENCH_DATA_SOURCE_PREFIX}/"
    )


def collect_agent_process_bench_metrics_by_data_source(
    data_sources: list[str] | np.ndarray,
    infos_dict: dict[str, list[Any]],
    manifest_by_source: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    required = (
        "agent_step_match",
        "agent_step_total",
        "agent_first_neg1_match",
        "agent_first_neg1_total",
    )
    if any(key not in infos_dict for key in required):
        return {}

    data_sources = np.asarray(data_sources, dtype=object)
    step_match = np.asarray(infos_dict["agent_step_match"], dtype=np.float64)
    step_total = np.asarray(infos_dict["agent_step_total"], dtype=np.float64)
    first_neg1_match = np.asarray(infos_dict["agent_first_neg1_match"], dtype=np.float64)
    first_neg1_total = np.asarray(infos_dict["agent_first_neg1_total"], dtype=np.float64)

    metrics_by_source: dict[str, dict[str, float]] = {}
    for source in np.unique(data_sources):
        source = str(source)
        if not is_agent_process_bench_source(source):
            continue
        mask = data_sources == source
        source_step_match = float(step_match[mask].sum())
        source_step_total = float(step_total[mask].sum())
        source_first_neg1_match = float(first_neg1_match[mask].sum())
        source_first_neg1_total = float(first_neg1_total[mask].sum())
        metrics_by_source[source] = {
            "StepAcc_kept": _safe_ratio(source_step_match, source_step_total),
            "FirstErrAcc_kept": _safe_ratio(source_first_neg1_match, source_first_neg1_total),
        }
        manifest_entry = (manifest_by_source or {}).get(source)
        if manifest_entry is not None:
            metrics_by_source[source].update(
                {
                    "StepAcc_all": _safe_ratio(source_step_match, float(manifest_entry["step_total"])),
                    "FirstErrAcc_all": _safe_ratio(
                        source_first_neg1_match,
                        float(manifest_entry["sample_total"]),
                    ),
                }
            )

    return metrics_by_source


def summarize_prmbench_metrics(
    *,
    tp: float,
    fp: float,
    tn: float,
    fn: float,
    correct_step_match: float,
    correct_step_total: float,
    wrong_step_match: float,
    wrong_step_total: float,
    total_step_match: float,
    total_step_total: float,
    first_error_match: float,
    first_error_total: float,
    classification: str,
) -> dict[str, float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = -1.0 if precision < 0 or recall < 0 or precision + recall == 0 else 2 * precision * recall / (precision + recall)

    negative_precision = _safe_ratio(tn, tn + fn)
    negative_recall = _safe_ratio(tn, tn + fp)
    negative_f1 = (
        -1.0
        if negative_precision < 0 or negative_recall < 0 or negative_precision + negative_recall == 0
        else 2 * negative_precision * negative_recall / (negative_precision + negative_recall)
    )

    prm_score = f1 if classification == "multi_solutions" else 0.5 * (f1 + negative_f1)
    if prm_score < 0:
        prm_score = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "negative_f1": float(negative_f1),
        "prm_score": float(prm_score),
        "correct_step_acc": _safe_ratio(correct_step_match, correct_step_total),
        "wrong_step_acc": _safe_ratio(wrong_step_match, wrong_step_total),
        "total_step_acc": _safe_ratio(total_step_match, total_step_total),
        "first_error_acc": _safe_ratio(first_error_match, first_error_total),
    }


def collect_prmbench_metrics_by_data_source(
    data_sources: list[str] | np.ndarray,
    infos_dict: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    required = (
        "prmbench_tp",
        "prmbench_fp",
        "prmbench_tn",
        "prmbench_fn",
        "prmbench_correct_step_match",
        "prmbench_correct_step_total",
        "prmbench_wrong_step_match",
        "prmbench_wrong_step_total",
        "prmbench_total_step_match",
        "prmbench_total_step_total",
        "prmbench_first_error_match",
        "prmbench_first_error_total",
    )
    if any(key not in infos_dict for key in required):
        return {}

    data_sources = np.asarray(data_sources, dtype=object)
    metrics_by_source: dict[str, dict[str, float]] = {}

    for source in np.unique(data_sources):
        source = str(source)
        if not source.startswith(f"{PRMBENCH_DATA_SOURCE_PREFIX}/"):
            continue
        mask = data_sources == source
        classification = source.split("/")[-1]
        metrics_by_source[source] = summarize_prmbench_metrics(
            tp=float(np.asarray(infos_dict["prmbench_tp"], dtype=np.float64)[mask].sum()),
            fp=float(np.asarray(infos_dict["prmbench_fp"], dtype=np.float64)[mask].sum()),
            tn=float(np.asarray(infos_dict["prmbench_tn"], dtype=np.float64)[mask].sum()),
            fn=float(np.asarray(infos_dict["prmbench_fn"], dtype=np.float64)[mask].sum()),
            correct_step_match=float(np.asarray(infos_dict["prmbench_correct_step_match"], dtype=np.float64)[mask].sum()),
            correct_step_total=float(np.asarray(infos_dict["prmbench_correct_step_total"], dtype=np.float64)[mask].sum()),
            wrong_step_match=float(np.asarray(infos_dict["prmbench_wrong_step_match"], dtype=np.float64)[mask].sum()),
            wrong_step_total=float(np.asarray(infos_dict["prmbench_wrong_step_total"], dtype=np.float64)[mask].sum()),
            total_step_match=float(np.asarray(infos_dict["prmbench_total_step_match"], dtype=np.float64)[mask].sum()),
            total_step_total=float(np.asarray(infos_dict["prmbench_total_step_total"], dtype=np.float64)[mask].sum()),
            first_error_match=float(np.asarray(infos_dict["prmbench_first_error_match"], dtype=np.float64)[mask].sum()),
            first_error_total=float(np.asarray(infos_dict["prmbench_first_error_total"], dtype=np.float64)[mask].sum()),
            classification=classification,
        )

    return metrics_by_source


def collect_prmbench_similarity_by_data_source(
    data_sources: list[str] | np.ndarray,
    infos_dict: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    required = ("prmbench_pair_id", "prmbench_model_response_acc")
    if any(key not in infos_dict for key in required):
        return {}

    data_sources = np.asarray(data_sources, dtype=object)
    pair_ids = np.asarray(infos_dict["prmbench_pair_id"], dtype=object)
    model_response_acc = np.asarray(infos_dict["prmbench_model_response_acc"], dtype=np.float64)

    correct_response_acc_by_pair_id: dict[str, list[float]] = defaultdict(list)
    for idx, source in enumerate(data_sources):
        if str(source) == PRMBENCH_CORRECT_SOURCE and model_response_acc[idx] != -1:
            correct_response_acc_by_pair_id[str(pair_ids[idx])].append(float(model_response_acc[idx]))

    averaged_correct_response_acc = {
        pair_id: float(np.mean(accs)) for pair_id, accs in correct_response_acc_by_pair_id.items()
    }

    similarity_values_by_source: dict[str, list[float]] = defaultdict(list)
    for idx, source in enumerate(data_sources):
        source = str(source)
        if not source.startswith(f"{PRMBENCH_DATA_SOURCE_PREFIX}/") or source == PRMBENCH_CORRECT_SOURCE:
            continue
        pair_id = str(pair_ids[idx])
        if pair_id in averaged_correct_response_acc and model_response_acc[idx] != -1:
            similarity_values_by_source[source].append(
                abs(float(model_response_acc[idx]) - averaged_correct_response_acc[pair_id])
            )

    metrics_by_source = {
        source: {"similarity": float(np.mean(vals))} for source, vals in similarity_values_by_source.items()
    }
    overall_similarity_values = [val for vals in similarity_values_by_source.values() for val in vals]
    if overall_similarity_values:
        metrics_by_source[f"{PRMBENCH_DATA_SOURCE_PREFIX}/overall"] = {
            "similarity": float(np.mean(overall_similarity_values))
        }
    return metrics_by_source


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()
    # standard deviation of sequence scores (non-aborted only)
    # use population std (unbiased=False) for stability on small batches
    score_std = torch.std(non_aborted_sequence_score, unbiased=False).detach().item()

    # Group-wise std by uid (within-query variance)
    # Compute per-uid std over non-aborted samples, then aggregate mean/min/max
    group_std_mean = group_std_min = group_std_max = None
    # Prefer uid; fall back to request_id (from rollout engines) if uid missing
    id_key = "uid" if "uid" in batch.non_tensor_batch else ("request_id" if "request_id" in batch.non_tensor_batch else None)
    logger.warning('batch.non_tensor_batch:', batch.non_tensor_batch)

    if id_key is not None:
        try:
            uids = batch.non_tensor_batch[id_key]
            if hasattr(uids, "tolist"):
                uids = uids.tolist()
            # gather per-uid scores on non-aborted samples
            non_aborted_scores_np = non_aborted_sequence_score.detach().cpu().numpy()
            non_aborted_uids = [uid for uid, keep in zip(uids, non_aborted_mask.tolist(), strict=True) if keep]

            buckets: dict[str, list[float]] = defaultdict(list)
            for uid, sc in zip(non_aborted_uids, non_aborted_scores_np, strict=True):
                buckets[str(uid)].append(float(sc))

            per_group_stds = []
            debug_group_sizes: Optional[dict[str, int]] = None
            aborted_per_uid: Optional[dict[str, int]] = None
            should_log_group_sizes = os.getenv("VERL_DEBUG_GROUP_SIZES", "0") in {"1", "true", "True"}
            if should_log_group_sizes or logger.isEnabledFor(logging.DEBUG):
                debug_group_sizes = {uid: len(vals) for uid, vals in buckets.items()}
                total_per_uid = Counter(str(uid) for uid in uids)
                aborted_per_uid = {
                    uid: total_per_uid.get(uid, 0) - debug_group_sizes.get(uid, 0)
                    for uid in total_per_uid.keys()
                }

            for vals in buckets.values():
                if len(vals) <= 1:
                    per_group_stds.append(0.0)
                else:
                    per_group_stds.append(float(np.std(vals, ddof=0)))

            if debug_group_sizes is not None:
                logger.debug(
                    "group_std debug: non_aborted=%s aborted=%s",  # noqa: G004
                    debug_group_sizes,
                    aborted_per_uid,
                )

            if len(per_group_stds) > 0:
                arr = np.asarray(per_group_stds, dtype=float)
                group_std_mean = float(arr.mean())
                group_std_min = float(arr.min())
                group_std_max = float(arr.max())
        except Exception as e:
            logger.warning('failed to compute bucket statistics', e)
            # swallow grouping errors silently to avoid breaking training
            pass
    else:
        logger.warning('No id_key')

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        "critic/score/std": score_std,
        **(
            {
                "critic/score/std_group/mean": group_std_mean,
                "critic/score/std_group/min": group_std_min,
                "critic/score/std_group/max": group_std_max,
            }
            if group_std_mean is not None
            else {}
        ),
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_rollout_metrics(batch: DataProto, aggregate_only: bool | None = None) -> dict:
    """Aggregate rollout-side metrics for logging.

    Expects `batch.non_tensor_batch["rollout_metrics"]` to be a list-like of per-request
    metrics dicts collected during rollout. Each dict maps tool_name -> list[metrics_dict]
    and may include a special key "<rollout>" -> list[{"multi_turn_rounds": int}].

    Returns a flat dict suitable for Tracking.log.
    """
    metric_dict: dict[str, float] = {}
    # Resolve aggregation control: prefer explicit arg; fallback to env; default False
    if aggregate_only is None:
        aggregate_only = os.environ.get("VERL_ROLLOUT_METRICS_AGGREGATE_ONLY", "") == "1"

    if "rollout_metrics" not in batch.non_tensor_batch:
        logger.warning("rollout_metrics not present in batch")
        return metric_dict

    metrics_list = batch.non_tensor_batch["rollout_metrics"]
    # Optional structural assertions to debug rollout metrics (enable by setting VERL_ASSERT_ROLLOUT_METRICS=1)
    if os.environ.get("VERL_ASSERT_ROLLOUT_METRICS", "") == "1":
        assert hasattr(metrics_list, "tolist") or isinstance(metrics_list, (list, tuple)), (
            f"rollout_metrics must be a numpy array or list, got {type(metrics_list)}"
        )
    # logger.warning('metrics_list:', metrics_list)
    # Normalize to python list
    if hasattr(metrics_list, "tolist"):
        try:
            metrics_list = metrics_list.tolist()
        except Exception:
            metrics_list = list(metrics_list)

    if os.environ.get("VERL_ASSERT_ROLLOUT_METRICS", "") == "1":
        assert isinstance(metrics_list, list), f"rollout_metrics after tolist() must be list, got {type(metrics_list)}"
        for i, req_metrics in enumerate(metrics_list):
            assert isinstance(req_metrics, dict), f"rollout_metrics[{i}] must be dict, got {type(req_metrics)}"
            for k, v in req_metrics.items():
                assert isinstance(k, str), f"rollout_metrics[{i}] key must be str, got {type(k)}"
                assert isinstance(v, list), f"rollout_metrics[{i}]['{k}'] must be list, got {type(v)}"
                for j, entry in enumerate(v):
                    assert isinstance(entry, (dict, str)), (
                        f"rollout_metrics[{i}]['{k}'][{j}] must be dict or str, got {type(entry)}"
                    )
                    if isinstance(entry, dict):
                        # Optional field type checks
                        if "tool_tokens" in entry:
                            assert isinstance(entry["tool_tokens"], (int, np.integer)), (
                                f"tool_tokens must be int, got {type(entry['tool_tokens'])}"
                            )
                        if "truncated" in entry:
                            assert isinstance(entry["truncated"], (bool, np.bool_)), (
                                f"truncated must be bool, got {type(entry['truncated'])}"
                            )
                        if "truncation_ratio" in entry:
                            assert isinstance(entry["truncation_ratio"], (float, int, np.floating, np.integer)), (
                                f"truncation_ratio must be number, got {type(entry['truncation_ratio'])}"
                            )

    total_tool_tokens = 0
    total_tool_calls = 0
    trunc_true = 0
    trunc_ratio_sum = 0.0
    trunc_ratio_cnt = 0
    mt_rounds = []

    # Per-tool success/error aggregation (keyed by tool name)
    per_tool_stats: dict[str, dict[str, Any]] = {}
    # Global aggregation across all tools (fewer metrics)
    overall_tool_stats: dict[str, Any] = {
        "success": 0,
        "error": 0,
        "total_eval": 0,
        "error_types": defaultdict(int),
    }

    def _sanitize_key(s: str) -> str:
        # keep alnum, dot; replace others with underscore; limit length
        return "".join(ch if (ch.isalnum() or ch == ".") else "_" for ch in str(s))[:80]

    saw_any_tool_entry = False
    for req_metrics in metrics_list:
        if not isinstance(req_metrics, dict):
            continue
        for name, vals in req_metrics.items():
            if name == "<rollout>":
                # list of dicts with multi_turn_rounds
                for v in vals or []:
                    if isinstance(v, dict) and "multi_turn_rounds" in v:
                        try:
                            mt_rounds.append(int(v["multi_turn_rounds"]))
                        except Exception:
                            pass
                continue

            # tool name -> list[metrics]
            if not isinstance(vals, list):
                continue
            for m in vals:
                # Count every entry as a tool call, even if no metrics dict was provided
                total_tool_calls += 1
                saw_any_tool_entry = True

                # Per-tool bucket
                tool_key = _sanitize_key(name)
                stats = per_tool_stats.setdefault(tool_key, {
                    "success": 0,
                    "error": 0,
                    "total_eval": 0,
                    "error_types": defaultdict(int),
                })

                # Token/truncation counters if dict
                if isinstance(m, dict):
                    total_tool_tokens += int(m.get("tool_tokens", 0) or 0)
                    if m.get("truncated", False):
                        trunc_true += 1
                    if "truncation_ratio" in m:
                        try:
                            trunc_ratio_sum += float(m["truncation_ratio"])
                            trunc_ratio_cnt += 1
                        except Exception:
                            pass

                # Classify success/failure
                # - For REST tools (m.tool_kind == 'rest_tool'): default-fail and always count in denominator
                # - For other tools: previous behavior (only count when status or error present)
                is_rest = isinstance(m, dict) and m.get("tool_kind") == "rest_tool"

                if is_rest:
                    # REST: always include in denominator
                    stats["total_eval"] += 1
                    overall_tool_stats["total_eval"] += 1

                    success = False
                    status_raw = m.get("status", "")
                    status_code = m.get("status_code", None)
                    try:
                        code = int(status_code)
                    except Exception:
                        code = None
                    status_ok = (code is not None) and (200 <= code < 300)
                    status_is_success = str(status_raw).lower() == "success"
                    if status_is_success and status_ok:
                        success = True

                    if success:
                        stats["success"] += 1
                        overall_tool_stats["success"] += 1
                    else:
                        stats["error"] += 1
                        overall_tool_stats["error"] += 1
                        # determine error type label
                        err_label = "unknown"
                        if m.get("error"):
                            err_label = _sanitize_key(str(m["error"]).split("\n", 1)[0])
                        else:
                            if m.get("status_code") is not None:
                                try:
                                    err_label = f"http:{int(m['status_code'])}"
                                except Exception:
                                    err_label = f"status:{_sanitize_key(str(m.get('status', '')))}"
                            elif m.get("detail"):
                                err_label = _sanitize_key(f"detail:{str(m['detail'])[:120]}")
                            elif m.get("status"):
                                err_label = _sanitize_key(f"status:{str(m['status'])}")
                        stats["error_types"][err_label] += 1
                        overall_tool_stats["error_types"][err_label] += 1
                else:
                    # Non-REST: only count when status or error present
                    if isinstance(m, dict):
                        err = m.get("error", None)
                        if err is not None and str(err) != "":
                            stats["error"] += 1
                            stats["total_eval"] += 1
                            overall_tool_stats["error"] += 1
                            overall_tool_stats["total_eval"] += 1
                            err_label = _sanitize_key(str(err).split("\n", 1)[0])
                            stats["error_types"][err_label] += 1
                            overall_tool_stats["error_types"][err_label] += 1
                        else:
                            status_raw = m.get("status", "") if "status" in m else ""
                            status = str(status_raw).lower()
                            if status != "":
                                # Prefer HTTP-like numeric status when available
                                ok = False
                                try:
                                    code = int(str(status_raw))
                                    ok = 200 <= code < 400
                                except Exception:
                                    ok_set = {"success", "ok", "succeeded", "200", "true"}
                                    ok = status in ok_set
                                if ok:
                                    stats["success"] += 1
                                    overall_tool_stats["success"] += 1
                                else:
                                    stats["error"] += 1
                                    err_label = _sanitize_key(f"status:{status}")
                                    stats["error_types"][err_label] += 1
                                    overall_tool_stats["error"] += 1
                                    overall_tool_stats["error_types"][err_label] += 1
                                stats["total_eval"] += 1
                                overall_tool_stats["total_eval"] += 1

    num_reqs = max(1, len(metrics_list))
    # Averages instead of totals
    if total_tool_calls > 0:
        metric_dict["rollout/avg_tool_tokens_per_call"] = float(total_tool_tokens) / float(total_tool_calls)
        metric_dict["rollout/avg_tool_calls_per_request"] = float(total_tool_calls) / float(num_reqs)
        metric_dict["rollout/truncation_rate"] = float(trunc_true) / float(total_tool_calls)
        if trunc_ratio_cnt > 0:
            metric_dict["rollout/truncation_ratio_avg"] = trunc_ratio_sum / float(trunc_ratio_cnt)

    if len(mt_rounds) > 0:
        mt_arr = np.array(mt_rounds)
        metric_dict["rollout/multi_turn_rounds/mean"] = float(mt_arr.mean())
        metric_dict["rollout/multi_turn_rounds/min"] = float(mt_arr.min())
        metric_dict["rollout/multi_turn_rounds/max"] = float(mt_arr.max())

    # Emit per-tool success/error ratios and error-type breakdown (can be disabled via env)
    if not aggregate_only:
        for tool_name, stats in per_tool_stats.items():
            tot = int(stats.get("total_eval", 0))
            if tot <= 0:
                continue
            succ = int(stats.get("success", 0))
            errc = int(stats.get("error", 0))
            prefix = f"rollout/tools/{tool_name}"
            metric_dict[f"{prefix}/success_ratio"] = succ / float(tot)
            metric_dict[f"{prefix}/error_ratio"] = errc / float(tot)
            metric_dict[f"{prefix}/success_count"] = succ
            metric_dict[f"{prefix}/error_count"] = errc

            # top-5 error types by count
            error_types: dict[str, int] = stats.get("error_types", {})
            if isinstance(error_types, defaultdict):
                error_types = dict(error_types)
            if error_types:
                top = sorted(error_types.items(), key=lambda kv: kv[1], reverse=True)[:5]
                for etype, cnt in top:
                    etype_pref = f"{prefix}/error_type/{etype}"
                    metric_dict[f"{etype_pref}/count"] = int(cnt)
                    metric_dict[f"{etype_pref}/ratio"] = int(cnt) / float(tot)

    # Emit overall aggregated tool success/error and error types (always on)
    tot_overall = int(overall_tool_stats.get("total_eval", 0))
    if tot_overall > 0:
        succ_overall = int(overall_tool_stats.get("success", 0))
        err_overall = int(overall_tool_stats.get("error", 0))
        prefix = "rollout/tools_overall"
        metric_dict[f"{prefix}/success_ratio"] = succ_overall / float(tot_overall)
        metric_dict[f"{prefix}/error_ratio"] = err_overall / float(tot_overall)
        metric_dict[f"{prefix}/success_count"] = succ_overall
        metric_dict[f"{prefix}/error_count"] = err_overall

        error_types_overall: dict[str, int] = overall_tool_stats.get("error_types", {})
        if isinstance(error_types_overall, defaultdict):
            error_types_overall = dict(error_types_overall)
        if error_types_overall:
            top = sorted(error_types_overall.items(), key=lambda kv: kv[1], reverse=True)[:10]
            for etype, cnt in top:
                etype_pref = f"{prefix}/error_type/{etype}"
                metric_dict[f"{etype_pref}/count"] = int(cnt)
                metric_dict[f"{etype_pref}/ratio"] = int(cnt) / float(tot_overall)


    # Final assertion to catch silent drops: if there were tool entries, expect some rollout/* outputs
    if os.environ.get("VERL_ASSERT_ROLLOUT_METRICS", "") == "1" and len(metrics_list) > 0 and saw_any_tool_entry:
        assert any(k.startswith("rollout/") for k in metric_dict.keys()), (
            "compute_rollout_metrics produced no rollout/* keys despite tool entries; check rollout_metrics wiring"
        )

    return metric_dict


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
