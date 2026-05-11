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
"""Multi-turn dataset variant that inserts step markers after each assistant turn."""

from __future__ import annotations

from typing import Any, Optional

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

class MarkerAnchoredMultiTurnSFTDataset(MultiTurnSFTDataset):
    """Dataset that appends a marker token (e.g. ``<extra_0>``) after assistant messages.

    The extra marker allows downstream reward-model heads to trivially locate step
    boundaries without having to re-parse the ChatML template. Marker tokens are
    included in ``attention_mask`` so the base model produces hidden states at
    those positions, but their ``loss_mask`` entries default to zero which keeps
    them excluded from standard next-token losses.

    Expected config additions (all optional)::

        marker_anchor:
            enable: True
            token: "<extra_0>"
            loss_mask_value: 0
            roles: ["assistant"]

    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        config = config or {}
        marker_cfg = config.get("marker_anchor", {})
        self._marker_enabled: bool = marker_cfg.get("enable", True)
        self._marker_token: str = marker_cfg.get("token", "<extra_0>")
        self._marker_loss_mask_value: int = int(marker_cfg.get("loss_mask_value", 0))
        roles = marker_cfg.get("roles", ("tool",))
        self._marker_roles = set(roles)
        self._marker_token_ids: list[int] = []

        super().__init__(parquet_files=parquet_files, tokenizer=tokenizer, config=config, **kwargs)

        if not self._marker_enabled:
            return

        marker_token_ids = self.tokenizer.encode(self._marker_token, add_special_tokens=False)
        if not marker_token_ids:
            raise ValueError(
                "MarkerAnchoredMultiTurnSFTDataset: encoding marker token produced an empty id sequence. "
                "Ensure the tokenizer vocabulary contains the marker (e.g. '<extra_0>')."
            )
        
        
        
        self._trajectory_labels = [{'positive':1, 'negative':0}[sample['ground_truth']] for sample in self.dataframe['reward_model'].tolist()]
        print('len(self._trajectory_labels):', self._trajectory_labels)
        self._marker_token_ids = marker_token_ids




    def __getitem__(self, item):
        tokenizer = self.tokenizer

        messages = self.messages[item]
        # 1) 外层 np.ndarray -> list
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()

        normalized = []
        for m in messages:
            # 有些 parquet 读出来可能是 tuple / 其他类型，统一成 dict
            if not isinstance(m, dict):
                m = dict(m)

            tc = m.get("tool_calls", None)
            # 2) 内层 tool_calls 的 np.ndarray -> list
            if isinstance(tc, np.ndarray):
                m["tool_calls"] = tc.tolist()

            normalized.append(m)

        messages = normalized


        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, is_tool=True, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            'trajectory_labels': self._trajectory_labels[item]
        }


    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        *,
        is_tool:bool = False,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        tokens, loss_mask, attention_mask = super()._process_message_tokens(
            messages,
            start_idx,
            end_idx,
            is_assistant=is_assistant,
            enable_thinking=enable_thinking,
            tools=tools,
        )

        loss_mask = [0] * len(loss_mask)

        if (
            self._marker_enabled
            and is_tool
            and "tool" in self._marker_roles
            and self._marker_token_ids
        ) or (
            self._marker_enabled
            and start_idx == len(messages) - 1
            and self._marker_token_ids
        ):
        # if (
        #     self._marker_enabled
        #     and start_idx == len(messages) - 1
        #     and self._marker_token_ids
        # ):
            # Append marker tokens after the assistant turn so the model produces
            # hidden states at explicit step boundaries.
            tokens = tokens + self._marker_token_ids
            loss_mask = [0] * len(loss_mask) + [self._marker_loss_mask_value] * len(self._marker_token_ids)
            attention_mask = attention_mask + [1] * len(self._marker_token_ids)

        return tokens, loss_mask, attention_mask
