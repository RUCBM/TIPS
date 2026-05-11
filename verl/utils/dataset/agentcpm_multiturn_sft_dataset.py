"""
Multi-turn SFT dataset that supports training on `json` format conversation data with multiple turns
"""


import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset, convert_nested_value_to_list_recursive
from tqdm import tqdm


def convert_sharegpt_to_conversation(sharegpt_sample):
    conversations = [
        {'role': 'system', 'content': sharegpt_sample['system']}
    ]
    for index, conversation in enumerate(sharegpt_sample["conversations"]):
        if conversation['value'].strip() == '':
            continue
        if conversation['from'] == 'human':
            conversations.append( {'role': 'user', 'content': conversation['value']})
        elif conversation['from'] == 'gpt':
            conversations.append({'role': 'assistant', 'content': conversation['value']})

    return conversations

class AgentCPMMultiTurnSFTDataset(MultiTurnSFTDataset):
    def __init__(
        self,
        json_files: str | list[str] | None = None,
        *,
        tokenizer,
        config=None,
        parquet_files: str | list[str] | None = None,
        **unused_kwargs,
    ):
        """Create dataset from ShareGPT-style JSON conversations.

        The regular trainer factory instantiates datasets with a ``parquet_files``
        argument. Accept it here as an alias so the dataset can be plugged into
        ``create_sft_dataset`` without additional glue code.
        """

        if json_files is not None and parquet_files is not None:
            raise ValueError("Pass either json_files or parquet_files, not both.")

        if parquet_files is not None:
            json_files = parquet_files

        if json_files is None:
            raise ValueError("AgentCPMMultiTurnSFTDataset expects json_files input.")

        if unused_kwargs:
            logging.warning("Unused keyword arguments received: %s", sorted(unused_kwargs))


        # 缓冲length，避免因为多轮构造导致爆长度
        self.buffer_len = 128
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.filter_empty = multiturn_config.get("filter_empty", False)

        self.enable_thinking = config.get("enable_thinking", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.max_prompt_length = config.get("max_prompt_length", self.max_length)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(json_files, list | ListConfig):
            json_files = [json_files]

        self.json_files = json_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._read_files_and_process()

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        self.data = []
        for json_file in self.json_files:
            sharegpt_data = json.load(open(json_file, 'r'))

            self.data += sharegpt_data

        self._maybe_filter_empty_samples()

        # Extract ShareGPT messages list from dataframe
        self.messages = [convert_sharegpt_to_conversation(sharegpt_sample) for sharegpt_sample in self.data]

        self.tools = [sharegpt_sample['tools'] for sharegpt_sample in self.data]

        self._maybe_filter_overlong_prompts()





    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking

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
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
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
        }

    def _maybe_filter_empty_samples(self):
        if not self.filter_empty:
            return

        def empty_sample(sample):
            for conv in sample['conversations']:
                if conv['from'] == 'gpt':
                    if '<tool_call>' in conv['value']:
                        continue
                    if '<answer>' in conv['value']:
                        continue
                    return True
            return False

        self.data = [sample for sample in self.data if not empty_sample(sample)]

    def _maybe_filter_overlong_prompts(self) -> None:
        if not self.filter_overlong_prompts or self.max_prompt_length is None:
            if self.filter_overlong_prompts:
                logging.warning("filter_overlong_prompts=True but max_prompt_length is None; skipping filtering")
            return

        enable_thinking = self.enable_thinking if isinstance(self.enable_thinking, bool) else None
        tools_iter = self.tools if self.tools is not None else [None] * len(self.messages)
        kept_indices: list[int] = []
        removed = 0
        batch_size = 512

        for start in tqdm(range(0, len(self.messages), batch_size), desc="filtering overlong prompts"):
            msg_batch = self.messages[start:start + batch_size]
            tool_batch = tools_iter[start:start + batch_size]

            templated = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                    **self.apply_chat_template_kwargs,
                )
                for messages, tools in zip(msg_batch, tool_batch)
            ]

            encoded = self.tokenizer(
                templated,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_length=True,
            )
            lengths = encoded["length"]

            for offset, length in enumerate(lengths):
                if length <= self.max_prompt_length - self.buffer_len:
                    kept_indices.append(start + offset)
                else:
                    removed += 1

        if removed:
            logging.warning(
                "Filtered %s overlong prompts (threshold=%s tokens). Remaining samples: %s",
                removed,
                self.max_prompt_length,
                len(kept_indices),
            )

        self.messages = [self.messages[i] for i in kept_indices]
        self.data = [self.data[i] for i in kept_indices]
        if self.tools is not None:
            self.tools = [self.tools[i] for i in kept_indices]


if __name__ == "__main__":
    import os
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    
    json_files = ["/path/to/agentcpm/sft_data.json"]


    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen/Qwen3-4B-Thinking-2507-keep-empty-think")
    config = {"max_length": 64000, "truncation": "error", "multiturn": {"messages_key": "conversations", 'filter_empty': True}}

    dataset = AgentCPMMultiTurnSFTDataset(json_files=json_files, tokenizer=tokenizer, config=config)


    print(dataset[30])
    print(dataset[2])
    print(dataset[1])
