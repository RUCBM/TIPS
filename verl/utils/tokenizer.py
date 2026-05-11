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
"""Utils for tokenization."""

import warnings
from typing import Any

__all__ = ["hf_tokenizer", "hf_processor", "render_chat_prompt"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def _stringify_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list | tuple):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image":
                    parts.append("<image>")
                elif item_type == "video":
                    parts.append("<video>")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def _render_plain_prompt(messages: list[dict[str, Any]], add_generation_prompt: bool = True) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _stringify_chat_content(message.get("content", "")).strip()

        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"{role.title()}: {content}")

    if add_generation_prompt:
        lines.append("Assistant:")

    return "\n".join(lines)


def _validate_plain_prompt_fallback(messages: list[dict[str, Any]], apply_kwargs: dict[str, Any]) -> None:
    if apply_kwargs.get("tools"):
        raise ValueError(
            "Plain-prompt fallback does not support tool schemas. "
            "Please provide a tokenizer chat_template or pass apply_chat_template_kwargs.chat_template."
        )

    if apply_kwargs.get("documents"):
        raise ValueError(
            "Plain-prompt fallback does not support chat-template documents. "
            "Please provide a tokenizer chat_template or pass apply_chat_template_kwargs.chat_template."
        )

    for idx, message in enumerate(messages):
        role = str(message.get("role", "user")).strip().lower()
        if message.get("tool_calls"):
            raise ValueError(
                f"Plain-prompt fallback does not support tool_calls in message {idx}. "
                "Please provide a tokenizer chat_template or pass apply_chat_template_kwargs.chat_template."
            )
        if role == "tool":
            raise ValueError(
                f"Plain-prompt fallback does not support tool-role messages (message {idx}). "
                "Please provide a tokenizer chat_template or pass apply_chat_template_kwargs.chat_template."
            )


def render_chat_prompt(tokenizer, messages, add_generation_prompt: bool = True, apply_chat_template_kwargs=None) -> str:
    """Render chat messages to a string prompt.

    If the tokenizer exposes a chat template, use it. Otherwise, fall back to
    a plain role-tagged prompt of the form ``User: ...`` / ``Assistant: ...``.
    """

    apply_kwargs = dict(apply_chat_template_kwargs or {})
    apply_kwargs.pop("tokenize", None)

    if apply_kwargs.get("chat_template") is not None or getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **apply_kwargs,
        )

    _validate_plain_prompt_fallback(messages, apply_kwargs)
    return _render_plain_prompt(messages, add_generation_prompt=add_generation_prompt)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
