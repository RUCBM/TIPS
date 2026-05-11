# -*- coding: utf-8 -*-
# verl/extensions/credit/ce_gate_credit.py
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

try:  # PyTorch FSDP v1
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FSDP = None

try:  # PyTorch FSDP v2 (composable API)
    from torch.distributed.fsdp import FSDPModule  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    FSDPModule = None

_FSDP_WRAPPER_TYPES: Tuple[type, ...] = tuple(
    cls for cls in (FSDP, FSDPModule) if cls is not None
)

# ---------------- ChatML 解析工具 ----------------

def _find_all_subseq(hay: List[int], needle: List[int]) -> List[int]:
    H, N = len(hay), len(needle)
    if N == 0 or H < N:
        return []
    hits = []
    for i in range(H - N + 1):
        if hay[i : i + N] == needle:
            hits.append(i)
    return hits

def _next_occurrence(hay: List[int], needle: List[int], start: int) -> int:
    H, N = len(hay), len(needle)
    for i in range(start, H - N + 1):
        if hay[i : i + N] == needle:
            return i
    return -1

@torch.no_grad()
def get_turns(input_ids: Tensor, attention_mask: Tensor, tokenizer) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    解析 ChatML (<|im_start|>role ... <|im_end|>)，返回每个样本的 assistant/tool spans。
    返回:
      {
        "assistant_turns": List[List[(s,e)]],
        "tool_turns":      List[List[(s,e)]],
      }
    """
    assert input_ids.dim() == 2 and attention_mask.dim() == 2
    B, T = input_ids.shape

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    im_start = enc("<|im_start|>")
    im_end   = enc("<|im_end|>")
    tok_nl   = enc("\n")
    role_assistant = enc("assistant")
    role_user      = enc("user")
    tool_l = enc("<tool_response>")
    tool_r = enc("</tool_response>")

    all_assistant_turns: List[List[Tuple[int, int]]] = []
    all_tool_turns: List[List[Tuple[int, int]]] = []

    for b in range(B):
        Tb = int(attention_mask[b].sum().item())
        ids = input_ids[b, :Tb].tolist()

        assistant_turns: List[Tuple[int, int]] = []
        tool_turns: List[Tuple[int, int]] = []

        starts = _find_all_subseq(ids, im_start)
        i = 0

        def _match_role_at(pos: int, role_tok: List[int]) -> Tuple[bool, int]:
            rlen = len(role_tok)
            if ids[pos : pos + rlen] == role_tok:
                after = pos + rlen
                if ids[after : after + len(tok_nl)] == tok_nl:
                    after += len(tok_nl)
                return True, after
            return False, -1

        while i < len(starts):
            s0 = starts[i]
            role_pos = s0 + len(im_start)
            is_assistant, content_start = _match_role_at(role_pos, role_assistant)
            is_user, cs_user = (False, -1)
            if not is_assistant:
                is_user, cs_user = _match_role_at(role_pos, role_user)

            search_from = content_start if is_assistant else (cs_user if is_user else role_pos)
            e0 = _next_occurrence(ids, im_end, start=search_from)
            if e0 == -1:
                break

            if is_assistant:
                if content_start < e0:
                    assistant_turns.append((content_start, e0))
            elif is_user:
                seg = ids[cs_user:e0]
                tl_rel = _find_all_subseq(seg, tool_l)
                tr_rel = _find_all_subseq(seg, tool_r)
                tl_rel.sort()
                tr_rel.sort()
                si = sj = 0
                while si < len(tl_rel) and sj < len(tr_rel):
                    tl = tl_rel[si]
                    tr = tr_rel[sj]
                    if tr <= tl:
                        sj += 1
                        continue
                    abs_l = cs_user + tl + len(tool_l)
                    abs_r = cs_user + tr
                    if abs_l < abs_r:
                        tool_turns.append((abs_l, abs_r))
                    si += 1
                    sj += 1
            i += 1

        all_assistant_turns.append(assistant_turns)
        all_tool_turns.append(tool_turns)

    return {"assistant_turns": all_assistant_turns, "tool_turns": all_tool_turns}

# ---------------- CE-Gate Credit 主体 ----------------

def _unwrap(module: nn.Module) -> nn.Module:
    """Unwrap standard wrappers while keeping FSDP handles intact."""

    while True:
        if _FSDP_WRAPPER_TYPES and isinstance(module, _FSDP_WRAPPER_TYPES):
            return module
        if not hasattr(module, "module"):
            return module
        module = module.module  # handle DDP / Accelerate wrappers


def _resolve_input_embeddings(causal_lm: nn.Module) -> nn.Module:
    """Fetch the input embedding module, traversing wrappers if needed."""

    module: Optional[nn.Module] = causal_lm
    while module is not None:
        getter = getattr(module, "get_input_embeddings", None)
        if callable(getter):
            embed = getter()
            if embed is not None:
                return embed
        module = getattr(module, "module", None)
    raise AttributeError("Model does not expose get_input_embeddings().")


def _resolve_embedding_hook_target(causal_lm: nn.Module) -> nn.Module:
    """Return the module to attach forward/backward hooks for embedding outputs.

    Prefer the FSDP wrapper around the embedding if present, so that hooks fire
    even when the embedding module is fully-sharded. Fallback to the bare
    embedding module when no wrapper exists.
    """
    # First, get the bare embedding for fallback check
    try:
        bare = _resolve_input_embeddings(causal_lm)
    except Exception:
        bare = None

    # Scan modules to find a wrapper that wraps an nn.Embedding
    for mod in causal_lm.modules():
        if _FSDP_WRAPPER_TYPES and isinstance(mod, _FSDP_WRAPPER_TYPES):
            inner = (
                getattr(mod, "_fsdp_wrapped_module", None)
                or getattr(mod, "module", None)
                or getattr(mod, "_module", None)
            )
            if isinstance(inner, nn.Embedding):
                return mod
    # No wrapper found; fallback to bare embedding
    if bare is not None:
        return bare
    # As a last resort, return the top-level module to avoid None
    return _unwrap(causal_lm)


def _sum_span(vec: Tensor, span: Optional[Tuple[int, int]]) -> float:
    if span is None:
        return 0.0
    s, e = span
    if e <= s:
        return 0.0
    s = max(0, min(int(s), vec.numel()))
    e = max(0, min(int(e), vec.numel()))
    return float(vec[s:e].sum().item())


def _last_answer_window(
    input_ids: Tensor,
    attention_mask: Tensor,
    turns: Dict[str, List[List[Tuple[int, int]]]],
    max_ctx: Optional[int],
    device: torch.device,
):
    if max_ctx is None:
        T = input_ids.size(1)
        pos = torch.arange(T, device=device).unsqueeze(0).expand_as(input_ids)
        offsets = [0 for _ in range(input_ids.size(0))]
        return input_ids, attention_mask, pos, offsets

    ids_w, msk_w, pos_w, offsets = [], [], [], []
    for b in range(input_ids.size(0)):
        s_last, e_last = turns["assistant_turns"][b][-1]
        left = max(0, s_last - max_ctx)
        ids_b = input_ids[b, left:]
        msk_b = attention_mask[b, left:]
        pos_b = torch.arange(left, left + ids_b.size(0), device=device, dtype=torch.long)
        ids_w.append(ids_b)
        msk_w.append(msk_b)
        pos_w.append(pos_b)
        offsets.append(left)

    ids_w = pad_sequence(ids_w, batch_first=True, padding_value=0)
    msk_w = pad_sequence(msk_w, batch_first=True, padding_value=0)
    pos_w = pad_sequence(pos_w, batch_first=True, padding_value=0)
    return ids_w.to(device), msk_w.to(device), pos_w.to(device), offsets


def compute_ce_gate_credit(
    model: nn.Module,
    tokenizer,
    input_ids: Tensor,          # [B, T]
    attention_mask: Tensor,     # [B, T]
    rho_tool: float = 0.3,      # 工具响应占比
    length_norm: str = "none",  # "none" | "len" | "l2"
    temp: float = 1.0,          # softmax 温度
    chunk_logits: int = 256,    # logits 分块
    retain_graph: bool = True,  # 在 PPO/GRPO 图内复用时需要 True
    max_ctx: Optional[int] = None,
) -> Dict[str, List[Tensor]]:
    device = input_ids.device
    assert attention_mask.shape == input_ids.shape

    turns = get_turns(input_ids, attention_mask, tokenizer)
    asst_turns_all = turns["assistant_turns"]
    tool_turns_all = turns["tool_turns"]

    embed = _resolve_input_embeddings(model)
    embed_hook_target = _resolve_embedding_hook_target(model)
    forward_module = _unwrap(model)

    ids_w, msk_w, pos_w, offsets = _last_answer_window(
        input_ids, attention_mask, turns, max_ctx=max_ctx, device=device
    )
    # Ensure contiguous tensors to avoid view/in-place issues in downstream kernels
    ids_w = ids_w.contiguous()
    msk_w = msk_w.contiguous()
    B, Tw = ids_w.shape

    labels_w = torch.full_like(ids_w, fill_value=-100)
    for b in range(B):
        asst_turns = asst_turns_all[b]
        assert len(asst_turns) >= 1, "need at least one assistant turn"
        s_last, e_last = asst_turns[-1]
        left = offsets[b]
        rel_start = max(0, min(Tw, s_last - left))
        rel_end = max(0, min(Tw, e_last - left))
        if rel_end > rel_start:
            labels_w[b, rel_start:rel_end] = ids_w[b, rel_start:rel_end]

    Ns_total = int((labels_w[:, 1:] != -100).sum().item())
    if Ns_total == 0:
        raise RuntimeError("No valid supervised positions in last answer spans.")

    forward_kwargs = dict(
        input_ids=ids_w,
        attention_mask=msk_w,
        use_cache=False,
        return_dict=True,
    )
    if pos_w is not None:
        forward_kwargs["position_ids"] = pos_w

    # Attach tensor-level backward hooks on embedding outputs (checkpoint-friendly)
    grad_capture: Dict[str, Tensor] = {}
    hook_handles: List[Any] = []

    def _fwd_capture(_m: nn.Module, _inp: Tuple[Tensor, ...], out: Tensor):
        def _tensor_bwd_hook(g: Tensor):
            grad_capture["grad"] = g.detach()
            return g

        try:
            handle = out.register_hook(_tensor_bwd_hook)
            hook_handles.append(handle)
        except Exception:
            pass

    hook_handles.append(embed_hook_target.register_forward_hook(_fwd_capture))

    dname = device.type
    autocast_dtype = torch.bfloat16 if dname in ("cuda", "npu") else None
    if autocast_dtype is not None:
        with torch.autocast(device_type=dname, dtype=autocast_dtype):
            out = forward_module(**forward_kwargs)
    else:
        out = forward_module(**forward_kwargs)

    # Compute CE only on last assistant span like SFT does (outside the model)
    logits = out.logits  # [B, Tw, V]
    losses: List[Tensor] = []
    Ns_total = 0
    for b in range(B):
        asst_turns = asst_turns_all[b]
        assert len(asst_turns) >= 1
        s_last, e_last = asst_turns[-1]
        left = offsets[b]
        tgt_abs = torch.arange(s_last, e_last, device=device)
        pos_win = tgt_abs - left
        mask_ok = (pos_win >= 0) & (pos_win < Tw)
        if not mask_ok.any():
            continue
        pos_win = pos_win[mask_ok]
        hpos = (pos_win - 1).clamp_min(0)
        y = ids_w[b, pos_win]
        logits_sel = logits[b, hpos, :]
        losses.append(torch.nn.functional.cross_entropy(logits_sel, y, reduction="sum"))
        Ns_total += logits_sel.size(0)

    if Ns_total == 0:
        raise RuntimeError("No valid supervised positions in last answer spans.")
    loss_ce = torch.stack(losses).sum() / max(1, Ns_total)
    print('loss_ce', loss_ce)
    # Trigger backward to populate grad_capture from the tensor-level hook
    try:
        loss_ce.backward(retain_graph=retain_graph)
    finally:
        for handle in hook_handles:
            try:
                handle.remove()
            except Exception:
                pass

    if "grad" not in grad_capture:
        raise RuntimeError("Failed to capture embedding gradients during backward.")

    grad_embed = grad_capture["grad"]

    # Clear gradients to avoid polluting subsequent optimizer steps
    for param in forward_module.parameters():
        if param.grad is not None:
            param.grad = None
    grad_token = (
        grad_embed.detach().abs().sum(dim=-1) * msk_w.float()
    ).to(torch.float32)

    token_credit_list: List[Tensor] = []
    s_turn_list: List[Tensor] = []
    w_turn_list: List[Tensor] = []

    for b in range(B):
        cred_full = torch.zeros_like(input_ids[b], dtype=torch.float32, device=device)
        left = offsets[b]
        fill_len = min(Tw, cred_full.size(0) - left)
        if fill_len > 0:
            cred_full[left:left + fill_len] = grad_token[b, :fill_len]
        token_credit_list.append(cred_full)

        asst_turns = asst_turns_all[b]
        tool_turns = tool_turns_all[b]
        K = len(asst_turns)
        s_turn = torch.zeros(K, device=device, dtype=torch.float32)
        len_turn = torch.zeros(K, device=device, dtype=torch.float32)
        l2_turn = torch.zeros(K, device=device, dtype=torch.float32)
        for k in range(K):
            a_span = asst_turns[k]
            tr_span = tool_turns[k] if k < len(tool_turns) else None
            sA = _sum_span(cred_full, a_span)
            sTR = _sum_span(cred_full, tr_span)
            s_turn[k] = sA + rho_tool * sTR

            s_k, e_k = a_span
            Lk = max(1, e_k - s_k)
            len_turn[k] = Lk
            seg = cred_full[s_k:e_k]
            l2_turn[k] = float(torch.sqrt((seg ** 2).sum() + 1e-12).item())

        if length_norm == "len":
            s_turn = s_turn / (len_turn + 1e-12)
        elif length_norm == "l2":
            s_turn = s_turn / (l2_turn + 1e-12)

        if K > 4:
            hi = torch.quantile(s_turn, 0.98)
            s_turn = torch.minimum(s_turn, hi)

        w_turn = torch.softmax(s_turn / max(1e-6, temp), dim=-1)
        s_turn_list.append(s_turn.detach())
        w_turn_list.append(w_turn.detach())

    return {
        "s_turn": s_turn_list,
        "w_turn": w_turn_list,
        "token_credit": token_credit_list,
        "turns": turns,
        "loss_ce": loss_ce.detach(),
    }
