# -*- coding: utf-8 -*-
# verl/trainers/hooks/credit_router.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
from torch import Tensor, nn

from verl.extensions.credit.ce_gate_credit import compute_ce_gate_credit

@dataclass
class CreditRouterConfig:
    enabled: bool = True
    rho_tool: float = 0.3          # 工具响应 credit 占比
    length_norm: str = "none"      # "none" | "len" | "l2"
    temp: float = 1.0              # 回合 softmax 温度
    chunk_logits: int = 256        # logits 分块
    adv_split_eta: float = 0.7     # A_k = ((1-eta)/K + eta*w_k) * A_group
    route_mode: str = "adv_split"  # "adv_split" | "shaping"

class GateCECreditRouter:
    """
    把 compute_ce_gate_credit 的回合权重，路由成：
      - 每回合优势 A_k（GRPO/组优势拆分）
      - 每 token 的权重 token_weight（可乘到 policy loss 的 per-token 项）
    """
    def __init__(self, cfg: CreditRouterConfig, model: nn.Module, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, batch: Dict[str, Tensor], group_adv: Tensor) -> Dict[str, Any]:
        """
        Args:
          batch: {"input_ids":[B,T], "attention_mask":[B,T]}
          group_adv: [B]（GRPO 每样本组优势）
        Returns:
          {
            "A_k": List[Tensor[K_b]],     # 每回合优势（stopgrad）
            "w_turn": List[Tensor[K_b]],  # 回合 softmax 权重
            "token_weight": Tensor[B,T],  # 每 token 权重（扩展 A_k 覆盖各回合 token 段）
            "aux": {...}
          }
        """
        if not self.cfg.enabled:
            return {"A_k": None, "w_turn": None, "token_weight": None, "aux": {}}

        res = compute_ce_gate_credit(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            rho_tool=self.cfg.rho_tool,
            length_norm=self.cfg.length_norm,
            temp=self.cfg.temp,
            chunk_logits=self.cfg.chunk_logits,
            retain_graph=False,
        )

        B, T = batch["input_ids"].shape
        token_weight = torch.zeros(B, T, device=batch["input_ids"].device, dtype=torch.float32)

        A_k_list: List[Tensor] = []
        w_turn_list: List[Tensor] = res["w_turn"]
        turns = res["turns"]["assistant_turns"]  # List[List[(s,e)]]
        eta = self.cfg.adv_split_eta

        for b in range(B):
            w_k = w_turn_list[b]   # [K_b]
            Kb = w_k.numel()
            A_grp = group_adv[b].detach()
            if self.cfg.route_mode == "adv_split":
                A_k = A_grp * ((1.0 - eta) / Kb + eta * w_k)
            else:  # shaping（示例：直接按 w_k 缩放）
                A_k = A_grp * w_k
            A_k_list.append(A_k.detach())

            # 扩展到 token 权重（仅 assistant token 段）
            for k, (s, e) in enumerate(turns[b]):
                if e > s:
                    token_weight[b, s:e] = A_k[k].item()

        return {
            "A_k": A_k_list,
            "w_turn": w_turn_list,
            "token_weight": token_weight,
            "aux": {
                "s_turn": res["s_turn"],
                "token_credit": res["token_credit"],
                "loss_ce_proxy": res["loss_ce"],
                "turns": res["turns"],
            },
        }
