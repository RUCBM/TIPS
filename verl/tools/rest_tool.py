import json, logging, os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RESTMCPTool(BaseTool):
    """
    与 BaseTool 风格一致的 REST 兜底工具类：
      - self.name 使用裸工具名（MCP Manager 会在调用时补齐 server）
      - create/execute/release 生命周期与现有一致
      - execute() 通过 AgentCPMMCPClineManager.MCPManager.call_tool(...) 走 REST
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)
        # Character-based truncation only (simple, no tokenizer required)
        self.max_text_length: Optional[int] = config.get("max_text_length")

    def _truncate_text(self, text: str) -> tuple[str, bool]:
        """Truncate tool text by characters. Returns (new_text, truncated_flag)."""
        if isinstance(self.max_text_length, int) and self.max_text_length > 0:
            if len(text) > self.max_text_length:
                return text[: self.max_text_length] + "\n...[truncated]", True
        return text, False

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    async def _call(self, instance_id: str, parameters: dict[str, Any]) -> tuple[str, dict]:
        # 从我们刚写的 REST Manager 调用（保持现有 import 习惯）
        from verl.tools.utils.mcp_clients.AgentCPMMCPClientManager import MCPManager as RESTManager

        # 约定：RESTManager 是全局单例或在外层初始化后已存在 —— 如果不是，你也可以做成模块级全局
        # 这里直接用类属性/单例缓存以避免重复初始化
        if not hasattr(RESTMCPTool, "_rest_manager") or RESTMCPTool._rest_manager is None:
            raise RuntimeError("REST MCPManager is not initialized. Call initialize_agentcpm_mcp_tool first.")

        mgr: RESTManager = RESTMCPTool._rest_manager
        result = await mgr.call_tool(self.name, parameters, timeout=self.timeout)
        # 规范化状态：优先使用 payload.status/status_code；否则退回 result.status
        status_val = result.status if hasattr(result, "status") else "unknown"
        status_code = None
        detail = None
        raw = getattr(result, "raw", None)
        if isinstance(raw, dict):
            if raw.get("status") is not None:
                status_val = str(raw.get("status"))
            if raw.get("status_code") is not None:
                try:
                    status_code = int(raw.get("status_code"))
                    status_val = str(status_code)
                except Exception:
                    status_val = str(raw.get("status_code"))
            if raw.get("detail"):
                try:
                    detail = str(raw.get("detail"))
                except Exception:
                    detail = None

        meta = {"status": status_val}
        if status_code is not None:
            meta["status_code"] = status_code
        if detail:
            meta["detail"] = detail

        # 返回文本与元信息
        return (result.content[0].text if result.content else ""), meta

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        if not parameters:
            msg = "parameters is empty"
            logger.error(msg)
            return ToolResponse(text=json.dumps({"error": msg}, ensure_ascii=False)), 0.0, {}
        try:
            text, meta = await self._call(instance_id, parameters)
            orig_len = len(text)
            text, truncated = self._truncate_text(text)
            final_len = len(text)
            ratio = (final_len / orig_len) if orig_len > 0 else 1.0
            self._instance_dict[instance_id]["reward"].append(text.strip())
            # 判定是否失败：优先使用显式 status_code >= 400；否则 status 不在 ok_set
            status_str = str(meta.get("status", ""))
            status_code = meta.get("status_code", None)
            if isinstance(status_code, str):
                try:
                    status_code = int(status_code)
                    meta["status_code"] = status_code
                except Exception:
                    status_code = None
            ok_set = {"success", "ok", "succeeded", "200", "true"}
            is_error = False
            if isinstance(status_code, int):
                is_error = status_code >= 400
            elif status_str != "":
                is_error = status_str.lower() not in ok_set
            if not is_error:
                # 统一成功标签，保证下游统计能够识别 success
                status_str = "success"
                meta["status"] = status_str
                if status_code is None:
                    status_code = 200
                    meta["status_code"] = status_code

            metrics = {
                # 仅用于聚合层识别为 REST 工具，不做额外分桶输出
                "tool_kind": "rest_tool",
                "status": status_str,
                "truncated": truncated,
                "tool_text_len": final_len,
                "tool_text_len_orig": orig_len,
                "truncation_ratio": ratio,
            }
            if status_code is not None:
                metrics["status_code"] = status_code
            if is_error:
                # 填充 error 字段，方便聚合层将其计为失败并归类
                if isinstance(status_code, int):
                    metrics["error"] = f"http:{status_code}"
                elif meta.get("detail"):
                    metrics["error"] = f"detail:{meta['detail'][:200]}"
                else:
                    metrics["error"] = f"status:{status_str}"
            return ToolResponse(text=text), 0.0, metrics
        except Exception as e:
            err = json.dumps({"error": f"Tool exec failed: {e}"}, ensure_ascii=False)
            return ToolResponse(text=err), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # 你原实现是返回字符串数组；若需要数值，这里可以自定义
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
