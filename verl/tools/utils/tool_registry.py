# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import importlib
import json, logging
import os
import sys
import threading
from enum import Enum

from omegaconf import OmegaConf

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.utils.mcp_clients.AgentCPMMCPClientManager import MCPManager as RESTManager
from verl.tools.rest_tool import RESTMCPTool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    NATIVE = "native"
    MCP = "mcp"
    AgentCPMMCP = "agent-cpm-mcp"


async def initialize_mcp_tool(tool_cls, tool_config) -> list:
    from verl.tools.utils.mcp_clients.McpClientManager import ClientManager

    tool_list = []
    mcp_servers_config_path = tool_config.mcp.mcp_servers_config_path
    tool_selected_list = tool_config.mcp.tool_selected_list if "tool_selected_list" in tool_config.mcp else None
    await ClientManager.initialize(mcp_servers_config_path, tool_config.config.rate_limit)
    # Wait for MCP client to be ready
    max_retries = 10
    retry_interval = 2  # seconds
    for i in range(max_retries):
        tool_schemas = await ClientManager.fetch_tool_schemas(tool_selected_list)
        if tool_schemas:
            break
        if i < max_retries - 1:
            logger.debug(f"Waiting for MCP client to be ready, attempt {i + 1}/{max_retries}")
            await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError("Failed to initialize MCP tools after maximum retries")
    # mcp registry
    assert len(tool_schemas), "mcp tool is empty"
    for tool_schema_dict in tool_schemas:
        logger.debug(f"tool_schema_dict: {tool_schema_dict}")
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)
    return tool_list

async def initialize_agentcpm_mcp_tool(tool_cls, tool_config) -> list:
    """
    - 从 mcp_servers_config_path 读取 base_url（键：mcpServers.http-agentmcp.url）
    - 初始化 REST Manager：拉 /servers 与 /tools
    - 用 REST 返回的 OpenAI function schema 批量实例化 tool_cls（建议传 RESTMCPTool）
    """
    tool_list = []

    with open(tool_config.mcp.mcp_servers_config_path, "r") as f:
        cfg = json.load(f)

    assert "mcpServers" in cfg and isinstance(cfg["mcpServers"], dict), "缺少 mcpServers"
    assert "http-agentmcp" in cfg["mcpServers"] and "url" in cfg["mcpServers"]["http-agentmcp"], \
        "配置需包含 mcpServers.http-agentmcp.url 指向 /mcpapi 基地址"

    base = cfg["mcpServers"]["http-agentmcp"]["url"]
    timeout = tool_config.config.timeout
    retries = tool_config.config.retries
    rate_limit = None
    try:
        rl_val = tool_config.config.get("rate_limit", None)
        if rl_val is not None:
            rate_limit = int(rl_val)
    except Exception as exc:
        logger.warning(f"Invalid rate_limit value '{tool_config.config.get('rate_limit', None)}': {exc}")
        rate_limit = None
    # 读取可选的 browser_agent 配置并转为普通 dict
    browser_agent = None
    try:
        if "browser_agent" in tool_config.config:
            browser_agent = OmegaConf.to_container(tool_config.config.browser_agent, resolve=True)
    except Exception:
        browser_agent = None
    logger.info(f"mcp base: {base}")
    logger.info(f"mcp timeout: {timeout}")
    logger.info(f"mcp retries: {retries}")
    if rate_limit:
        logger.info(f"mcp rate_limit per tool: {rate_limit}")


    # 1) 初始化 REST Manager，并缓存到 RESTMCPTool 的类属性，供所有实例共享
    rest_mgr = RESTManager(
        manager_url=base,
        timeout=timeout,
        retries=retries,
        browser_agent=browser_agent,
        rate_limit=rate_limit,
    )
    ok = await rest_mgr.initialize()

    assert ok, "REST MCPManager 初始化失败"
    RESTMCPTool._rest_manager = rest_mgr

    # 2) 选择要加载的工具（可用 tool_selected_list 白名单）
    selected = set(tool_config.mcp.tool_selected_list) if "tool_selected_list" in tool_config.mcp else None
    schemas = await rest_mgr.fetch_tool_schemas(list(selected) if selected else None)
    assert schemas, "mcp tool is empty (REST)"
    assert issubclass(tool_cls, RESTMCPTool), "REST 接法 仅支持 RESTMCPTool 作为工具类"

    # 3) 批量实例化工具（工具类建议用 RESTMCPTool；如果传入自定义子类也行）
    for schema_dict in schemas:
        try:
            tool_schema = OpenAIFunctionToolSchema.model_validate(schema_dict)
        except Exception as e:
            logger.debug(f"schema_dict: {schema_dict} Exception: {e}")
            continue
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)

    return tool_list

def get_tool_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name)
    return tool_cls


def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    # Use a temporary event loop in a new thread because event
    # loop may already exist in new async architecture while retaining
    # backwards compatibility
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=tmp_event_loop.run_forever, name="mcp tool list fetcher", daemon=True)

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()

        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        for tool_config in tools_config.tools:
            cls_name = tool_config.class_name
            tool_type = ToolType(tool_config.config.type)
            tool_cls = get_tool_class(cls_name)

            match tool_type:
                case ToolType.NATIVE:
                    if tool_config.get("tool_schema", None) is None:
                        tool_schema = None
                    else:
                        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                    tool = tool_cls(
                        config=OmegaConf.to_container(tool_config.config, resolve=True),
                        tool_schema=tool_schema,
                    )
                    tool_list.append(tool)
                case ToolType.MCP:
                    mcp_tools = run_coroutine(initialize_mcp_tool(tool_cls, tool_config))
                    tool_list.extend(mcp_tools)
                case ToolType.AgentCPMMCP:
                    mcp_tools = run_coroutine(initialize_agentcpm_mcp_tool(tool_cls, tool_config))
                    tool_list.extend(mcp_tools)
                case _:
                    raise NotImplementedError
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()

    return tool_list
