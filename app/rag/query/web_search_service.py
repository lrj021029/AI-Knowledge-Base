import asyncio
import json
from typing import Any
from agents.mcp import MCPServerStreamableHttp
from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


def validate_and_get_data(state:QueryGraphState)->str:
    rewritten_query:str = state.get("rewritten_query")
    if not(rewritten_query):
        logger.error(f"rewritten_query is None")
        raise ValueError("rewritten_query is None")
    return rewritten_query


async def call_bailian_mcp_server(rewritten_query:str) -> Any:
    async with MCPServerStreamableHttp(
        name="Bailian MCP Server",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 100,
        },
        cache_tools_list=True,
        max_retry_attempts=3
    ) as server:
        await server.connect()
        result = await server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 10
            }
        )
        logger.info(f"Bailian MCP Server 搜索完成: {rewritten_query}")
        return result


def normalize_web_search_result(bailian_result:Any)->list[dict]:
    result_str = bailian_result.content[0].text
    result_dict:dict = json.loads(result_str)
    result_list:list[dict] = result_dict.get("pages",[])
    web_search_docs:list[dict] = [
        {
            "content": item.get("content"),
            "title": item.get("title"),
            "url": item.get("url"),
            "type": "web"
        }
        for item in result_list
    ]
    logger.info(f"搜索到{len(web_search_docs)}")
    return web_search_docs


def search_by_web(state: QueryGraphState) -> QueryGraphState:
    rewritten_query:str = validate_and_get_data(state)
    bailian_result = asyncio.run(call_bailian_mcp_server(rewritten_query))
    web_search_docs:list[dict] = normalize_web_search_result(bailian_result)
    state["web_search_docs"] = web_search_docs
    return state