from langgraph.graph import StateGraph, END
from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_intent_recognize import node_intent_recognize
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import (
    INTENT_BOOK_DETAIL,
    INTENT_BOOK_RECOMMEND,
    INTENT_NOTE_SEARCH,
)
from app.shared.runtime.logger import logger

# 仅依赖知识库内已入库文档的意图（SPEC 约束：推荐/详情无法获取外部在线书籍数据）
# 这些意图不接入联网检索，避免外部书籍污染推荐/详情答案
_LOCAL_ONLY_INTENTS = {INTENT_BOOK_RECOMMEND, INTENT_BOOK_DETAIL}

query_graph_builder = StateGraph(state_schema=QueryGraphState)

query_graph_builder.add_node(node_answer_output)
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_intent_recognize)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)

# 入口：先做意图识别（分流 + 构造过滤条件）
query_graph_builder.set_entry_point("node_intent_recognize")


def after_node_intent_recognize(state: QueryGraphState):
    """
    意图分流：
    - 书籍推荐/详情：依赖知识库内已入库文档，无法获取外部在线书籍数据，
      只走本地多路召回（HyDE + 向量检索），不接入联网检索。
    - 听书笔记检索：查询书评/笔记，允许本地 + 联网结果共同送入 Reranker。
    - 通用问答：走原有 item_name 确认逻辑。
    """
    intent: str = state.get("intent") or ""
    if intent in _LOCAL_ONLY_INTENTS:
        logger.info(f"识别到听书业务意图:{intent},仅走本地检索分支（推荐/详情不接入联网）")
        return "node_search_embedding_hyde", "node_search_embedding"
    if intent == INTENT_NOTE_SEARCH:
        logger.info(f"识别到听书笔记检索意图:{intent},进入本地+联网检索分支")
        return "node_search_embedding_hyde", "node_search_embedding", "node_web_search_mcp"
    logger.info(f"识别到通用问答意图:{intent},进入原有 RAG 分支")
    return "node_item_name_confirm"


# 判断是否存在 item_name
def after_node_item_name_confirm(state: QueryGraphState):
    # 1) 主体已确认：正常继续检索
    if state.get("item_names"):
        logger.info(f"识别到 item_name: {state.get('item_names')},业务继续")
        return "node_search_embedding_hyde", "node_search_embedding", "node_web_search_mcp"
    # 2) 主体不明确但检索到了候选：直接把「请选择主体」的话术作为答案返回（保留原交互）
    if state.get("answer"):
        logger.info("主体存在候选,返回澄清话术")
        return "node_answer_output"
    # 3) 完全没识别到主体（如「什么是黑暗森林法则」这类主题解释型问题）：
    #    需求 4.3 要求这类问题也要返回相关书籍资料，因此不再直接终止，
    #    而是退化为「不做元数据过滤」的全库检索（build_filter_expr 返回 None）。
    logger.info("未识别到 item_name,退化为全库检索继续作答")
    return "node_search_embedding_hyde", "node_search_embedding", "node_web_search_mcp"


path_map: dict = {
    "node_answer_output": "node_answer_output",
    "node_item_name_confirm": "node_item_name_confirm",
    "node_search_embedding_hyde": "node_search_embedding_hyde",
    "node_search_embedding": "node_search_embedding",
    "node_web_search_mcp": "node_web_search_mcp",
}
# 条件边：意图分流
query_graph_builder.add_conditional_edges("node_intent_recognize", after_node_intent_recognize, path_map)
# 条件边：通用问答主体确认
query_graph_builder.add_conditional_edges("node_item_name_confirm", after_node_item_name_confirm, path_map)

query_graph_builder.add_edge("node_search_embedding", "node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph_builder.add_edge("node_web_search_mcp", "node_rrf")
query_graph_builder.add_edge("node_rrf", "node_rerank")
query_graph_builder.add_edge("node_rerank", "node_answer_output")
query_graph_builder.add_edge("node_answer_output", END)

query_graph_app = query_graph_builder.compile()
