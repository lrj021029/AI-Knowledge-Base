from langgraph.graph import StateGraph, END
from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

query_graph_builder = StateGraph(state_schema=QueryGraphState)

query_graph_builder.add_node(node_answer_output)
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)

query_graph_builder.set_entry_point("node_item_name_confirm")
# 判断是否存在 item_name
def after_node_item_name_confirm(state:QueryGraphState):
    if not state.get("item_names"):
        logger.info(f"未识别到 item_name,业务终止")
        return "node_answer_output"
    else:
        logger.info(f"识别到 item_name: {state.get('item_names')},业务继续")
        return "node_search_embedding_hyde","node_search_embedding","node_web_search_mcp"



path_map:dict = {
    node_answer_output:node_answer_output,
    "node_search_embedding_hyde":"node_search_embedding_hyde",
    "node_search_embedding":"node_search_embedding",
    "node_web_search_mcp":"node_web_search_mcp",
}
# 条件边
query_graph_builder.add_conditional_edges("node_item_name_confirm",after_node_item_name_confirm, path_map)

query_graph_builder.add_edge("node_search_embedding","node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde","node_rrf")
query_graph_builder.add_edge("node_web_search_mcp","node_rrf")
query_graph_builder.add_edge("node_rrf","node_rerank")
query_graph_builder.add_edge("node_rerank","node_answer_output")
query_graph_builder.add_edge("node_answer_output",END)

query_graph_app = query_graph_builder.compile()