from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pymilvus import AnnSearchRequest

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import ITEM_NAME_CONFIRM_THRESHOLD, ITEM_NAME_CANDIDATE_THRESHOLD, ITEM_NAME_OPTIONS_TOPK
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_and_get_data(state:QueryGraphState) -> tuple[str,str]:
    # state获取数据
    session_id, original_query = state.get("session_id"), state.get("original_query")
    # 非空校验  空 -> ValueError / 非空
    if not session_id or not original_query:
        logger.error("session_id or original_query is None")
        raise ValueError("session_id or original_query is None")
    return session_id, original_query


def valid_chat_message_list(session_id:str) -> list[dict]:
    recent_message_list:list[dict] = history_repository.list_recent(session_id=session_id, limit = 20)
    # 条件1: 必须有item_names | 条件2: 过滤 role = user
    valid_message_list:list[dict] = [
        item
        for item in recent_message_list if item.get("item_names") and item.get("role") == "user"
    ]
    logger.info(f"recent message_list total: {len(recent_message_list)},valid message_list: {valid_message_list}")
    return valid_message_list


def call_llm_confirm_item_name_and_rewritten_query(
    message_list: list[dict],
    original_query: str,
) -> dict[str, str | list[str]]:
    """
    调用大模型：改写用户问题 + 提取/确认商品主体名称。
    返回：{"rewritten_query": str, "item_names": list[str]}
    """
    llm_client = llm_provider.chat(json_model=True)
    # 拼接历史上下文
    if message_list:
        history_text = "\n".join(
            f"{index},原始问题:{msg.get('text')},"
            f"重写后的问题:{msg.get('rewritten_query')},"
            f"关联主体: {'、'.join(msg.get('item_names') or [])}"
            for index, msg in enumerate(message_list, start=1)
        )
    else:
        history_text = "没有有效聊天记录"

    # 加载提示词 + 构造消息
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text=history_text,
        query=original_query,
    )
    messages = [HumanMessage(content=prompt)]
    chains = llm_client | JsonOutputParser()
    result:dict =  chains.invoke(messages)

    if "rewritten_query" not in result:
        logger.warning(f"rewritten_query is None, rewritten_query: {original_query}")
        result["rewritten_query"] = original_query

    if "item_names" not in result:
        logger.warning(f"item_names is None, item_names: {original_query}")
        result["item_names"] = []

    logger.info(f"完成主体识别: {result}")
    # 返回结果
    return result


def milvus_hybrid_search(item_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    """
    对每个 item_name 做稠密+稀疏混合检索，
    返回 {模型item_name: [{"item_name": 检索到的名称, "score": 分数}, ...]}
    """
    # milvus_client = milvus_gateway.client
    milvus_search_result:dict[str,list[dict[str,Any]]] = {}
    for item_name_llm in item_names:
        result = llm_provider.generate_infra_embeddings([item_name_llm])
        dense_vector = result['dense'][0]
        sparse_vector = result['sparse'][0]
        reqs:list[AnnSearchRequest] = milvus_gateway.create_requests(dense_vector=dense_vector, sparse_vector=sparse_vector,limit=5*2)

        milvus_result = milvus_gateway.hybrid_search(
                # 修复原有缺陷：主体确认应在「主体名称集合」中检索，
                # 此前误检索 chunks 集合，导致分数恒低于阈值、主体无法确认。
                collection_name=milvus_gateway.item_name_collection,
                reqs=reqs,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["item_name", "file_title"],
            )

        if milvus_result and milvus_result[0]:
            milvus_item_name_list:list[dict[str,Any]] = [
                {
                    "item_name": item.get("entity",{}).get("item_name"),
                    "score": item.get("distance",0.0),
                }
                for item in milvus_result[0]
            ]
            logger.debug(f"model recognized item_name:{item_name_llm}, milvus_item_name_list: {milvus_item_name_list}")
            milvus_search_result[item_name_llm] = milvus_item_name_list
        else:
            logger.warning(f"model recognized item_name:{item_name_llm},data is none.")

    return milvus_search_result


def select_item_names(milvus_result: dict[str,list[dict[str, Any]]]) ->dict[str,list[str]]:
    confirmed_item_name_list:list[str] = []
    optional_item_name_list:list[str] = []

    for _,milvus_item_list in milvus_result.items():
        confirmed_list:list[dict] = [
            item for item in milvus_item_list
            if item.get("score",0.0) > ITEM_NAME_CONFIRM_THRESHOLD
        ]
        if confirmed_list:
            confirmed_item_name_list.append(confirmed_list[0].get("item_name"))
            continue
        optional_list:list[dict] = [
            item for item in milvus_item_list
            if ITEM_NAME_CANDIDATE_THRESHOLD < item.get("score",0.0) < ITEM_NAME_CONFIRM_THRESHOLD
        ]
        if optional_list:
            optional_item_name_list.extend([item.get("item_name") for item in optional_list[:ITEM_NAME_OPTIONS_TOPK]])
            continue

    return {
        "confirmed_item_name_list": confirmed_item_name_list,
        "optional_item_name_list": optional_item_name_list,
    }


def update_query_state(item_list:dict[str,list[str]],rewritten_query:str,state:QueryGraphState):
    confirm_item_name_list:list[str] = item_list.get("confirmed_item_name_list",[])
    optional_item_name_list:list[str] = item_list.get("optional_item_name_list",[])
    # 无论主体是否确认，都先写入改写后的问题，供后续检索节点使用
    # （未确认主体时会退化为全库检索，同样需要 rewritten_query）
    if rewritten_query:
        state["rewritten_query"] = rewritten_query
    if confirm_item_name_list:
        state["item_names"] = confirm_item_name_list
        return
    if optional_item_name_list:
        state["answer"] = f"未明确识别到您想查询的主体，您是否想了解：{'、'.join(optional_item_name_list)}？请补充说明后重试。"
        return

    # 未识别到主体：不在这里终止。由 main_graph.after_node_item_name_confirm 决定
    # 退化为「无元数据过滤」的全库检索，以便回答「某个主题的解释」类问题。
    logger.info("未识别到主体,交由后续节点退化为全库检索")


def save_user_chat_message(session_id: str, message: str, rewritten_query: str, item_names: list[str]):
    """持久化用户消息（修复原有参数错位：text 应为用户原始问题）。"""
    history_repository.save_message(
        session_id=session_id,
        role="user",
        text=message,
        rewritten_query=rewritten_query,
        item_names=item_names,
        image_urls=[]
    )
    logger.info(f"history saved {'validly' if item_names else 'invalidly'} ")


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    session_id, original_query = validate_and_get_data(state)
    valid_message_list: list[dict] = valid_chat_message_list(session_id=session_id)
    result = call_llm_confirm_item_name_and_rewritten_query(valid_message_list, original_query)
    milvus_search_result: dict[str, list[dict[str, Any]]] = milvus_hybrid_search(item_names=result.get("item_names") or [])
    item_name_dict: dict[str, list[str]] = select_item_names(milvus_search_result)
    update_query_state(item_name_dict, result.get("rewritten_query"), state)
    save_user_chat_message(
        session_id=session_id,
        message=original_query,
        rewritten_query=result.get("rewritten_query"),
        item_names=state.get("item_names", []),
    )
    return state