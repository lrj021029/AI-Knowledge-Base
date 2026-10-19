from pymilvus import AnnSearchRequest
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.query.chunk_utils import build_output_fields, normalize_retrieved_chunk
from app.rag.query.metadata_filter_service import build_filter_expr


def validate_and_get_data(state: QueryGraphState) -> tuple[str, list[str]]:
    item_names: list[str] = state.get("item_names")
    rewritten_query: str = state.get("rewritten_query")
    # 听书业务分支可能没有 item_name，但一定有改写后的问题
    if not rewritten_query:
        logger.error(f"rewritten_query为空,业务终止")
        raise ValueError(f"rewritten_query为空,业务终止")
    return rewritten_query, item_names


def rewritten_query_embedding(rewritten_query: str) -> tuple[list, dict]:
    result = llm_provider.generate_infra_embeddings([rewritten_query])

    dense_vector = result['dense'][0]
    sparse_vector = result['sparse'][0]
    return dense_vector, sparse_vector


def milvus_hybrid_search(expr: str | None, dense_vector, sparse_vector) -> list[dict]:
    reqs: list[AnnSearchRequest] = milvus_gateway.create_requests(
        dense_vector=dense_vector, sparse_vector=sparse_vector, limit=2 * 5, expr=expr
    )
    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=build_output_fields(),
    )

    if resp and resp[0]:
        logger.info(f"完成数据查询,共查询到数据:{len(resp[0])}")
        return resp[0]
    logger.warning(f"没有查询到任何数据")
    return []


def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    # 获取参数和校验
    rewritten_query, item_names = validate_and_get_data(state)
    # 重写问题进行向量化处理
    dense_vector, sparse_vector = rewritten_query_embedding(rewritten_query)
    # 构造过滤条件（听书元数据优先，回退 item_name）
    expr: str | None = build_filter_expr(state)
    # 进行数据混合检索
    milvus_result: list[dict] = milvus_hybrid_search(expr, dense_vector, sparse_vector)
    # 若因「内容类型」约束查不到数据，则放宽该条件重试一次
    if not milvus_result and (state.get("content_types") or []):
        relaxed_expr: str | None = build_filter_expr(state, ignore_content_type=True)
        if relaxed_expr and relaxed_expr != expr:
            logger.info("按内容类型过滤无结果,放宽内容类型约束后重试")
            milvus_result = milvus_hybrid_search(relaxed_expr, dense_vector, sparse_vector)
    # 进行结果归一化处理
    if not milvus_result:
        logger.warning(f"问题:{rewritten_query} , item_names:{item_names} 没有匹配到任何数据!")
        state['embedding_chunks'] = []
        return state
    embedding_chunks: list[dict] = normalize_retrieved_chunk(milvus_result)
    # 修改state
    state['embedding_chunks'] = embedding_chunks
    # 返回结果
    return state
