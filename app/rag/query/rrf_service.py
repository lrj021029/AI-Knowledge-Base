from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_TOP_K
from app.shared.runtime.logger import logger


def validate_and_get_data(state: QueryGraphState) -> tuple[list[dict], list[dict]]:
    """校验并获取多路召回结果；允许某些通路为空（听书业务过滤后可能无召回）。"""
    embedding_chunks: list = state.get("embedding_chunks") or []
    embedding_chunks_hyde: list = state.get("embedding_chunks_hyde")
    if embedding_chunks_hyde is None:
        embedding_chunks_hyde = state.get("hyde_embedding_chunks") or []
    return embedding_chunks, embedding_chunks_hyde


def reciprocal_rank_fusion(deal_result_list: list[tuple[float, list[dict]]], k: int = 60) -> list[dict]:
    dict_score: dict[str, float] = {}
    dict_chunk: dict[str, dict] = {}

    for weight, list_chunks in deal_result_list:
        for rank, chunk in enumerate(list_chunks or [], start=1):
            chunk_id: str = chunk.get("chunk_id")
            current_score = dict_score.get(chunk_id, 0.0) + weight * (1 / (rank + k))
            dict_score[chunk_id] = current_score
            dict_chunk[chunk_id] = chunk
    rrf_chunks: list[dict] = []
    for chunk_id, chunk in dict_chunk.items():
        chunk_copy = dict(chunk)
        chunk_copy["score"] = dict_score.get(chunk_id, 0.0)
        rrf_chunks.append(chunk_copy)
    rrf_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    rrf_chunks = rrf_chunks[:NODE_RRF_TOP_K]
    return rrf_chunks


def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    embedding_chunks, embedding_chunks_hyde = validate_and_get_data(state)
    if not embedding_chunks and not embedding_chunks_hyde:
        logger.warning("多路召回结果均为空,RRF 融合跳过")
        state["rrf_chunks"] = []
        return state
    # 仅对非空通路做加权融合，避免空通路拉低分数
    deal_result_list: list[tuple[float, list[dict]]] = []
    if embedding_chunks:
        deal_result_list.append((0.6, embedding_chunks))
    if embedding_chunks_hyde:
        deal_result_list.append((0.4, embedding_chunks_hyde))

    rrf_chunks: list[dict] = reciprocal_rank_fusion(deal_result_list)
    state["rrf_chunks"] = rrf_chunks
    return state
