from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_TOP_K
from app.shared.runtime.logger import logger


def validate_and_get_data(state: QueryGraphState) -> tuple[list[dict], list[dict]]:
    embedding_chunks: list = state.get("embedding_chunks")
    embedding_chunks_hyde: list = state.get("embedding_chunks_hyde")
    if (not embedding_chunks) or (not embedding_chunks_hyde):
        logger.error(f"embeeding_chunks或embedding_chunks_hyde为空,业务终止")
        raise ValueError(f"embeeding_chunks或embedding_chunks_hyde为空,业务终止")
    return embedding_chunks, embedding_chunks_hyde


def reciprocal_rank_fusion(deal_result_list:list[tuple[float,list[dict]]], k:int=60) -> list[dict]:
    dict_score:dict[str, float] = {}
    dict_chunk:dict[str, dict] = {}

    for weight, list_chunks in deal_result_list:
        for rank, chunk in enumerate(list_chunks,start=1):
            chunk_id:str = chunk.get("chunk_id")
            current_score = dict_score.get(chunk_id,0.0) + weight * (1/(rank+k))
            dict_chunk[chunk_id] = chunk
    rrf_chunks:list[dict]=[]
    for chunk_id, chunk in dict_chunk.items():
        chunk["score"] = dict_score.get(chunk_id,0.0)
        rrf_chunks.append(chunk)
    rrf_chunks.sort(key=lambda x: x.get("score",0.0), reverse=True)
    rrf_chunks = rrf_chunks[:NODE_RRF_TOP_K]
    return rrf_chunks


def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    embedding_chunks, embedding_chunks_hyde = validate_and_get_data(state)
    deal_result_list:list[tuple[float,list[dict]]] = [
        (0.6,embedding_chunks),
        (0.4,embedding_chunks_hyde),
    ]
    rrf_chunks: list[str] = reciprocal_rank_fusion(deal_result_list)
    state["rrf_chunks"] = rrf_chunks
    return state