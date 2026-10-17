from pymilvus import AnnSearchRequest
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway


def validate_and_get_data(state:QueryGraphState) -> tuple[str,list[str]]:
    item_names:list[str] = state.get("item_names")
    rewritten_query:str  = state.get("rewritten_query")
    if (not item_names) and (not rewritten_query):
        logger.error(f"item_names或者rewritten_query为空,业务终止")
        raise ValueError(f"item_names或者rewritten_query为空,业务终止")
    return rewritten_query,item_names


def rewritten_query_embedding(rewritten_query:str) -> tuple[list,dict]:
    result = llm_provider.generate_infra_embeddings([rewritten_query])

    dense_vector  = result['dense'][0]
    sparse_vector = result['sparse'][0]
    return dense_vector,sparse_vector


def milvus_hybrid_search(item_names:list[str],dense_vector,sparse_vector) -> list[dict]:
    expr_str:str = f"item_name in {item_names}"
    reqs:list[AnnSearchRequest]  = milvus_gateway.create_requests(dense_vector=dense_vector,sparse_vector=sparse_vector,limit=2*5,expr=expr_str)
    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.5,0.5),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id","title","parent_title","part","file_title","item_name","content"]
    )

    if resp and resp[0]:
        logger.info(f"完成数据查询,共查询到数据:{len(resp[0])}")
        return resp[0]
    logger.warning(f"没有查询到任何数据")
    return []


def normalize_retrieved_chunk(chunks:list[dict]) -> list[dict]:
    embedding_chunks:list[dict] = [
        {
            "chunk_id": chunk.get("entity",{}).get("chunk_id"),
            "score": chunk.get("distance",0.0),
            "title": chunk.get("entity",{}).get("title"),
            "parent_title": chunk.get("entity",{}).get("parent_title"),
            "part": chunk.get("entity",{}).get("part"),
            "file_title": chunk.get("entity",{}).get("file_title"),
            "item_name": chunk.get("entity",{}).get("item_name"),
            "content": chunk.get("entity",{}).get("content"),
            "type": "milvus"
        }
        for chunk in chunks
    ]
    # 4.2 返回对应结果
    logger.info(f"完成结果的归一化处理,数据总量:{len(embedding_chunks)}, 获取第一条示例输出:\n {embedding_chunks[0]}")
    return embedding_chunks

def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    # 获取参数和校验
    rewritten_query,item_names = validate_and_get_data(state)
    # 重写问题进行向量化处理
    dense_vector,sparse_vector = rewritten_query_embedding(rewritten_query)
    # 进行数据混合检索
    milvus_result:list[dict] = milvus_hybrid_search(item_names,dense_vector,sparse_vector)
    # 进行结果归一化处理
    if not milvus_result:
        logger.warning(f"问题:{rewritten_query} , item_names:{item_names} 没有匹配到任何数据!")
        return state
    embedding_chunks:list[dict] = normalize_retrieved_chunk(milvus_result)
    # 修改state
    state['embedding_chunks'] = embedding_chunks
    # 返回结果
    return state