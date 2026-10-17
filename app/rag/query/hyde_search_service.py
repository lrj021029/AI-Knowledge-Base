from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import AnnSearchRequest
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.runtime.load_prompt import load_prompt


def validate_and_get_data(state:QueryGraphState) -> tuple[str,list[str]]:
    item_names:list[str] = state.get("item_names")
    rewritten_query:str  = state.get("rewritten_query")
    if (not item_names) and (not rewritten_query):
        logger.error(f"item_names或者rewritten_query为空,业务终止")
        raise ValueError(f"item_names或者rewritten_query为空,业务终止")
    return rewritten_query,item_names


def call_llm_generate_hyde_answer(rewritten_query:str)->str:
    llm_client = llm_provider.chat()
    hyde_prompt_str:str = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=hyde_prompt_str)
    ]
    chains = llm_client | StrOutputParser()
    hyde_answer:str =  chains.invoke(messages)
    return hyde_answer


def rewritten_query_embedding(rewritten_query:str,hyde_answer:str) -> tuple[list,dict]:
    result = llm_provider.generate_infra_embeddings([rewritten_query + ":" + hyde_answer])
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
        logger.info(f"完成数据查询,共查询到:{len(resp[0])}")
        return resp[0]
    logger.warning(f"未查询到任何数据")
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
    logger.info(f"完成结果的归一化处理数据总量为:{len(embedding_chunks)}, 获取第一条示例输出:\n {embedding_chunks[0]}")
    return embedding_chunks

def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    rewritten_query,item_names = validate_and_get_data(state)
    hyde_answer:str = call_llm_generate_hyde_answer(rewritten_query)
    dense_vector,sparse_vector = rewritten_query_embedding(rewritten_query,hyde_answer)
    milvus_result:list[dict] = milvus_hybrid_search(item_names,dense_vector,sparse_vector)
    if not milvus_result:
        logger.warning(f"问题:{rewritten_query} , item_names:{item_names} 没有匹配到任何数据!")
        return state
    embedding_chunks:list[dict] = normalize_retrieved_chunk(milvus_result)

    state['hyde_embedding_chunks'] = embedding_chunks
    return state