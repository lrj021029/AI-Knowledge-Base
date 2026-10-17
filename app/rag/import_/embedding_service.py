from typing import Any
from app.shared.runtime.logger import logger, step_log
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.process.import_.agent.state import ImportGraphState
from app.infra.llm.providers import llm_provider


@step_log("require_chunks")
def require_chunks(state: ImportGraphState) -> list[dict[str, Any]]:
    """校验切块，空则抛异常"""
    chunks:list[dict[str,Any]] = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空,业务终止!")
        raise ValueError("chunks为空,业务终止!")
    return chunks


@step_log("batch_chunks_embedding")
def batch_chunks_embedding(chunks: list[dict[str,Any]]):
    """
    批量为文本切片生成稠密和稀疏向量
    功能：分批拼接文本 → 调用 Embedding 模型 → 绑定向量字段 → 异常隔离
    :param chunks: 文档切片列表，每个元素包含 item_name、content 等字段
    :param step: 批次大小，默认由配置 EMBEDDING_BATCH_SIZE 控制
    :return: 带向量字段的切片列表
    """
    # 获取总切片数
    total = len(chunks)

    # 按批次大小循环处理，每次处理 step 条切片
    for index in range(0, total, EMBEDDING_BATCH_SIZE):
        sub_chunks:list[dict[str,Any]] = chunks[index:index + EMBEDDING_BATCH_SIZE]
        item_name_content:list[str] = []
        for sub_chunk in sub_chunks:
            item_name_content.append(sub_chunk["item_name"]+":"+sub_chunk["content"])
        result:dict[str,list] = llm_provider.generate_infra_embeddings(
            item_name_content
        )
        for sub_index in range(len(sub_chunks)):
            dense_vector = result['dense'][sub_index]
            sparse_vector = result['sparse'][sub_index]
            sub_chunks[sub_index]["dense_vector"] = dense_vector
            sub_chunks[sub_index]["sparse_vector"] = sparse_vector
    logger.info(f"已完成所有切片向量化,打印示例:{chunks[0]}")


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    chunks:list[dict[str,Any]] = require_chunks(state)
    batch_chunks_embedding(chunks)
    state["embeddings_content"] = chunks
    return state