from typing import Any
from pymilvus import DataType
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.import_.config import (
    MILVUS_CHUNK_CONTENT_MAX_LENGTH,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)
from app.shared.config.milvus_config import DENSE_METRIC_TYPE, SPARSE_METRIC_TYPE

@step_log("validate_and_get_chunks")
def validate_and_get_chunks(state: ImportGraphState):
    # 优先读chunks
    embeddings_content = state.get("embeddings_content")
    if not embeddings_content:
        embeddings_content = state.get("chunks", [])

    if not embeddings_content:
        raise ValueError(f"embeddings_content为空,业务终止")
    return embeddings_content


@step_log("prepare_chunks_collection")
def prepare_chunks_collection():
    """准备切块集合，不存在则创建schema+索引"""
    milvus_client = milvus_gateway.client
    collection_name = milvus_gateway.chunks_collection

    if milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"集合 {collection_name} 已存在，跳过创建")
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name='chunk_id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name='parent_title', datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name='part', datatype=DataType.INT8)
    schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH)
    schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name='dense_vector',
        index_type='HNSW',
        index_name='dense_vector_index',
        metric_type=DENSE_METRIC_TYPE,
        index_params={
            "M": 64,
            "efConstruction": 100,
        }
    )
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        index_name='sparse_vector_index',
        metric_type=SPARSE_METRIC_TYPE,
        index_params={'inverted_index_algo': 'SINDI'}
    )
    milvus_client.create_collection(
        collection_name=milvus_gateway.chunks_collection,
        schema=schema,
        index_params=index_params
    )
    milvus_client.load_collection(
        collection_name=milvus_gateway.chunks_collection,
    )


@step_log("insert_update_chunks")
def insert_update_chunks(embeddings_content: list[dict[str,Any]]):
    """分批插入，返回全部自动生成chunk_id"""
    milvus_client = milvus_gateway.client
    file_title: str = embeddings_content[0].get("file_title")
    delete_result = milvus_client.delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"file_title == '{file_title}'",
    )
    delete_count:int = delete_result.get("delete_count", 0)
    logger.info(f"删除:{delete_count}条数据")
    insert_result = milvus_client.insert(
        collection_name=milvus_gateway.chunks_collection,
        data=embeddings_content,
    )
    insert_count:int = insert_result.get("insert_count", 0)
    logger.info(f"插入了{insert_result}条数据")
    ids:list = insert_result.get("ids", [])
    for id,chunk in zip(ids,embeddings_content):
        chunk['chunk_id'] = id


@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1.准备集合 schema 和索引
    2.根据 item_name 删除旧数据
    3.批量插入新的 chunks
    4.回写 chunk_id 到state内chunks
    """
    embeddings_content:list[dict[str,Any]] = validate_and_get_chunks(state)
    prepare_chunks_collection()
    insert_update_chunks(embeddings_content)
    return state
