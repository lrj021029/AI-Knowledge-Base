"""
Milvus 配置模块，负责读取向量库相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str

# 稠密向量统一度量：Milvus 要求“建索引”与“检索”的 metric_type 必须一致，
# 因此集中定义，避免建库用 COSINE 而检索用 IP 导致 hybrid_search 报错。
DENSE_METRIC_TYPE: str = env_str("MILVUS_DENSE_METRIC_TYPE") or "COSINE"
# 稀疏向量统一度量（BGE-M3 稀疏向量适配内积）
SPARSE_METRIC_TYPE: str = env_str("MILVUS_SPARSE_METRIC_TYPE") or "IP"


@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str

milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL"),
    chunks_collection=env_str("CHUNKS_COLLECTION"),
    entity_name_collection=env_str("ENTITY_NAME_COLLECTION"),
    item_name_collection=env_str("ITEM_NAME_COLLECTION"),
)
