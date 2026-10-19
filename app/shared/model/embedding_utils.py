"""
工具模块，负责提供 embedding 相关的辅助能力。
"""
import threading

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from app.shared.config.embedding_config import embedding_config
from app.shared.runtime.logger import logger

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_DEVICE = "cpu"
_bge_m3_ef: BGEM3EmbeddingFunction | None = None
# 模型加载锁：多任务（多文件导入 / 查询与导入并发）会同时触发首次加载，
# 并发加载同一个 fp16 模型会因显存/内存不足导致加载中断，
# 表现为 NotImplementedError: Cannot copy out of meta tensor; no data!
# 这里用锁保证「全局只加载一次」，其余调用方等待后复用同一实例。
_bge_m3_lock = threading.Lock()
# 编码锁：FlagEmbedding 每次 encode 都会调用 self.model.half()（dtype 转换），
# 转换为原地修改参数；若两个检索分支（如 embedding 检索与 HyDE 检索）首次并发编码，
# 一个线程正在 half() 改参数、另一个线程同时在 forward，会抛
# RuntimeError: expected scalar type Float but found Half（服务重启后的首个复杂问题必现）。
# 这里串行化编码调用，代价是一次只有一个编码在跑，与 MPS 单设备本身也基本等价。
_bge_m3_encode_lock = threading.Lock()


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 并发保护：多个线程同时首次调用时，只允许一个真正加载，其余等待复用
    with _bge_m3_lock:
        if _bge_m3_ef is not None:
            logger.debug("BGE-M3模型单例已由其他线程加载完成，直接返回实例")
            return _bge_m3_ef
        return _load_bge_m3_ef()


def _load_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """真正执行模型加载（调用方需持有 _bge_m3_lock）。"""
    global _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or _DEFAULT_EMBEDDING_MODEL
    device = embedding_config.bge_device or _DEFAULT_EMBEDDING_DEVICE
    use_fp16 = embedding_config.bge_fp16

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化 BGE-M3 模型，开启原生 L2 归一化（适配 Milvus IP 内积检索）
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # 模型原生对稠密+稀疏向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts: list[str]) -> dict[str, list]:
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if any(not isinstance(text, str) for text in texts):
        logger.warning("生成向量入参不合法，texts中存在非字符串内容")
        raise ValueError("参数texts必须是字符串列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        # 串行化编码：避免并发编码时 half() 改 dtype 与 forward 竞争（见 _bge_m3_encode_lock）
        with _bge_m3_encode_lock:
            embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        # # 把模型输出的 CSR 稀疏矩阵 ，按“每条文本一行”拆成 {特征索引: 权重} 字典
        # # - indices ：非零元素的“列号（特征ID）”
        # # - data ：对应列号的权重值
        # # - indptr ：每一行在 indices/data 里的起止位置指针
        # # 数据示例:
        # # indices = [3, 8, 20, 1, 9]
        # # data    = [0.7, 0.2, 0.1, 0.6, 0.4]  -> milvus -> 稠密向量 [1024] 稀疏向量 : {index:值,index:值}
        # # indptr  = [0, 3, 5]
        # # 获取对应的数据
        # # - 第0条文本用 0:3 => indices=[3,8,20] , data=[0.7,0.2,0.1]
        # # - 第1条文本用 3:5 => indices=[1,9] , data=[0.6,0.4]
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            # embeddings["dense"] = [[1稠密向量],[2稠密向量],[...]  -> 1024]
            # embeddings["sparse"] = [[1稀疏向量],[2稠密向量],[...]  -> 1024]
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


