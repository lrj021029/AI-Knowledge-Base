import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, DataType

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_RECOGNITION_CHUNKS_TOP_K
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import  load_prompt
from app.infra.vectorstore.milvus_gateway import  milvus_gateway


@step_log("validate_and_get_data")
def validate_and_get_data(state:ImportGraphState) -> tuple[str,list[dict[str,Any]]]:
    # 1.1 获取相关参数 file_title / chunks
    file_title:str             = state.get("file_title")
    chunks:list[dict[str,Any]] = state.get("chunks")
    md_path:str                = state.get("md_path")
    # 1.2 非空判断
    if not chunks:
        if md_path and Path(md_path).is_file():
            # 字符串存在,且文件真实存在
            json_file_obj:Path = Path(md_path).parent / "chunks.json"
            # 备份
            # 简单
            # json_file_obj.write_text(data=json.dumps(chunks),encoding="utf-8")
            #
            # with json_file_obj.open("w",encoding="utf-8") as f:
            #     json.dump(chunks,f,ensure_ascii=False,indent=4)
            # 还原
            # 简单
            chunks = json.loads(json_file_obj.read_text(encoding="utf-8"))
            # with json_file_obj.open("r",encoding="utf-8") as f:
            #    chunks  =  json.load(f)
        if not chunks:
            logger.error(f"chunks没有内容,读取备份文件也没有读取到,业务无法继续,提前终止!!")
            raise ValueError(f"chunks没有内容,读取备份文件也没有读取到,业务无法继续,提前终止!!")
        state["chunks"] = chunks
    if not file_title:
        file_title = "default_title"
    #1.3 返回结果
    return file_title,chunks


# 部分大模型会把提示词里的“返回空字符串”理解成输出字面量（如“空字符串”“无”“NONE”），
# 这里统一按“未识别到”处理，避免把占位文字当成真实主体名称写入向量库。
_NULL_ITEM_NAME_TOKENS:set[str] = {
    "", "空字符串", "空", "无", "暂无", "未知", "无法识别", "没有", "无内容",
    "none", "null", "n/a", "na", "-", "—",
}


def normalize_item_name(raw:str) -> str:
    """归一化模型输出的主体名称：去除包裹标点；空值语义统一返回空字符串。"""
    text:str = (raw or "").strip().strip("。.、,，:：;；\"'“”‘’`【】[]()（） \n\t")
    if text.lower() in _NULL_ITEM_NAME_TOKENS:
        return ""
    return text


@step_log("call_llm_item_name_recognition")
def call_llm_item_name_recognition(chunks:list[dict[str,Any]],file_title:str,fallback_item_name:str="") -> str:
    #  2.1 加载模型的客户端对象(llm/providers)
    llm_client = llm_provider.chat(json_model=False)
    # 2.2 加载提示词和封装提示词Message对象
    # 2.2.1 凑齐提示词需要的动态字符串 file_title / context = 没有 -> chunks拼接的内容
    # context -> 模型 -> item_name ->  parent_title part content   parent_title part content
    context:str = ""
    for chunk in chunks[:ITEM_NAME_RECOGNITION_CHUNKS_TOP_K]:
        context += f"父标题:{chunk.get('parent_title')},第{chunk.get('part')}部分,内容为: {chunk.get('content')}\n"
    recognition_prompt:str = load_prompt("item_name_recognition",file_title = file_title,context = context)
    message = HumanMessage(
        content = recognition_prompt
    )
    # 2.3 定义调用链chain对象  = 模型 | 输出结果处理器对象
    chains = llm_client | StrOutputParser()
    # 2.4 执行调用链获取结果
    raw_item_name:str = chains.invoke([message])
    # 2.5 归一化：模型常把提示词里的“返回空字符串”理解为输出字面量（“空字符串”/“无”/“NONE”），
    #     需要按“未识别到”处理；无法识别时优先用书籍主体名兜底，其次用文件名兜底。
    item_name:str = normalize_item_name(raw_item_name)
    if not item_name:
        fallback:str = (fallback_item_name or "").strip() or file_title
        logger.warning(f"未识别到有效item_name(模型原始输出:{raw_item_name!r}),兜底使用:{fallback}")
        item_name = fallback
    # 2.6 返回结果
    return item_name


@step_log("item_name_embedding")
def item_name_embedding(item_name:str)->tuple[list,dict]:
    # 3.1 调用infra生成向量的方法进行item_name向量生成
    # todo: 注意需要的列表
    result = llm_provider.generate_infra_embeddings([item_name])
    # 3.2 解析出稠密和稀疏向量
    # result => {"dense":[[1024]],"sparse":[{}]}
    dense_vector  = result['dense'][0]
    sparse_vector = result['sparse'][0]
    # 3.3 返回结果
    return dense_vector,sparse_vector


@step_log("prepared_item_name_collection")
def  prepared_item_name_collection():
    # 4.1 检查集合是否存在,存在不需要创建直接返回
    milvus_client:MilvusClient =  milvus_gateway.client
    has:bool =  milvus_client.has_collection(collection_name=milvus_gateway.item_name_collection)
    if has:
        logger.info(f"{milvus_gateway.item_name_collection}已经存在,无需创建,直接返回!")
        return
    # 4.2 创建集合对应schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    # schema添加字段  item_name -> pk file_title item_name dense_vector sparse_vector
    # datatype= pymilvus_DataType.类型
    # schema.add_field(field_name="age", datatype=DataType.INT64, nullable=True, default_value=18)
    schema.add_field(field_name="pk",datatype=DataType.INT64,nullable=False,is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=600)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=600)
    # 稠密向量 -> [] -> 元素类型 -> FP32  | FP16 (cpu/gpu)  || 使用dim 参数指定密集向量的维度。 (嵌入式模型)
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4.3 创建集合对应index
    index_params = milvus_client.prepare_index_params()

    # 向量场
    # 稠密
    index_params.add_index(
        field_name="dense_vector",  # 给哪个字段的字段名字
        index_type="HNSW",          # 索引类型  FLAT 蛮力全局检索,召回率100% 慢 | 准
                                    #         IVF_FLAT  将数据分类,每类有一个中心点,先找中心点所在类,类中进行检索 快 / 精度少差
                                    #         HNSW  逐层向下精准定位 效率高于FLAT 精度高于IVF_FLAT
        index_name="dense_vector_index",  # 索引名字 [可选]
        metric_type="COSINE",             # 向量场相似度计算方式  稠密: [cosine] ip l2 稀疏: [ip] / bm25
        index_params={
            # 索引的额外参数
            "M": 64,  #  每个点最大连接的邻居的数量
            "efConstruction": 100  # 从多少个点中选出邻居的数量
        }
    )

    #稀疏
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 稀疏倒排索引  从 数据->向量 倒过来 向量坐标关联数据 每次查询时候,先定位坐标关联的数据集合! 在进行集合内检索!
        index_name="sparse_vector_index",
        metric_type="IP",     # IP / BM25(还不推荐)
        index_params={
            "inverted_index_algo": "SINDI"  # IP提高检索效率的方式  DAAT_MAXSCORE
        }

    )

    # 标量场 不讲了
    # 4.4 创建集合
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_name_collection,
        schema=schema,
        index_params=index_params
    )


    # 4.5 加载集合 从磁盘 加载到内存
    milvus_client.load_collection(
        collection_name=milvus_gateway.item_name_collection
    )


@step_log("insert_item_name_collection")
def insert_item_name_collection(item_name, dense_vector, sparse_vector, file_title):
    milvus_client: MilvusClient = milvus_gateway.client
    coll_name = milvus_gateway.item_name_collection

    # ========== 先查询主键，再按主键删除 ==========
    query_res = milvus_client.query(
        collection_name=coll_name,
        filter=f"file_title == '{file_title}'",
        output_fields=["pk"],          # ← 改成 pk（主键）
        consistency_level="Eventually"
    )
    del_ids = [row["pk"] for row in query_res]   # ← 改成 pk
    delete_count = 0
    if del_ids:
        result = milvus_client.delete(
            collection_name=coll_name,
            ids=del_ids
        )
        delete_count = result.get("delete_count", 0)
    logger.info(f"已经完成数据删除,删除数据条数为:{delete_count}")

    # 5.2 插入新的item_name和file_title
    data: dict = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    insert_result = milvus_client.insert(
        collection_name=coll_name,
        data=[data]
    )
    logger.info(f"完成{item_name}数据插入,影响条数为:{insert_result.get('insert_count')}")

@step_log("improve_chunks_item_name")
def improve_chunks_item_name(chunks,item_name):
    for chunk in chunks:
        chunk['item_name'] = item_name

@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取数据和校验
    file_title,chunks = validate_and_get_data(state)
    # 1.1 听书场景：优先用元数据抽取出的书名作为主体名称兜底（item_name 语义即“主体名称/书名”）
    audiobook_meta:dict[str,Any] = state.get("audiobook_meta") or {}
    book_name:str = (audiobook_meta.get("book_name") or "").strip()
    # 2. 调用模型识别item_name
    item_name:str = call_llm_item_name_recognition(chunks,file_title,fallback_item_name=book_name)
    # 3. item_name进行向量化
    dense_vector,sparse_vector = item_name_embedding(item_name)
    # 4. 准备存储item_name的集合
    prepared_item_name_collection()
    # 5. 数据删除和插入
    insert_item_name_collection(item_name,dense_vector,sparse_vector,file_title)
    # 6. 完善chunks数据
    improve_chunks_item_name(chunks,item_name)
    # 7. 更新state item_name chunks
    state['item_name'] = item_name
    state['chunks']    = chunks
    # 8. 返回结果
    return state