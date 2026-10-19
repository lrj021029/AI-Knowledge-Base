"""
工具模块，负责提供 task 相关的辅助能力。
"""
from typing import Any, Dict, List, Optional
from .sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID，字符串）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_audiobook_meta": "听书元数据抽取",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点 ---
    "node_item_name_confirm": "确认问题产品",
    "node_intent_recognize": "意图识别",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}

# 从 dict 结构中提取节点名时，依次尝试这些字段
_NODE_NAME_KEYS = ("node_name", "name", "node", "node_id", "id")


def _extract_node_name(raw: Any) -> Optional[str]:
    """
    从任意入参中提取“节点名字符串”。

    - str / int 等标量：直接 str() 返回
    - dict：按 _NODE_NAME_KEYS 顺序查找第一个非空字段
    - 其他：返回 None（调用方需忽略）
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in _NODE_NAME_KEYS:
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
            # 某些写入方可能嵌套了一层，如 {"node": {"node_name": "..."}}
            if isinstance(value, dict):
                nested = _extract_node_name(value)
                if nested:
                    return nested
        return None
    # int / float / bool 等
    if isinstance(raw, (int, float, bool)):
        return str(raw)
    return None


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: Any) -> str:
    """
    将节点名转换为中文展示名；若无映射则返回原名。

    兼容 node_name 为 dict 的场景（从常见字段中提取字符串）。
    """
    name = _extract_node_name(node_name)
    if not name:
        # 无法识别时返回空串，避免污染前端展示
        return ""
    return _NODE_NAME_TO_CN.get(name, name)


def add_running_task(task_id: str, node_name: Any, is_stream: bool = False) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)，可以是字符串或包含节点名的 dict
    """
    name = _extract_node_name(node_name)
    if not name:
        return

    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if name not in running:
        running.append(name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: Any, is_stream: bool = False) -> None:
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)，可以是字符串或包含节点名的 dict
    """
    name = _extract_node_name(node_name)
    if not name:
        return

    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if name not in done:
        done.append(name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer / error）。"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段（如 answer / error）。"""
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """获取当前任务状态。"""
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表（中文展示）。"""
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    # 兜底：过滤 None/空串，避免历史上被污染的数据导致报错
    return [cn for cn in (_to_cn(n) for n in done) if cn]


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行节点列表（中文展示）。"""
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [cn for cn in (_to_cn(n) for n in running) if cn]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """更新任务状态。"""
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)