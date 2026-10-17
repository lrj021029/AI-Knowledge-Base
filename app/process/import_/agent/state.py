from typing import TypedDict
import copy

class ImportGraphState(TypedDict):
    """图状态"""
    task_id:str
    is_md_read_enabled:bool
    is_pdf_read_enabled:bool
    local_dir:str
    local_file_path:str
    file_title:str
    pdf_path:str
    md_path:str
    md_content:str
    chunks:list
    item_name:str
    embedding_content:str

graph_default_state:ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "file_title": "",
    "pdf_path": "",
    "md_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embedding_content": "",
}

def create_default_state(**kwargs) -> ImportGraphState:
    """创建默认状态"""
    state = copy.deepcopy(graph_default_state)
    state.update(kwargs)
    return state

def get_default_state(**kwargs) -> ImportGraphState:
    """获取默认状态"""
    return copy.deepcopy(graph_default_state)