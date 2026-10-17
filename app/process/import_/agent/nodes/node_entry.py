from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_state

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """入口节点"""
    add_running_task(state["task_id"], "node_entry")
    state = resolve_state(state)
    add_done_task(state["task_id"], state)
    return state

