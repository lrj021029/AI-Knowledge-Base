import sys
import time
import signal

from app.shared.runtime.logger import node_log
from app.rag.query.rrf_service import fuse_by_rrf
from app.shared.utils.task_utils import add_done_task, add_running_task


def timeout_handler(signum, frame):
    raise TimeoutError(">>> 执行超时，进程卡死！")


def run_with_timeout(func, *args, seconds=30, **kwargs):
    """给任意函数加上超时保护，超时直接抛异常"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutError as e:
        print(e)
        raise


@node_log("node_rrf")
def node_rrf(state):
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = fuse_by_rrf(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("=" * 60)
    print(">>> 排查开始")

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
    }

    # ---- 1. 检查 Milvus 集合是否存在 ----
    print("\n>>> [1] 检查 Milvus 集合")
    try:
        from app.shared.clients.milvus_utils import get_milvus_client
        client = get_milvus_client()
        collections = client.list_collections()
        print(f">>> 当前 Milvus 集合列表: {collections}")
    except Exception as e:
        print(f">>> Milvus 检查失败: {e}")

    # ---- 2. 测试向量检索节点 ----
    print("\n>>> [2] 执行 node_search_embedding")
    from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
    try:
        emb_res = run_with_timeout(node_search_embedding, mock_state, seconds=60)
        print(">>> node_search_embedding 完成")
        print(f">>> embedding_chunks 数量: {len(emb_res.get('embedding_chunks') or [])}")
        mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    except Exception as e:
        print(f">>> node_search_embedding 出错/超时: {e}")
        sys.exit(1)

    # ---- 3. 测试 HyDE 检索节点 ----
    print("\n>>> [3] 执行 node_search_embedding_hyde")
    from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
    try:
        hyde_res = run_with_timeout(node_search_embedding_hyde, mock_state, seconds=60)
        print(">>> node_search_embedding_hyde 完成")
        print(f">>> hyde_embedding_chunks 数量: {len(hyde_res.get('hyde_embedding_chunks') or [])}")
        mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []
    except Exception as e:
        print(f">>> node_search_embedding_hyde 出错/超时: {e}")
        sys.exit(1)

    # ---- 4. 测试 RRF 融合节点 ----
    print("\n>>> [4] 执行 node_rrf")
    try:
        result = run_with_timeout(node_rrf, mock_state, seconds=60)
        print(">>> node_rrf 完成")
        print(">>> 最终结果:")
        print(result)
    except Exception as e:
        print(f">>> node_rrf 出错/超时: {e}")
        sys.exit(1)

    print("\n>>> 全部执行完毕")