from pathlib import Path
from dashscope import HistoryItem
from fastapi import FastAPI,BackgroundTasks
from fastapi.responses import FileResponse,EventSourceResponse
from fastapi.requests import Request
from mimetypes import guess_type
from app.api.schema.query_schema import QueryAsyncResponse, QueryResponse, QueryRequest,HistoryDeleteResponse,HistoryItemResponse,HistoryResponse
from app.process.query.agent.main_graph import query_graph_app
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.task_utils import update_task_status,TASK_STATUS_COMPLETED, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED,clear_task,get_done_task_list
from app.shared.utils.sse_utils import sse_generator,push_to_session,SSEEvent,create_sse_queue

import ast

def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [value]

app = FastAPI(description='查询模块的接口')

@app.get("/html")
def return_chat_html():
    html_path_obj:Path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj)[0],
    )

@app.get("/health")
def health():
    return {"code":0}

@app.get("/stream/{session_id}")
def stream(session_id:str,request:Request):
    return  EventSourceResponse(sse_generator(session_id,request))


def invoke_query_graph(session_id:str,original_query:str,is_stream:bool):
    state:QueryGraphState = create_query_default_state(
        session_id=session_id,
        original_query=original_query,
        is_stream=is_stream
    )
    try:
        update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)
        result = query_graph_app.invoke(state)
        update_task_status(session_id,TASK_STATUS_COMPLETED,is_stream)

        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": result.get("answer"),
                    "status": result.get("status"),
                    "image_urls": result.get("image_urls"),
                }
            )
        return result
    except Exception as e:
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)
        logger.exception(f"查询异常:{str(e)}")
        if is_stream:
            push_to_session(session_id,SSEEvent.ERROR,{"error": f"查询信息异常:{str(e)}"})


@app.post("/query")
def query(tasks:BackgroundTasks,params:QueryRequest):
    session_id:str = params.session_id
    original_query = params.query
    is_stream = params.is_stream
    # todo
    clear_task(session_id)
    # 判断 stream 是否异步
    if is_stream:
        create_sse_queue(session_id)
        tasks.add_task(invoke_query_graph,session_id=session_id,original_query=original_query,is_stream=is_stream)
        return QueryAsyncResponse(
            message=f"{session_id}任务查询中..",
            session_id=session_id,
        )
    else:
        result:QueryGraphState = invoke_query_graph(session_id=session_id,original_query=original_query,is_stream=is_stream)

        answer:str = result.get("answer")
        image_urls:list[str] = result.get("image_urls",[])
        done_list:list[str] = get_done_task_list(session_id)

        return QueryResponse(
            message=f"{session_id}处理完毕",
            session_id=session_id,
            answer=answer,
            done_list=done_list,
            image_urls=image_urls
        )

    return QueryAsyncResponse / QueryResponse

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    deleted_count = history_repository.clear_session(session_id=session_id)
    logger.info(f"共删除了:{deleted_count}条于{session_id}历史记录中")
    return HistoryDeleteResponse(
        message=f"共删除了:{deleted_count}条于{session_id}历史记录中",
        deleted_count=deleted_count,
    )


@app.get("/history/{session_id}")
def list_history(session_id: str):
    history_message: list[dict] = history_repository.list_recent(session_id=session_id, limit=10)
    item_list: list[HistoryItemResponse] = [
        HistoryItemResponse(
            session_id=session_id,
            id=str(message_dict.get("_id")),
            role=message_dict.get("role"),
            text=message_dict.get("text"),
            rewritten_query=message_dict.get("rewritten_query"),
            item_names=_ensure_list(message_dict.get("item_names")),
            image_urls=_ensure_list(message_dict.get("image_urls")),
            ts=message_dict.get("ts"),
        )
        for message_dict in history_message
    ]
    logger.info(f"查询的{session_id},对应聊天记录共{len(item_list)}条")
    return HistoryResponse(
        session_id=session_id,
        items=item_list,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
