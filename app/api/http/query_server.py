from pathlib import Path
from dashscope import HistoryItem
from fastapi import FastAPI,BackgroundTasks
from fastapi.responses import FileResponse,EventSourceResponse
from fastapi.requests import Request
from mimetypes import guess_type
from app.api.schema.query_schema import QueryAsyncResponse, QueryResponse, QueryRequest,HistoryDeleteResponse,HistoryItemResponse,HistoryResponse,SessionItemResponse,SessionListResponse
from app.api.schema.book_schema import (
    BookRecommendRequest,
    BookRecommendResponse,
    BookDetailRequest,
    BookDetailResponse,
)
from app.process.query.agent.main_graph import query_graph_app
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.rag.query.book_recommend_service import recommend_books
from app.rag.query.book_detail_service import get_book_detail
from app.rag.query.config import BOOK_RECOMMEND_DEFAULT_LIMIT
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.task_utils import update_task_status,TASK_STATUS_COMPLETED, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED,clear_task,get_done_task_list
from app.shared.utils.sse_utils import sse_generator,push_to_session,SSEEvent,create_sse_queue

import ast
import threading

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


def _prewarm_recommend_cache() -> None:
    """
    启动时异步预热精品推荐默认页面。
。
    """
    try:
        result = recommend_books(limit=BOOK_RECOMMEND_DEFAULT_LIMIT, offset=0)
        logger.info(f"精品推荐缓存预热完成,书籍数:{result.get('total')}")
    except Exception as e:
        logger.warning(f"精品推荐缓存预热失败(不影响服务):{e}")


@app.on_event("startup")
def _startup_prewarm() -> None:
    threading.Thread(target=_prewarm_recommend_cache, name="prewarm-recommend", daemon=True).start()

# 允许前端（含精品推荐页面被单独打开/file:// 场景）跨域访问业务接口
from fastapi.middleware.cors import CORSMiddleware
from app.shared.config.settings_config import settings as _settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins) or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/html")
def return_chat_html():
    html_path_obj:Path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj)[0],
    )


@app.get("/html/book-recommend")
def return_book_recommend_html():
    """精品推荐页面（听书首页核心模块）。"""
    html_path_obj:Path = PROJECT_ROOT / "app" / "resources" / "html" / "book-recommend.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj)[0],
    )


@app.post("/book/recommend", response_model=BookRecommendResponse)
def book_recommend(params: BookRecommendRequest):
    """
    精品推荐查询：传入筛选标签（类别/场景），返回书籍推荐列表。

    - 元数据过滤基于 Milvus（向量库）已入库文档的听书元数据字段；
    - 推荐理由 LLM 生成，失败时规则兜底。
    """
    result = recommend_books(
        book_name=params.book_name,
        author_name=params.author_name,
        categories=params.categories,
        scene_tags=params.scene_tags,
        content_types=params.content_types,
        query=params.query,
        limit=params.limit,
        offset=params.offset,
        with_reasons=params.with_reasons,
    )
    return BookRecommendResponse(
        code=200,
        total=result.get("total", 0),
        offset=result.get("offset", 0),
        limit=result.get("limit", params.limit),
        filters=result.get("filters", {}),
        books=result.get("books", []),
    )


@app.post("/book/detail", response_model=BookDetailResponse)
def book_detail(params: BookDetailRequest):
    """书籍详情查询：传入书名，返回结构化书籍详情。"""
    detail = get_book_detail(params.book_name)
    return BookDetailResponse(
        code=200,
        found=bool(detail.get("found")),
        book_name=detail.get("book_name", params.book_name),
        data=detail,
        message=detail.get("message", ""),
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
                    "citations": result.get("citations", []),
                    "intent": result.get("intent", ""),
                    "structured_result": result.get("structured_result", {}),
                }
            )
        return result
    except Exception as e:
        update_task_status(session_id,TASK_STATUS_FAILED,is_stream)
        logger.exception(f"查询异常:{str(e)}")
        # 修复原有缺陷：异常时返回安全兜底结构，避免调用方 result.get 触发 NoneType 崩溃
        fallback_answer: str = f"抱歉，本次查询处理失败：{str(e)}。请稍后重试。"
        state["answer"] = fallback_answer
        state["image_urls"] = state.get("image_urls", []) or []
        if is_stream:
            push_to_session(session_id,SSEEvent.ERROR,{"error": f"查询信息异常:{str(e)}"})
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": fallback_answer,
                    "status": "failed",
                    "image_urls": [],
                    "citations": [],
                    "intent": state.get("intent", ""),
                    "structured_result": {},
                }
            )
        return state


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
            image_urls=image_urls,
            citations=result.get("citations", []) or [],
            intent=result.get("intent", "") or "",
            structured_result=result.get("structured_result", {}) or {},
        )

@app.get("/history/sessions")
def list_history_sessions(limit: int = 30):
    """
    最近的历史会话清单（按最后一条消息时间倒序）。

    必须声明在 `/history/{session_id}` 之前，否则 "sessions" 会被当成一个 session_id 吃掉。
    """
    session_list: list[dict] = history_repository.list_sessions(limit=limit)
    logger.info(f"查询历史会话清单,共{len(session_list)}个")
    return SessionListResponse(
        sessions=[
            SessionItemResponse(
                session_id=str(item.get("session_id") or ""),
                title=str(item.get("title") or ""),
                count=int(item.get("count") or 0),
                last_ts=float(item.get("last_ts") or 0),
            )
            for item in session_list
        ]
    )


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
    # 单次最多回放 50 条（此前写死 10 条，长会话只能看到最后几轮）
    history_message: list[dict] = history_repository.list_recent(session_id=session_id, limit=50)
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
