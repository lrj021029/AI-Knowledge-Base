"""
答案生成服务（查询侧）。

扩展点（听书业务）：
- 依据意图（general_qa / book_recommend / book_detail / note_search）选择不同的业务提示词；
- 所有业务答案统一追加“参考来源”（书名、作者、内容类型、来源文件名），保证可溯源；
- 复用原有流式/非流式输出能力；
- 交付结构化结果（structured_result）供后续接口/前端使用。

同时修复了原有实现中的若干缺陷：
- 图片链接写入了不存在的 state 键（img_urls -> image_urls）；
- 历史会话拼接下标越界/遍历字符串；
- 无召回时的空答案处理。
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import (
    INTENT_BOOK_DETAIL,
    INTENT_BOOK_RECOMMEND,
    INTENT_GENERAL_QA,
    INTENT_NOTE_SEARCH,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.sse_utils import SSEEvent
from app.shared.utils.task_utils import push_to_session
import re

# 意图 -> 业务提示词
_INTENT_PROMPT_MAP: dict[str, str] = {
    INTENT_BOOK_RECOMMEND: "audiobook_book_recommend",
    INTENT_BOOK_DETAIL: "audiobook_book_detail",
    INTENT_NOTE_SEARCH: "audiobook_note_answer",
    INTENT_GENERAL_QA: "answer_out",
}


def state_has_item_names(state: QueryGraphState) -> tuple[bool]:
    """判断是否识别到主体"""
    item_names: list[str] = state.get("item_names", [])
    has_item_names: bool = True if item_names else False
    return has_item_names


def _load_history_text(state: QueryGraphState) -> str:
    """读取并拼接最近历史会话（修复原有遍历字符串/下标越界缺陷）。"""
    session_id: str = state.get("session_id")
    recent_message_list: list[dict] = history_repository.list_recent(session_id=session_id, limit=20)
    valid_message_list: list[dict] = [item for item in recent_message_list if item.get("item_names")]
    if not valid_message_list:
        return "当前会话无历史聊天记录"

    lines: list[str] = []
    for index, message in enumerate(reversed(valid_message_list), start=1):
        role = "用户" if message.get("role") == "user" else "助手"
        lines.append(
            f"第{index}轮[{role}] 原始内容:{message.get('text')},"
            f"重写后的问题:{message.get('rewritten_query')},"
            f"关联主体:{message.get('item_names')}"
        )
    return "\n".join(lines)


def _build_context(reranked_docs: list[dict]) -> str:
    """构造参考上下文，附带听书溯源元数据。"""
    context: str = ""
    for chunk in reranked_docs:
        if chunk.get("type") == "web":
            context += (
                f"1.来源: 联网检索  2.标题: {chunk.get('title')}  3.链接: {chunk.get('url')}\n"
                f"4.内容: {chunk.get('content')}\n\n"
            )
        else:
            context += (
                f"1.来源: 本地向量库"
                f"  2.书名: {chunk.get('book_name') or chunk.get('item_name')}"
                f"  3.作者: {chunk.get('author_name') or '未知'}"
                f"  4.有声时长: {chunk.get('audio_duration') or '未知'}"
                f"  5.演播人: {chunk.get('narrator') or '未知'}"
                f"  6.内容类型: {chunk.get('content_type')}"
                f"  7.来源文件名: {chunk.get('source_file_name') or chunk.get('file_title')}"
                f"  8.内容标题: {chunk.get('title')}"
                f"  9.内容置信度: {chunk.get('score')}"
                f"  10.内容: {chunk.get('content')}\n\n"
            )
    return context


def build_citations(reranked_docs: list[dict]) -> list[dict]:
    """从召回结果中抽取溯源信息（书名/作者/内容类型/来源文件名/链接）。"""
    citations: list[dict] = []
    seen: set[tuple] = set()
    for chunk in reranked_docs:
        if chunk.get("type") == "web":
            key = ("web", chunk.get("url"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "type": "web",
                "title": chunk.get("title"),
                "url": chunk.get("url"),
            })
        else:
            book_name = chunk.get("book_name") or chunk.get("item_name") or ""
            source_file_name = chunk.get("source_file_name") or chunk.get("file_title") or ""
            key = ("local", book_name, chunk.get("author_name"), chunk.get("content_type"), source_file_name)
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "type": "local",
                "book_name": book_name,
                "author_name": chunk.get("author_name") or "",
                "content_type": chunk.get("content_type") or "",
                "source_file_name": source_file_name,
                "title": chunk.get("title") or "",
            })
    return citations


def _format_sources(citations: list[dict]) -> str:
    """将溯源信息渲染为答案末尾的“参考来源”区块。"""
    if not citations:
        return ""
    lines: list[str] = ["", "---", "**参考来源**"]
    for index, item in enumerate(citations, start=1):
        if item.get("type") == "web":
            lines.append(f"{index}. {item.get('title') or '联网资料'}（{item.get('url')}）")
        else:
            parts = [f"《{item.get('book_name')}》"] if item.get("book_name") else []
            if item.get("author_name"):
                parts.append(item["author_name"])
            if item.get("content_type"):
                parts.append(item["content_type"])
            if item.get("source_file_name"):
                parts.append(f"来源文件:{item['source_file_name']}")
            lines.append(f"{index}. " + " ｜ ".join(parts))
    return "\n".join(lines)


def _load_answer_prompt(state: QueryGraphState, context: str) -> str:
    """按意图加载对应提示词。"""
    intent: str = state.get("intent") or INTENT_GENERAL_QA
    prompt_name: str = _INTENT_PROMPT_MAP.get(intent, "answer_out")
    question: str = state.get("rewritten_query") or state.get("original_query")
    book_name: str = state.get("book_name") or ""
    history: str = _load_history_text(state)

    if prompt_name == "answer_out":
        item_names: list[str] = state.get("item_names")
        return load_prompt("answer_out", context=context, history=history, item_names=item_names, question=question)

    if prompt_name == "audiobook_book_recommend":
        return load_prompt(
            "audiobook_book_recommend",
            question=question,
            category=state.get("category") or "不限",
            scene_tags="、".join(state.get("scene_tags") or []) or "不限",
            content_types="、".join(state.get("content_types") or []) or "不限",
            context=context,
        )

    if prompt_name == "audiobook_book_detail":
        return load_prompt(
            "audiobook_book_detail",
            question=question,
            book_name=book_name or "未指定",
            context=context,
        )

    return load_prompt(
        "audiobook_note_answer",
        question=question,
        book_name=book_name or "未指定",
        context=context,
    )


def call_model_answer_out(state: QueryGraphState, context: str):
    """调用大模型生成答案，支持流式/非流式。"""
    llm_client = llm_provider.chat()
    answer_out_prompt: str = _load_answer_prompt(state, context)
    messages = [HumanMessage(content=answer_out_prompt)]
    is_stream: bool = state.get("is_stream")
    chains = llm_client | StrOutputParser()
    answer: str = ""

    if is_stream:
        stream = chains.stream(messages)
        for content in stream:
            push_to_session(session_id=state.get("session_id"), event=SSEEvent.DELTA, data={"delta": content})
            answer += content
    else:
        answer = chains.invoke(messages)
    state["answer"] = answer


def extract_img_urls_by_reranked_docs(state: QueryGraphState) -> list[str]:
    img_urls: list[str] = []
    img_rep = re.compile(r"\!\[.*?\]\((.*?)\)")
    reranked_docs: list[dict] = state.get("reranked_docs", [])
    # 循环获取每个 chunk
    for chunk in reranked_docs:
        type_str: str = chunk.get("type")
        if type_str == 'web':
            url: str = chunk.get("url") or ""
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                img_urls.append(url)
        else:
            content: str = chunk.get("content") or ""
            content_img_urls: list[str] = img_rep.findall(content)
            if content_img_urls:
                img_urls.extend(content_img_urls)
    # 去重保序
    deduped: list[str] = []
    for url in img_urls:
        if url not in deduped:
            deduped.append(url)
    # 修复原有 bug：state 字段名应为 image_urls
    state["image_urls"] = deduped
    return deduped


def build_structured_result(state: QueryGraphState, citations: list[dict]) -> dict:
    """构造结构化业务结果（书籍推荐/详情），供接口与前端复用。"""
    intent: str = state.get("intent") or INTENT_GENERAL_QA
    reranked_docs: list[dict] = state.get("reranked_docs", [])
    if intent == INTENT_BOOK_RECOMMEND:
        books: dict[str, dict] = {}
        for chunk in reranked_docs:
            book_name = chunk.get("book_name") or chunk.get("item_name")
            if not book_name or book_name in books:
                continue
            books[book_name] = {
                "book_name": book_name,
                "author_name": chunk.get("author_name") or "",
                "category": chunk.get("category") or "",
                "scene_tags": chunk.get("scene_tags") or "",
                "audio_duration": chunk.get("audio_duration") or "",
                "narrator": chunk.get("narrator") or "",
                "highlights": chunk.get("highlights") or "",
                "summary": chunk.get("summary") or "",
            }
        return {"books": list(books.values())}
    if intent == INTENT_BOOK_DETAIL:
        book = reranked_docs[0] if reranked_docs else {}
        return {
            "book_name": state.get("book_name") or book.get("book_name") or "",
            "author_name": book.get("author_name") or "",
            "audio_duration": book.get("audio_duration") or "",
            "narrator": book.get("narrator") or "",
            "category": book.get("category") or "",
            "scene_tags": book.get("scene_tags") or "",
            "highlights": book.get("highlights") or "",
            "summary": book.get("summary") or "",
        }
    return {}


def save_assistant_chat_message(state: QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role='assistant',
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )


def _fallback_no_result_answer(state: QueryGraphState) -> str:
    """无召回时的兜底答案（不调用大模型，避免幻觉）。"""
    intent: str = state.get("intent") or INTENT_GENERAL_QA
    query: str = state.get("original_query") or state.get("rewritten_query") or ""
    if intent == INTENT_BOOK_RECOMMEND:
        return f"知识库中暂未检索到与「{query}」匹配的听书资料，请确认相关书籍资料已导入，或调整筛选条件后重试。"
    if intent == INTENT_BOOK_DETAIL:
        book_name = state.get("book_name") or query
        return f"知识库中暂无《{book_name}》的详情资料，请确认该书资料已导入。"
    return f"知识库中暂未检索到与「{query}」相关的内容，请补充相关资料或换个问法。"


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    生成最终回答：
    - 有召回：按意图调用业务提示词生成答案，并追加参考来源；
    - 无召回但已有预置答案（如未识别到主体）：保留预置答案；
    - 无召回且无预置答案：输出兜底提示，避免大模型幻觉。
    """
    reranked_docs: list[dict] = state.get("reranked_docs", [])
    citations: list[dict] = build_citations(reranked_docs)
    state["citations"] = citations
    state["structured_result"] = build_structured_result(state, citations)

    if reranked_docs:
        context: str = _build_context(reranked_docs)
        call_model_answer_out(state, context)
        # 统一追加溯源信息
        sources: str = _format_sources(citations)
        if sources:
            if state.get("is_stream"):
                push_to_session(state.get("session_id"), SSEEvent.DELTA, {"delta": sources})
            state["answer"] = (state.get("answer") or "") + sources
        extract_img_urls_by_reranked_docs(state)
    else:
        # 无召回：保留已预置的答案（例如未识别到主体时的提示）
        if not (state.get("answer") or "").strip():
            state["answer"] = _fallback_no_result_answer(state)
        state["image_urls"] = state.get("image_urls", [])

    save_assistant_chat_message(state)
    return {
        "session_id": state.get("session_id"),
        "answer": state.get("answer"),
        "image_urls": state.get("image_urls"),
        "is_stream": state.get("is_stream"),
        "citations": state.get("citations", []),
        "structured_result": state.get("structured_result", {}),
        "intent": state.get("intent", ""),
    }
