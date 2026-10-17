from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.infra.persistence.history_repository import history_repository
import re


def state_has_item_names(state: QueryGraphState) -> tuple[bool]:
    """判断是否识别到主体"""
    item_names:list[str] = state.get("item_names",[])
    has_item_names:bool = True if item_names else False
    return has_item_names


def _load_answer_ouput_prompt(state: QueryGraphState) -> str:
    """加载提示词文本"""
    question:str = state.get("rewritten_query")
    item_names:list[str] = state.get("item_names")
    session_id:str = state.get("session_id")
    recent_message_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=20)
    # 过滤有效聊天记录
    valid_message_list:list[dict] = [
        item for item in recent_message_list if item.get('item_names')
    ]
    history:str = ""
    if valid_message_list:
        for index in range(1,len(history),2):
            message_assistant:dict = valid_message_list[index]
            message_user:dict = valid_message_list[index+1]
            history += (
                f"第{index/2+1}轮对话:"
                f"[本轮提问] 原始内容:{message_user.get('text')},重写后的问题:{message_user.get('rewritten_query')},本次提问关联的主体:{message_user.get('item_names')}\n"
                f"[本轮回答] 原始内容:{ message_assistant.get('text')},重写后的问题:{ message_assistant.get('rewritten_query')},本次提问关联的主体:{ message_assistant.get('item_names')}\n\n"
            )
    else:
        history = "当前会话无历史聊天记录"
    context:str = ""
    reranked_docs:list[dict] = state.get("reranked_docs",[])

    for chunk in reranked_docs:
        context += f"1.来源: { '联网检索'  if chunk.get('type') == 'web' else '本地向量库'}  2. 内容标题: {chunk.get('title')}  3. 内容置信度: {chunk.get('score')}  4.内容:  {chunk.get('content')} \n\n"

    answer_output_prompt:str = load_prompt("answer_out", context=context,history=history,item_names=item_names,question=question)
    return answer_output_prompt


def call_model_answer_out(state: QueryGraphState):
    llm_client = llm_provider.chat()
    answer_out_prompt:str = _load_answer_ouput_prompt(state)
    messages = [HumanMessage(content=answer_out_prompt)]
    # 判断流式
    is_stream:bool = state.get("is_stream")
    chains = llm_client | StrOutputParser()
    answer:str = ""

    if is_stream:
        stream = chains.stream(messages)
        for content in stream:
            push_to_session(session_id=state.get("session_id"),event=SSEEvent.DELTA,data={"delta":content})
            answer += content
    else:
        answer:str = chains.invoke(messages)
    state["answer"] = answer


def extract_img_urls_by_reranked_docs(state: QueryGraphState) -> list[str]:
    img_urls:list[str] = []
    img_rep = re.compile(r"\!\[.*?\]\((.*?)\)")
    reranked_docs:list[dict] = state.get("reranked_docs",[])
    # 循环获取每个 chunk
    for chunk in reranked_docs:
        type_str:str = chunk.get("type")
        if type_str == 'web':
            url:str = chunk.get("url")
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                img_urls.append(url)
        else:
            content:str = chunk.get("content")
            content_img_urls:list[str] = img_rep.findall(content)
            if content_img_urls:
                img_urls.extend(content_img_urls)
    state["img_urls"] = img_urls


def save_assistant_chat_message(state: QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role='assistant',
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=state.get("image_urls",[]),
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    has_item_names = state.get("has_item_names")
    if has_item_names:
        call_model_answer_out(state)
        extract_img_urls_by_reranked_docs(state)
    save_assistant_chat_message(state)
    return {
        "session_id": state.get("session_id"),
        "answer": state.get("answer"),
        "image_urls": state.get("image_urls"),
        "is_stream": state.get("is_stream"),
    }