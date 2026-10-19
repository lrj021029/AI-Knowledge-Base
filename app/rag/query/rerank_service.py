from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, RERANK_SUMMARY_CHAR_RATIO, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_and_get_data(state: QueryGraphState) -> tuple[str, list[dict], list[dict]]:
    """校验获取参数；允许本地召回或联网结果为空的场景（联网不可用时降级本地）。"""
    rewritten_query: str = state.get('rewritten_query')
    rrf_chunks: list[dict] = state.get('rrf_chunks') or []
    web_search_docs: list[dict] = state.get('web_search_docs') or []

    if not rewritten_query:
        logger.error(f"rewritten_query为空,业务终止")
        raise ValueError(f"rewritten_query为空,业务终止")
    return rewritten_query, rrf_chunks, web_search_docs


def merge_rrf_web_list(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    # 数据合并：联网结果不参与 RRF，仅与本地 RRF 结果一起送入 Reranker
    merge_list: list[dict] = list(rrf_chunks) + list(web_search_docs)
    logger.info(f"数据融合后长度:{len(merge_list)}")
    return merge_list


def create_question_answer_pair_list(rewritten_query: str, merge_list: list[dict]) -> list[list[str]]:
    question_answer_pair: list[list[str]] = []
    question_token_number: int = llm_provider.reranker_str_to_tokens_number(rewritten_query)

    for chunk in merge_list:
        content: str = chunk.get('content') or ""
        answer_token_number: int = llm_provider.reranker_str_to_tokens_number(content)
        if (answer_token_number + question_token_number) > RERANK_MAX_INPUT_TOKENS:
            compass_char_len: int = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - answer_token_number) / RERANK_SUMMARY_CHAR_RATIO),
            )
            reranker_prompt: str = load_prompt(
                "rerank_text_refine",
                question=rewritten_query,
                answer=content,
                limit=compass_char_len,
            )
            messages = [
                HumanMessage(content=reranker_prompt)
            ]
            llm_client = llm_provider.chat()
            chains = llm_client | StrOutputParser()
            content = chains.invoke(messages)
        # 将问题 + content 列表添加到容器
        question_answer_pair.append([rewritten_query, content])
    return question_answer_pair


def reranker_compute_score(question_answer_pair: list[list[str]]) -> list[float]:
    score_list: list[float] = llm_provider.reranker_computer_scores(question_answer_pair)
    logger.info(f"计算得分结果:{score_list}")
    return score_list


def merge_list_add_score(merge_list: list[dict], score_list: list[float]):
    logger.info(f"赋分前数据示例:{merge_list[0]}")
    for score, chunk in zip(score_list, merge_list):
        chunk['score'] = score
    logger.info(f"赋分后数据示例:{merge_list[0]}")


def sort_merge_list(merge_list: list[dict]):
    logger.info(f"排序前数据:{merge_list}")
    # 修复原有 bug：`x['score',0.0]` 是元组键，会 KeyError；应为 x.get('score',0.0)
    merge_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    logger.info(f"排序后数据:{merge_list}")


def dynamic_top_k(merge_list: list[dict]) -> list[dict]:
    # 定义常量
    max_top_k = RERANK_MAX_TOPK
    min_top_k = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS
    # 解决 max_top_k > len 导致 range 循环,index 越界问题
    max_top_k = min(len(merge_list), max_top_k)
    # 定义截取数量
    top_k = max_top_k
    if max_top_k > min_top_k:
        for index in range(min_top_k - 1, max_top_k - 1):
            pre_score: float = merge_list[index].get('score', 0.0)
            next_score: float = merge_list[index + 1].get('score', 0.0)
            abs_score = pre_score - next_score
            ratio = abs_score / (pre_score + 1e-8)

            if abs_score > gap_abs or ratio > gap_ratio:
                top_k = index + 1
                logger.info(f"找到断崖位置,截取数量:{top_k}")
                break
    reranker_docs: list[dict] = merge_list[:top_k]
    return reranker_docs


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    # 获取并校验参数
    rewritten_query, rrf_chunks, web_search_docs = validate_and_get_data(state)
    # 合并数据
    merge_list: list[dict] = merge_rrf_web_list(rrf_chunks, web_search_docs)
    if not merge_list:
        logger.warning("融合后候选为空,跳过重排序")
        state['reranked_docs'] = []
        return state
    # 拼接问题\答案 对（依赖 Reranker 分词器，需一并纳入降级保护）
    try:
        question_answer_pair_list: list[list[str]] = create_question_answer_pair_list(rewritten_query, merge_list)
        # 模型打分
        score_list: list[float] = reranker_compute_score(question_answer_pair_list)
        #  给 merge_list 赋值,确保原有 content
        merge_list_add_score(merge_list, score_list)
        # merge_list 排序
        sort_merge_list(merge_list)
        # 动态 top_k 截取
        reranker_docs: list[dict] = dynamic_top_k(merge_list)
    except Exception as e:
        # Reranker 不可用时降级：保留 RRF 融合顺序（稳定性要求）
        logger.warning(f"Reranker 不可用,降级使用 RRF 融合顺序:{e}")
        reranker_docs = merge_list[:RERANK_MAX_TOPK]
    # 更新 state
    state['reranked_docs'] = reranker_docs
    return state
