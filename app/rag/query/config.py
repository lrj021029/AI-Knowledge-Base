from app.shared.config.common import env_int

# 主体名称确认阈值：高于该分数 → 直接确认 -> 1
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2

# rrf切割的top_k
NODE_RRF_TOP_K = 5

RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.2
RERANK_GAP_ABS: float = 0.2

RERANK_MAX_INPUT_TOKENS: int = 512  # 重排序模型的上下文长度
RERANK_SUMMARY_CHAR_RATIO: float = 1.2  # 经验值 token 转 字符串比  120 token => 100
RERANK_MIN_SUMMARY_CHARS: int = 50     #  最小压缩的字符数量(面临截取)  512 -> 500 -> 12 -> 10  <- 500

# =========================
# 听书业务意图（新增）
# =========================
INTENT_BOOK_RECOMMEND = "book_recommend"   # 书籍推荐
INTENT_BOOK_DETAIL = "book_detail"         # 书籍详情
INTENT_NOTE_SEARCH = "note_search"         # 听书笔记/资料检索
INTENT_GENERAL_QA = "general_qa"           # 通用知识问答

# 听书业务分支（复用 HyDE + 多路召回 + RRF + Rerank 链路，但走元数据过滤与业务回答）
AUDIOBOOK_INTENTS = {INTENT_BOOK_RECOMMEND, INTENT_BOOK_DETAIL, INTENT_NOTE_SEARCH}

ALL_INTENTS = {INTENT_BOOK_RECOMMEND, INTENT_BOOK_DETAIL, INTENT_NOTE_SEARCH, INTENT_GENERAL_QA}

# 书籍推荐默认返回的书籍数量
BOOK_RECOMMEND_DEFAULT_LIMIT = 12
# 单本书籍详情聚合时最多读取的切片数
BOOK_DETAIL_MAX_CHUNKS = 200

# 精品推荐「LLM 推荐理由」进程内缓存。
# 该步骤是 /book/recommend 的绝对耗时大头（实测约 10s，Milvus 过滤仅约 30ms），
# 缓存命中后同一筛选条件下的翻页/重复访问可降到毫秒级。
# TTL 设为 0 表示关闭缓存；书籍内容变化时缓存键会变化，无需手动失效。
BOOK_RECOMMEND_REASON_CACHE_TTL = env_int("BOOK_RECOMMEND_REASON_CACHE_TTL", 1800)
BOOK_RECOMMEND_REASON_CACHE_MAXSIZE = env_int("BOOK_RECOMMEND_REASON_CACHE_MAXSIZE", 128)