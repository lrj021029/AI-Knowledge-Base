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