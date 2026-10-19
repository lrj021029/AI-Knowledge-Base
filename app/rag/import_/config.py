from app.shared.config.common import env_bool

# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION = "vlm"
# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 -> 一个pdf 约等于 1秒
MINERU_POLL_TIMEOUT_SECONDS = 200
# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 600

IMAGE_CONTEXT_CHAR_LENGTH = 100

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400
# item_name识别的时候,截取chunks的前几个对应的切块
ITEM_NAME_RECOGNITION_CHUNKS_TOP_K = 5
# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5

# Milvus VARCHAR 字段最大长度（用于 title、item_name 等短文本）
MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512
# Milvus content 字段最大长度（用于存储长文本内容）
MILVUS_CHUNK_CONTENT_MAX_LENGTH = 65535
# Milvus 向量维度（BGE-M3 稠密向量维度）
MILVUS_VECTOR_DIM = 1024

# =========================
# 听书业务元数据配置（新增）
# =========================
# 内容类型枚举（听书场景），用于元数据抽取与检索过滤
AUDIOBOOK_CONTENT_TYPES = [
    "有声书信息",
    "书籍简介",
    "作者介绍",
    "听书笔记",
    "推荐运营资料",
    "用户评论摘要",
    "常见问答",
]
# 书籍类别枚举
AUDIOBOOK_CATEGORIES = ["科幻", "悬疑", "文学", "历史", "教育", "儿童", "小说", "非小说", "其他"]
# 听书场景标签枚举
AUDIOBOOK_SCENE_TAGS = ["通勤", "睡前", "碎片时间", "长途旅行"]
# 元数据抽取时送入大模型的 Markdown 正文最大字符数（防止超长）
AUDIOBOOK_META_MD_MAX_CHARS = 6000
# 元数据抽取时送入大模型的切片数量上限
AUDIOBOOK_META_CHUNKS_TOP_K = 8
# 手动填写元数据是否优先于 LLM 自动抽取结果。
# True  -> 手动填写优先（符合 SPEC 5.1 与验收标准）
# False -> LLM 自动抽取优先（可用环境变量 AUDIOBOOK_META_MANUAL_FIRST=False 切换）
AUDIOBOOK_META_MANUAL_FIRST = env_bool("AUDIOBOOK_META_MANUAL_FIRST", True)
# 听书业务元数据在 Milvus 中作为动态字段（dynamic field）写入的字段名清单
AUDIOBOOK_META_FIELDS = [
    "content_type",
    "book_name",
    "author_name",
    "entry_name",
    "category",
    # 类别/标签的多值全集（如「科幻|宇宙文明|悬疑」）。
    # category 只是其中最贴合枚举的一个“主类别”，二者含义不同：
    # category 用于展示与筛选栏，category_tags 用于“多标签召回”。
    # 入库时包裹为「|a|b|」，查询侧用 like "%|悬疑|%" 做精确标签匹配，
    # 避免「小说」误命中「非小说」这类子串问题。
    "category_tags",
    "scene_tags",
    "audio_duration",
    "narrator",
    "highlights",
    "suitable_audience",
    "summary",
    "source_file_name",
    "source_path",
    "meta_source",
]