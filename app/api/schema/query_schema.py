from pydantic import BaseModel

class QueryRequest(BaseModel):
    query:str
    session_id:str
    is_stream:bool

class QueryAsyncResponse(BaseModel):
    message:str
    session_id:str

class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer:str
    done_list:list[str]
    image_urls:list[str]
    # 听书业务扩展：溯源信息、意图、结构化结果
    citations: list[dict] = []
    intent: str = ""
    structured_result: dict = {}


class HistoryDeleteResponse(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponse(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str
    item_names:list[str]
    image_urls:list[str]
    ts:float   # Any


class HistoryResponse(BaseModel):
    session_id:str
    items:list[HistoryItemResponse]

class SessionItemResponse(BaseModel):
    session_id:str
    title:str          # 会话标题：该会话的第一条用户提问
    count:int          # 消息条数
    last_ts:float      # 最后一条消息时间戳（秒）

class SessionListResponse(BaseModel):
    sessions:list[SessionItemResponse]