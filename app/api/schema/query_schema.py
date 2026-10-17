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