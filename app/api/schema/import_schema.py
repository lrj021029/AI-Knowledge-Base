from pydantic import BaseModel


class UploadResponse(BaseModel):
    code: int
    messages: str
    task_ids: list[str]


class StatusResponse(BaseModel):
    code: int
    status: str
    task_id: str
    done_list: list[str]
    running_list: list[str]
    # 失败原因（导入失败时持久化，便于前端与排查）
    error: str = ""
    # 该任务抽取到的听书元数据（导入完成后返回）
    audiobook_meta: dict = {}
