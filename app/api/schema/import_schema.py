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