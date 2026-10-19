import json
import uuid
import shutil
from mimetypes import guess_type
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, BackgroundTasks, UploadFile, Form
from app.api.schema.import_schema import UploadResponse, StatusResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    get_task_result,
    set_task_result,
    update_task_status,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from app.process.import_.agent.main_graph import graph_app
from app.process.import_.agent.state import create_default_state, ImportGraphState
from datetime import datetime

app = FastAPI(description="导入模块的接口")

# 前台可填写的听书元数据表单字段（作为手动元数据写入导入状态）
_MANUAL_META_FORM_FIELDS = (
    "book_name",
    "author_name",
    "content_type",
    "category",
    "scene_tags",
    "audio_duration",
    "narrator",
    "highlights",
    "suitable_audience",
    "summary",
    "source_path",
)


def invoke_import_graph(task_id: str, local_file_path: str, local_dir: str, manual_meta: dict):
    """
    执行单个文件的导入图。

    异常隔离：单个文档导入失败只影响自身任务，不中断其它文件；
    失败原因通过 set_task_result 持久化，并写入文件日志。
    """
    state: ImportGraphState = create_default_state(
        task_id=task_id,
        local_file_path=local_file_path,
        local_dir=local_dir,
        manual_meta=manual_meta or {},
    )
    logger.info(f"执行参数: {json.dumps(state, indent=4, ensure_ascii=False, default=str)}")
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        result: ImportGraphState = graph_app.invoke(state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        # 回写抽取到的听书元数据，供 /status 查询
        set_task_result(task_id, "audiobook_meta", result.get("audiobook_meta", {}))
    except Exception as e:
        logger.exception(f"导入流程异常 task_id={task_id}:{str(e)}")
        set_task_result(task_id, "error", str(e))
        update_task_status(task_id, TASK_STATUS_FAILED)


@app.get("/html")
def return_import_html():
    html_file_obj: Path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(
        path=str(html_file_obj),
        media_type=guess_type(html_file_obj)[0],
    )


def _save_upload_file(current_file: UploadFile, task_id: str) -> tuple[str, str]:
    """保存单个上传文件，返回 (local_file_path, local_dir)。"""
    local_dir_obj: Path = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    filename: str = Path(current_file.filename or f"unnamed_{task_id}").name
    local_file_path_obj: Path = local_dir_obj / filename
    with open(local_file_path_obj, "wb") as output_file:
        shutil.copyfileobj(current_file.file, output_file)
    return str(local_file_path_obj), str(local_dir_obj)


@app.post("/upload")
def upload_file(
    tasks: BackgroundTasks,
    files: list[UploadFile],
    book_name: str = Form(""),
    author_name: str = Form(""),
    content_type: str = Form(""),
    category: str = Form(""),
    category_tags: str = Form(""),
    scene_tags: str = Form(""),
    audio_duration: str = Form(""),
    narrator: str = Form(""),
    highlights: str = Form(""),
    suitable_audience: str = Form(""),
    summary: str = Form(""),
    source_path: str = Form(""),
):
    """
    上传文件（支持单文件导入与文件夹选择多文件导入）。

    - 每个文件独立生成 task_id 与输出目录，互不影响；
    - 同一份手填元数据表单会应用到本次上传的所有文件。
    """
    if not files:
        return UploadResponse(code=400, messages="未接收到任何文件", task_ids=[])

    manual_meta: dict = {
        "book_name": book_name,
        "author_name": author_name,
        "content_type": content_type,
        "category": category,
        "category_tags": category_tags,
        "scene_tags": scene_tags,
        "audio_duration": audio_duration,
        "narrator": narrator,
        "highlights": highlights,
        "suitable_audience": suitable_audience,
        "summary": summary,
        "source_path": source_path,
    }
    # 过滤掉空值，避免空字符串覆盖自动抽取结果
    manual_meta = {k: v for k, v in manual_meta.items() if v not in (None, "")}

    task_ids: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []
    for current_file in files:
        task_id = str(uuid.uuid4())
        try:
            local_file_path, local_dir = _save_upload_file(current_file, task_id)
        except Exception as e:
            logger.exception(f"文件保存失败:{current_file.filename},{str(e)}")
            failed.append(current_file.filename or task_id)
            continue
        task_ids.append(task_id)
        succeeded.append(current_file.filename or task_id)
        # 每个文件独立后台任务，单文件异常不影响其它文件
        tasks.add_task(
            invoke_import_graph,
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
            manual_meta=manual_meta,
        )

    message = f"已接收{len(succeeded)}个文件,正在解析..."
    if failed:
        message += f";{len(failed)}个文件保存失败:{','.join(failed)}"
    return UploadResponse(code=200, messages=message, task_ids=task_ids)


@app.get("/status/{task_id}")
def status_info(task_id: str):
    status = get_task_status(task_id)
    running_list = get_running_task_list(task_id)
    done_list = get_done_task_list(task_id)
    return StatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        running_list=running_list,
        done_list=done_list,
        error=get_task_result(task_id, "error", "") or "",
        audiobook_meta=get_task_result(task_id, "audiobook_meta", {}) or {},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
