import json
import uuid
import shutil
from mimetypes import guess_type
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, BackgroundTasks, UploadFile
from app.api.schema.import_schema import UploadResponse, StatusResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.task_utils import get_done_task_list, get_running_task_list, get_task_status, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from app.process.import_.agent.main_graph import graph_app
from app.process.import_.agent.state import create_default_state, ImportGraphState
from datetime import datetime
app = FastAPI(description="导入模块的接口")


def invoke_import_graph(task_id:str,local_file_path:str,local_dir:str):
    state: ImportGraphState = create_default_state(
        task_id=task_id,
        local_file_path=local_file_path,
        local_dir=local_dir,
    )
    logger.info(f"执行参数: {json.dumps(state, indent=4, ensure_ascii=False)}")
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        result: ImportGraphState = graph_app.invoke(state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        logger.exception(f"导入流程异常:{str(e)}")
        update_task_status(task_id, TASK_STATUS_FAILED)


@app.get("/html")
def return_import_html():
    html_file_obj:Path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(
        path=str(html_file_obj),
        media_type=guess_type(html_file_obj)[0],
    )


@app.post("/upload")
def upload_file(tasks:BackgroundTasks, files:list[UploadFile]):
    current_file:UploadFile = files[0]
    task_id = str(uuid.uuid4())
    local_dir_obj:Path = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    local_file_path_obj:Path = local_dir_obj / current_file.filename
    with open(local_file_path_obj, "wb") as output_file:
        shutil.copyfileobj(current_file.file, output_file)
    tasks.add_task(invoke_import_graph, task_id=task_id, local_file_path=str(local_file_path_obj), local_dir=str(local_dir_obj))
    return UploadResponse(
        code=200,
        messages=f"{current_file.filename}已上传,正在解析...",
        task_ids=[task_id]
    )

@app.get("/status/{task_id}")
def status_info(task_id:str):
    status = get_task_status(task_id)
    running_list = get_running_task_list(task_id)
    done_list = get_done_task_list(task_id)
    return StatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        running_list=running_list,
        done_list=done_list,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)