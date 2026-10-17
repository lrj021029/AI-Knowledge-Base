import shutil
import time
from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger , PROJECT_ROOT
from app.infra.config.providers import infra_config
import requests


def validates_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """校验 PDF 路径"""
    # 从state中获取pdf_path和local_dir地址
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 非空校验
    if not pdf_path:
        logger.error("pdf_path为空，业务无法进行")
        raise ValueError("pdf_path为空，业务无法进行")
    if not local_dir:
        # 默认值
        local_dir = PROJECT_ROOT / "output"
        logger.warning(f"local_dir无默认值,配置默认输出地址为:{local_dir},业务继续")
        state['local_dir'] = str(local_dir)

    # 将pdf_path和local_dir转成Path对象
    pdf_path_obj:Path = Path(pdf_path)
    local_dir_obj:Path = Path(local_dir)

    # 检查pdf_path_obj是否真实存在
    if not pdf_path_obj.is_file():
        logger.error(f"pdf 路径地址{pdf_path}，但pdf对象不存在，业务无法进行")
        raise FileNotFoundError(f"pdf 路径地址{pdf_path}，但pdf对象不存在，业务无法进行")
    # 检查local_dir_obj是否是真实存在
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir地址{local_dir},但是路径不存在，需创建再继续业务")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


def apply_mineru_get_upload_url(pdf_file_name: str) ->tuple[str, str]:
    """申请 mineru 上传链接"""
    # 对应的请求头(token)
    token = infra_config.mineru_config.api_key
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    # 对应的请求体json数据(请求参数)
    data ={
        "files": [
            {"name": pdf_file_name}
        ],
        "model_version": MINERU_MODEL_VERSION,
        "extra_formats": ["html"]
    }
    # 向对应的接口发起post方式请求, 并等待
    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"
    # 获取结果
    response = requests.post(
        url = url,
        json=data,
        headers=header,
        timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )
    # 状态码
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"网络异常 status_code:{status_code}, 业务无法进行")
        raise RuntimeError(f"网络异常 status_code:{status_code}, 业务无法进行")
    # 判断业务状态码code
    response_json = response.json()
    code = response_json.get("code", -1)
    if code != 0:
        logger.error(f"业务状态码 code:{code},业务无法进行")
        raise RuntimeError(f"业务状态码 code:{code},业务无法进行")
    # 获取目标结果batch_id / upload_url地址
    json_data = response_json.get("data", {})
    batch_id:str = json_data.get("batch_id")
    upload_file_url:str = json_data.get("file_urls")[0]
    # 返回结果即可
    return batch_id, upload_file_url

"""
upload_file_to_url(upload_url地址:str,pdf_path_obj:Path) -> None:
读取上传文件的数据pdf_path_obj->字节的数据
向指定的地址发起put请求,传递pdf文件数据
判断响应状态码,不是200抛出异常! [本次只需要判断http状态码,不需要判断业务code! 本身就是不是minerU业务接口,文件服务器的上传地址]
"""
def upload_file_to_url(upload_url:str,pdf_path_obj:Path) -> None:
    """文件上传 URL"""
    data = pdf_path_obj.read_bytes()
    # 发起 put 请求
    with requests.Session() as s:
        s.trust_env = False
        response = s.put(url = upload_url, data=data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    # 状态码检查
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"{upload_url}上传文件失败，状态码{status_code}")
        raise RuntimeError(f"{upload_url}上传文件失败，状态码{status_code}")


def poll_mineru_zip_url(batch_id:str) -> str:
    """
    常量参数
    最大轮询时间 / 轮询间隔时间 / 记录起始时间
    """
    max_timeout = MINERU_POLL_TIMEOUT_SECONDS
    interval_time  = MINERU_POLL_INTERVAL_SECONDS
    start_time = time.time()
    url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"

    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
    }

    while True:
        # 是否超时
        if time.time() - start_time > max_timeout:
            logger.error(f"获取结果超时，等待时间为{time.time() - start_time},超过最大等待时间{max_timeout},业务终止。")
            raise TimeoutError(f"获取结果超时，等待时间为{time.time() - start_time},超过最大等待时间{max_timeout},业务终止。")
        # 查询解析状态
        response = requests.get(url = url, timeout = MINERU_POLL_TIMEOUT_SECONDS, headers = header)
        status_code = response.status_code

        # 网络状态码
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"获取结果超时，状态码{status_code},稍后重试")
                time.sleep(interval_time)
                continue
            # 不可重试
            logger.error(f"解析结果异常，状态码{status_code},异常无法修复，业务终止")
            raise RuntimeError(f"解析结果异常，状态码{status_code},异常无法修复，业务终止")
        json_data = response.json()
        code = json_data.get("code", -1)
        # 业务状态码
        if code != 0:
            logger.error(f"解析结果状态异常{code},异常无法修复，业务终止")
            raise RuntimeError(f"解析结果状态异常{code},异常无法修复，业务终止")
        # 判断文件解析状态
        result = json_data.get('data',{}).get('extract_result',[])[0]
        state = result.get("state")

        if state == 'done':
            full_zip_url:str = result.get('full_zip_url')
            logger.info(f"完成内容解析，下载地址:{full_zip_url}")
            return full_zip_url
        elif state == 'failed':
            logger.error(f"解析文件失败,业务终止")
            raise RuntimeError(f"解析文件失败,业务终止")
        else:
            logger.info(f"文件解析中，请稍等")
            time.sleep(interval_time)
            continue


def download_and_extract_md(zip_file_url:str, local_dir_obj:Path , stem:str) -> str:
    """下载并提取 md 文件"""
    # 向指定的zip下载地址发起请求(get) 响应字节数据(运行内存)
    response = requests.get(url = zip_file_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"状态码错误{status_code},业务终止")
        raise RuntimeError(f"状态码错误{status_code},业务终止")
    # 创建一个存储zip数据的文件  output / stem_result.zip
    zip_file_obj: Path = local_dir_obj / f"{stem}.zip"
    # 将字节数据写到zip的文件中(write_bytes(data))
    data = response.content
    zip_file_obj.write_bytes(data)
    zip_file_dir: Path = local_dir_obj / stem
    # 判断文件地址是否存在 存在先清空, 删除
    if zip_file_dir.is_dir():
        shutil.rmtree(zip_file_dir)
    # 真实创建对应的文件夹
    zip_file_dir.mkdir(parents=True, exist_ok=True)
    # output / stem
    # zip文件解压到对应输出文件夹
    shutil.unpack_archive(zip_file_obj,zip_file_dir)
    # 检查输出文件夹中是否存在md文件
    md_list:list[Path] = list(zip_file_dir.rglob("*.md"))
    # 没有 -> 终止业务
    if not md_list:
        logger.error(f"文件已解压到{local_dir_obj},但是找不到 md")
        raise RuntimeError(f"文件已解压到{local_dir_obj},但是找不到 md")
    # 检查md文件是否叫 stem.md, 是直接返回即可
    for md_file in md_list:
        if md_file.stem == stem:
            logger.info(f"解压后文件名为{stem},无需重命名")
            return str(md_file)
    # Path -> str
    # 再检查叫不叫full.md -> 重命名 -> 新地址返回
    target_md_obj:Path | None = None
    for md_file in md_list:
        if md_file.stem == "full":
            target_md_obj = md_file
            break
    if target_md_obj:
        # 重命名
        new_target_md_obj: Path = target_md_obj.with_name(f"{stem}.md")
        # 移动到new_target_obj位置
        new_md_obj = target_md_obj.rename(new_target_md_obj)
        logger.info(f"已完成重命名，新地址为{new_md_obj}")
        return str(new_md_obj)
    else:
        # 终止业务
        logger.error(f"解压后文件中有{len(md_list)}个文件，但是不存在文件名为{stem}/full.md的文件，业务终止")
        raise RuntimeError(f"解压后文件中有{len(md_list)}个文件，但是不存在文件名为{stem}/full.md的文件，业务终止")


def parse_pdf2md(state: ImportGraphState) -> ImportGraphState:
    """pdf 转 md"""
    # 参数校验
    pdf_path_obj, local_dir_obj = validates_pdf_paths(state)
    # 向minerU申请文件上传地址和batch_id
    batch_id, upload_file_url = apply_mineru_get_upload_url(pdf_path_obj.name)
    # 向指定的文件地址(预签名地址)上传文件数据
    upload_file_to_url(upload_file_url,pdf_path_obj)
    # 向指定的文件地址(预签名地址)上传文件数据
    full_zip_url:str = poll_mineru_zip_url(batch_id)
    md_path:str = download_and_extract_md(full_zip_url,local_dir_obj,pdf_path_obj.stem)
    state["md_path"] = md_path
    return state