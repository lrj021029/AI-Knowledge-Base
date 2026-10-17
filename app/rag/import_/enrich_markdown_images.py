from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.rag.import_.config import IMAGE_CONTEXT_CHAR_LENGTH, SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.object_storage.minio_gateway import minio_gateway
from app.shared.runtime.load_prompt import load_prompt
from minio import Minio
from mimetypes import guess_type
import base64
import  re

def validates_and_get_data(state:ImportGraphState) -> tuple[str,Path,Path]:
    """校验并获取数据"""
    # 从 state 中获取 md_path
    md_path = state.get("md_path")
    # md_path非空校验
    if not md_path:
        logger.error(f"md_path内容为空，业务终止")
        raise RuntimeError(f"md_path内容为空，业务终止")
    # 转成md_path_obj Path对象, 做存在性校验! is_file()
    md_path_obj:Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path地址{md_path},但是没有对应的文件，业务终止")
        raise FileNotFoundError(f"md_path地址{md_path},但是没有对应的文件，业务终止")
    # 使用md_path对象读取md_content,更新state
    md_content:str = md_path_obj.read_text(encoding="utf-8")
    state["md_path"] = md_content
    # md_path_obj可以获取images path地址
    images_path_obj:Path = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_path_obj


def scan_images_info(images_path_obj:Path,md_content:str) -> list[tuple[str,Path,tuple[str,str]]]:
    """提取图片对象上下文信息"""
    # 获取images下的所有图片Path对象 [列表/生成器]
    generator = images_path_obj.iterdir()
    images_info_list:list[tuple[str,Path,tuple[str,str]]] = []
    # 循环每一张图片对象
    for image_file_obj in generator:
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.debug(f"{images_path_obj.name}不是图片，跳过处理后续图片内容")
            continue
        # 定义这个图片的正则规则
        # 使用正则到md_content中进行匹配
        image_name:str = image_file_obj.name
        # match = re.match(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)", md_content)
        match = re.search(rf"!\[.*?\]\(.*?{re.escape(image_name)}.*?\)", md_content, flags=re.DOTALL)
        if not match:
            logger.debug(f"{image_name}没有在 md_content被使用过，跳过处理后续内容")
            continue
        start = match.start()
        end = match.end()
        # 获取对应上下信息
        pre_context = md_content[max(start-IMAGE_CONTEXT_CHAR_LENGTH,0):start]
        post_context = md_content[end:min(end+IMAGE_CONTEXT_CHAR_LENGTH,len(md_content))]
        # 组装当前的数据元组 (图片名字str,图片Path对象,(上文,下文)) 添加到大的列表
        images_info_list.append((image_name,image_file_obj,(pre_context,post_context)))

    # 返回列表数据
    logger.info(f"完成了{len(images_info_list)}张图片内容匹配")
    return images_info_list


"""
summary_image_info(image_info:list[tuple[str,Path,tuple[str,str]]],root_folfer:str) -> dict[str,str]
数据准备工作 初始化一个视觉模型对象 | 定义一个接收数据的字典
循环处理每一张图片    for str,Path,context in  list[tuple[str,Path,tuple[str,str]]]
封装模型提示词信息 
封装调用链: 直接 model.invoke | 调用链 提示词 | 模型 | 解析器 OutParser() 
使用调用链进行调用和获取结果
添加到字典中 {image_name,image_summary}
跳出循环->返回整体字典
"""
def summary_image_info(image_info:list[tuple[str,Path,tuple[str,str]]],root_folder:str) -> dict[str,str]:
    """总结图片信息"""
    # 初始化接收数据的空字典
    image_summary:dict[str,str] = {}
    # 初始化视觉模型
    vision_model = llm_provider.vision_chat()
    # 循环处理图片
    for image_name, image_path_obj, image_context in image_info:
        # 封装模型提示词信息
        image_des: str = load_prompt("image_summary", image_content=image_context, root_folder=root_folder)        # 类型校验
        image_type = guess_type(image_name)[0]
        # base64编码
        image_b64 = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": image_des},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_type};base64,{image_b64}"},
                },
            ]
        )
        # 封装调用链: 直接 model.invoke | 调用链 提示词 | 模型 | 解析器 OutParser()
        chains = vision_model | StrOutputParser()
        # 调用链并获取结果
        summary:str = chains.invoke([message])
        # 添加到字典
        image_summary[image_name] = summary
        logger.debug(f"图片{image_name}通过视觉模型识别，结果为{summary}")
    logger.info(f"共{len(image_summary)}被解析，于{root_folder}")
    return image_summary


"""
upload_image_to_minio(image_info:list[tuple[str,Path,tuple[str,str]]],stem) -> dict[str,str]:
4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
4.2 根据桶信息查询对应旧图片对象
4.3 批量删除旧图片对象
4.4 循环处理每一张图片  for image_name,image_path_obj,_ in image_info:
    4.5 单次循环的异常处理,失败,跳过图片即可,进行下一张!
    4.6 调用minio_client.fput方法将图片上传到minio服务器
    4.7 根据信息调用infra构建图片的访问地址
    4.8 将图片访问地址存储到dict image_name | image_url
4.9 最终返回dict数据
"""
def upload_image_to_minio(image_info:list[tuple[str,Path,tuple[str,str]]],stem) -> dict[str,str]:
    """上传图片至 minio"""
    images_url:dict[str,str] = {}
    minio_client:Minio = minio_gateway.client
    # 根据桶信息查询对应旧图片对象
    objects:list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=minio_gateway.image_dir[1:]+"/"+stem,
        recursive=True
    )
    # 批量删除旧图片对象
    if objects:
        delete_object_list = [DeleteObject(obj.object_name) for obj in objects]
        errors = minio_client.remove_objects(
            bucket_name=minio_gateway.bucket_name,
            delete_object_list=delete_object_list
        )
        # remove_objects方法返回的是一个生成器, 需要循环处理结果, 才会真的触发!!!remove_objects方法返回的是一个生成器,需要循环处理结果,才会真的触发!!!
        for error in errors:
            logger.warning(f"删除{stem}文件出现异常，{error}")

    # 循环处理每一张图片
    for image_name, image_path_obj, _ in image_info:
        minio_client.fput_object(
            bucket_name=minio_gateway.bucket_name,
            object_name=minio_gateway.image_dir+"/"+stem+"/"+image_name,
            file_path=str(image_path_obj),
            content_type=guess_type(image_name)[0]
        )
        try:
            url:str = minio_gateway.build_image_url(stem, image_name)
            images_url[image_name] = url
            logger.debug(f"{image_name}文件上传到服务器，地址为：{url}")
        except Exception as e:
            logger.exception(f"{image_name}上传失败，跳过进行下一张")
            continue
    logger.info(f"已将所有图片上传{stem}，数量为{len(images_url)}")
    return images_url


def replace_old_md_content(md_content:str, images_url:dict[str,str], images_summary:dict[str,str]) -> str:
    """替换新内容（图片描述&网络地址）"""
    # 循环images_url可以获取每张图片的image_name 网络地址
    for image_name, url in images_url.items():
        summary = images_summary.get(image_name)
        image_re = re.compile(
            rf"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)"
        )
        # 替换后内容
        md_content:str = image_re.sub(lambda _: f"![{summary}]({url})", md_content)
    return md_content


def backup_md_return_path(md_content:str, md_path_obj:Path):
    """写入新替换内容至磁盘"""
    new_md_path_obj:Path = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    new_md_path_obj.write_text(data=md_content, encoding="utf-8")
    return str(new_md_path_obj)


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """md 图片处理"""
    # 获取参数并校验
    md_content, md_path_obj,images_path_obj = validates_and_get_data(state)
    if not list(images_path_obj.iterdir()):
        logger.info("文件中不包含图片，跳转至下一节点")
        return state
    # 图片信息识别
    images_info_list:list[tuple[str,Path,tuple[str,str]]] = scan_images_info(images_path_obj,md_content)
    image_summary:dict[str,str] = summary_image_info(images_info_list,md_path_obj.stem)
    # 图片上传
    images_url:dict[str,str] = upload_image_to_minio(images_info_list, images_path_obj.stem)
    # 内容替换
    new_md_content:str = replace_old_md_content(md_content, images_url, image_summary)
    # 数据备份
    new_md_path:str = backup_md_return_path(new_md_content, md_path_obj)
    # 更新 state
    state['md_path'] = new_md_path
    state['md_content'] = new_md_content
    return state