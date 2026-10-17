import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger
from typing import Any


def validates_and_get_data(state:ImportGraphState) -> tuple[str,str,Path]:
    md_content:str = state.get("md_content")
    md_path   :str = state.get("md_path")
    file_title:str = state.get("file_title")

    if not md_path:
        logger.error(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
        raise ValueError(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
    md_path_obj:Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")
        raise FileNotFoundError(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")

    if not md_content:
        # 关键修复：统一使用utf‑8‑sig兼容Windows带BOM md文件
        md_content = md_path_obj.read_text(encoding="utf-8-sig")
        md_content = md_content.lstrip("\ufeff")
        if not md_content:
            logger.error(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
            raise ValueError(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
        state['md_content'] = md_content

    if not file_title:
        file_title = md_path_obj.stem
        logger.warning(f"file_title为空,给与默认值:{file_title},业务继续!")

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 调试，上线可注释
    logger.debug(f"md_content头部:{repr(md_content[:80])}")
    return md_content,file_title,md_path_obj



def document_split_by_title(md_content:str) -> list[dict[str,Any]]:
    """原则一: 语义优先"""
    chunks_title:list[dict[str,Any]] = []
    current_title:str = None
    current_title_lines:list[str] = []
    is_code:bool = False
    # 对清洗后的内容按照"\n"进行切割
    md_lines:list[str] = md_content.split("\n")
    # 定义正则对应的标题
    title_re = re.compile(r"^\s*#{1,6}\s.+$", re.DOTALL)
    for line in md_lines:
        # 空行处理
        line = line.strip()
        if not line:
            logger.debug(f"当前行为空行,跳过")
            continue
        logger.debug(f"当前行为正常行,继续")

        # 代码块处理
        if line.startswith("```") or line.startswith("~~~"):
            is_code = not is_code
            current_title_lines.append(line)
            continue
        # 是否是标题
        match = title_re.match(line)

        if not is_code and match:
            if current_title and len(current_title_lines)>1:
                chunks_title.append(
                    {
                        "title":current_title,
                        "content":"\n".join(current_title_lines)
                    }
                )
            if not current_title and len(current_title_lines)>0:
                current_title = line
                current_title_lines.append(line)
                continue

            current_title = line
            current_title_lines = [line]
        else:
            current_title_lines.append(line)
    if current_title and len(current_title_lines) > 1:
        chunks_title.append(
            {
                "title": current_title,
                "content": "\n".join(current_title_lines)
            }
        )
    logger.info(f"将文档切成了{len(chunks_title)}块")
    return chunks_title


def improve_chunk_size(chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
    """原则二和原则三: 合适大小和边界清晰"""
    logger.info(f"没有切割,没有合并的原始数据长度{len(chunks)}")
    # 定义接收的 chunks
    final_chunks:list[dict[str,Any]] = []
    for chunk in chunks:
        if len(chunk.get("content")) > 600:
            # 切割
            final_chunks.extend(_content_split(chunk))
        else:
            final_chunks.append(chunk)
    logger.info(f"按标题切割,但是没有合并的数据长度{len(final_chunks)}")
    # 合并同一个标题下过短的 chunk 内容,content 小于 400
    final_chunks = _content_merge(final_chunks)
    logger.info(f"按标题切割,合并后的数据长度{len(final_chunks)}")
    return final_chunks

def _content_split(chunk:dict[str,Any]) -> list[dict[str,Any]]:
    """递归切割器"""
    current_title_chunks:list[dict[str, Any]] = []
    # 数据前置处理
    content:str = chunk.get("content")
    title:str = chunk.get("title")
    # 标题\n
    content = content[len(title)+1:]
    pre_fix:str = title+"\n"

    # 声明递归切割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。","！","？","；"],
        # 配置四个常量
        chunk_size=CHUNK_SIZE - len(pre_fix),
        chunk_overlap=CHUNK_OVERLAP
    )

    # 递归切割器切割数据
    chunk_list:list[str] = text_splitter.split_text(content)
    for index, chunk_content in enumerate(chunk_list, start=1):
        sub_chunk = {
            "parent_title":title,
            "title":f"{title}_{index}",
            "part":index,
            "content":pre_fix+chunk_content
        }
        current_title_chunks.append(sub_chunk)
    logger.info(f"完成{title}内容切割,数量为{len(current_title_chunks)}")
    return current_title_chunks


def _content_merge(final_chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
    chunks:list[dict[str,Any]] = []
    base_chunk:dict[str,Any] = None
    logger.info(f"合并后的数据:{len(final_chunks)}")
    for next_chunk in final_chunks:
        if not base_chunk:
            # 第一次循环
            base_chunk = next_chunk
            continue
        if len(base_chunk.get("content")) < CHUNK_MIN:
            # 判断是否为同一父标题 (非空校验)
            if base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get("parent_title"):
                if (len(base_chunk.get("content")) + len(next_chunk.get("content"))) < CHUNK_MAX_SIZE:
                    base_chunk["content"] = base_chunk.get("content") + "\n" + next_chunk.get("content")
                else:
                    chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                chunks.append(base_chunk)
                base_chunk = next_chunk
        else:
            chunks.append(base_chunk)
            base_chunk = next_chunk
    if base_chunk:
        # 最后未添加的
        chunks.append(base_chunk)
    logger.info(f"合并后数据长度:{len(chunks)}")
    return chunks


def append_chunk_meta_data(chunks:list[dict[str,Any]], file_title:str) -> list[dict[str,Any]]:
    """补充追加元数据"""
    for chunk in chunks:
        chunk["file_title"] = file_title
        if "part" not in chunk:
            chunk["part"] = 1
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk["title"]
    logger.info(f"完成 chunk 元数据基本补充")


def backup_chunks_json(chunks, md_path_obj:Path):
    """备份 chunk"""
    # 对象路径拼接
    chunks_json_obj:Path = md_path_obj.parent / "chunks.json"
    chunks_json_obj.write_text(data=json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"已完成{md_path_obj.name}对应的数据内容备份")


def split_document(state: ImportGraphState) -> ImportGraphState:
    """切分文档"""
    # 参数校验&数据获取
    md_content, file_title, md_path_obj = validates_and_get_data(state)
    # 初步语义切割
    title_chunks:list[dict[str,Any]] = document_split_by_title(md_content)
    # 同标题下长度优化处理&添加重叠部分
    chunks:list[dict[str,Any]] = improve_chunk_size(title_chunks)
    # 元数据补充
    append_chunk_meta_data(chunks, file_title)
    # 数据备份
    backup_chunks_json(chunks, md_path_obj)
    # 更新&返回 state
    state['chunks'] = chunks
    return state