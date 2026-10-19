from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf2md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_audiobook_meta import node_audiobook_meta
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus

load_dotenv()

workflow = StateGraph(ImportGraphState)

# 添加节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_audiobook_meta", node_audiobook_meta)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_md_img", node_md_img)

# 开始节点
workflow.set_entry_point("node_entry")

# 条件边函数
def after_node_entry(state: ImportGraphState):
    if state["is_md_read_enabled"]:
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    else:
        return END

# 添加条件边
workflow.add_conditional_edges("node_entry", after_node_entry,
                               {
                                   "node_md_img": "node_md_img",
                                   "node_pdf_to_md": "node_pdf_to_md",
                                   END: END
                               })
# 添加边
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_audiobook_meta")
workflow.add_edge("node_audiobook_meta", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 编译图
graph_app = workflow.compile()

if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    import os
    from app.shared.runtime.logger import logger

    logger.info("===== 开始执行【本地MD直接导入】测试，跳过PDF解析 =====")

    sub_dir = "20261010/4332f43b-b17f-4201-9745-d65c3ef92f6a/hak180产品安全手册"
    md_filename = "hak180产品安全手册_new.md"
    test_md_path = os.path.join(PROJECT_ROOT, "output", sub_dir, md_filename)
    test_output_dir = os.path.join(PROJECT_ROOT, "output", sub_dir)

    os.makedirs(test_output_dir, exist_ok=True)

    if not os.path.exists(test_md_path):
        logger.error(f"MD文件不存在：{test_md_path}")
    else:
        test_state = ImportGraphState({
            "task_id": "test_md_import_workflow_001",
            "local_md_path": test_md_path,
            "local_file_path": test_md_path, # 补齐entry强校验字段
            "local_dir": test_output_dir,
            "is_pdf_read_enabled": False,
            "is_md_read_enabled": True,
        })
        try:
            logger.info(f"MD文件路径：{test_md_path}")
            logger.info("流程链路：entry → md_img → split → item_name → embedding → milvus")

            final_state = None
            for step in graph_app.stream(test_state, stream_mode="values"):
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step

            if final_state:
                logger.info("-" * 80)
                logger.info("===== MD导入流程执行成功，核心结果预览 =====")
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]
                item_name = final_state.get("item_name", "未识别")
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False

                logger.info(f"📄 MD内容预览（前150字符）：{md_content}...")
                logger.info(f"🏷️  识别的主体名称：{item_name}")
                logger.info(f"📝 文档切分总切片数：{chunk_count}")
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"🗄️  所有切片是否完成Milvus入库：{'是' if has_chunk_id else '否'}")
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            logger.exception(f"===== MD导入流程运行失败 =====")
    logger.info("===== MD导入测试结束 =====")
