Based on the code map provided, I can see this is a RAG (Retrieval-Augmented Generation) knowledge base project. Let me create a comprehensive README for this project.

# AI 知识库 (AI Knowledge Base)

这是一个基于 RAG 技术的智能知识库系统，支持文档导入、智能问答、混合搜索等功能。项目采用 FastAPI 构建 RESTful API，结合 Milvus 向量数据库、LLM 大语言模型等组件实现高效的知识检索与问答。

## 功能特性

### 文档导入
- **PDF 转 Markdown**：支持将 PDF 文档转换为 Markdown 格式
- **文档智能分块**：基于标题和内容进行智能文档分割
- **BGE 向量嵌入**：使用 BGE-M3 模型生成文档块向量
- **商品名称识别**：通过 LLM 自动识别文档中的商品名称
- **Markdown 图片增强**：自动处理 Markdown 中的图片并上传至对象存储

### 智能问答
- **混合搜索**：支持向量搜索与关键词搜索的混合检索
- **HyDE 搜索**：使用 HyDE (Hypothetical Document Embeddings) 技术提升搜索效果
- **Web 搜索**：集成 MCP Web 搜索能力
- **Rerank 重排序**：使用重排序模型对检索结果进行优化
- **RRF 融合**：通过倒数排名融合 (Reciprocal Rank Fusion) 合并多路搜索结果
- **对话历史管理**：支持多会话对话历史存储与管理

### 技术架构
- **后端框架**：FastAPI
- **向量数据库**：Milvus
- **对象存储**：MinIO
- **大语言模型**：支持多种 LLM 提供商
- **嵌入模型**：BGE-M3
- **重排序模型**：FlagReranker

## 项目结构

```
ai_0812_knowledge_base/
├── app/
│   ├── api/                  # API 接口层
│   │   ├── http/             # HTTP 接口
│   │   │   ├── import_server.py    # 文档导入接口
│   │   │   └── query_server.py     # 问答接口
│   │   └── schema/           # 数据模型
│   ├── infra/                # 基础设施层
│   │   ├── config/           # 配置管理
│   │   ├── llm/              # LLM 提供商
│   │   ├── object_storage/   # 对象存储网关
│   │   ├── persistence/      # 持久化层
│   │   └── vectorstore/      # 向量数据库网关
│   ├── process/              # 业务流程
│   │   ├── import_/          # 导入流程
│   │   │   └── agent/        # 导入 Agent 图
│   │   └── query/            # 查询流程
│   │       └── agent/        # 查询 Agent 图
│   ├── rag/                  # RAG 核心服务
│   │   ├── import_/          # 导入相关服务
│   │   └── query/            # 查询相关服务
│   ├── shared/               # 共享模块
│   │   ├── clients/          # 客户端工具
│   │   ├── config/           # 配置定义
│   │   ├── model/            # 模型工具
│   │   ├── runtime/          # 运行时工具
│   │   └── utils/            # 工具函数
│   └── resources/
│       ├── html/             # 前端页面
│       └── prompts/          # 提示词模板
├── fast_api/                 # 扩展 API 模块
├── test/                     # 测试代码
└── main.py                   # 应用入口
```

## 快速开始

### 环境要求

- Python 3.10+
- Milvus 向量数据库
- MinIO 对象存储
- MongoDB (用于对话历史存储)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 文件并配置相关参数：

```bash
cp .env.example .env
```

主要配置项：
- LLM API 配置
- Milvus 连接配置
- MinIO 连接配置
- MongoDB 连接配置
- Embedding 模型配置

### 启动服务

```bash
python main.py
```

服务启动后，可访问：
- API 文档：`http://localhost:8000/docs`
- 导入界面：`http://localhost:8000/html/import`
- 问答界面：`http://localhost:8000/html/chat`

## API 接口

### 文档导入

| 接口 | 方法 | 描述 |
|------|------|------|
| `/upload` | POST | 上传文档文件 |
| `/status/{task_id}` | GET | 获取导入任务状态 |

### 智能问答

| 接口 | 方法 | 描述 |
|------|------|------|
| `/query` | POST | 提交问答请求 |
| `/stream/{session_id}` | GET | 流式获取回答 |
| `/history/{session_id}` | GET | 获取对话历史 |
| `/history/{session_id}` | DELETE | 清空对话历史 |
| `/health` | GET | 健康检查 |

## 开发指南

### 添加新的导入节点

在 `app/process/import_/agent/nodes/` 目录下创建新的节点文件：

```python
from app.process.import_.agent.state import ImportGraphState

@node_log("node_xxx")
def node_xxx(state: ImportGraphState) -> ImportGraphState:
    # 实现节点逻辑
    return state
```

### 添加新的查询服务

在 `app/rag/query/` 目录下创建新的服务文件，并在查询图 `main_graph.py` 中注册。

## 技术栈

- **Web Framework**: FastAPI
- **Vector Database**: Milvus
- **Object Storage**: MinIO
- **LLM**: OpenAI Compatible APIs
- **Embedding**: BGE-M3
- **Reranker**: FlagReranker
- **Graph Workflow**: LangGraph

## 许可证

本项目仅供学习和研究使用。