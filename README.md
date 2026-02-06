# 智学助手

AI 驱动的个性化学习平台，基于多代理架构，结合 RAG、网络搜索与多代理协作。

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.10+, FastAPI, LangChain + LangGraph, LlamaIndex |
| 前端 | Next.js 16, React 19, TypeScript, Tailwind CSS |
| 数据库 | PostgreSQL / SQLite（可配置） |
| LLM | OpenAI, Anthropic, Ollama, Dashscope, DeepSeek 等 |

## 功能模块

- **问答解题** — 基于知识库的 RAG 问答
- **问题生成** — 从知识库自动生成练习题
- **引导式学习** — 多轮交互式引导学习
- **深度研究** — 多代理协作的深度知识研究与报告生成
- **知识库管理** — 文档上传、向量索引构建
- **笔记本** — 研究笔记管理

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- PostgreSQL 16（可选，默认 SQLite）

### 1. 安装依赖

```bash
# 后端
pip install -r requirements.txt

# 前端
cd web && npm install
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写必要配置：

```bash
cp .env.example .env
```

必填项：

```env
# LLM 配置
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_HOST=https://api.openai.com/v1

# Embedding 配置（知识库功能需要）
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_HOST=https://api.openai.com/v1
```

### 3. 启动服务

```bash
# 一键启动（后端 + 前端）
python scripts/start_web.py

# 或分别启动
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload  # 后端
cd web && npm run dev                                            # 前端
```

默认端口：后端 `8001`，前端 `3782`。

### Docker 部署

```bash
# 基础部署
docker compose up -d

# 使用 PostgreSQL
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
```

## 项目结构

```
src/
├── agents/           # 代理模块
│   ├── base_agent.py # 统一基类
│   ├── chat/         # 聊天代理
│   ├── guide/        # 引导式学习
│   ├── question/     # 问题生成
│   └── research/     # 深度研究（LangGraph）
├── api/              # FastAPI 后端
│   ├── routers/      # API 路由
│   └── main.py       # 应用入口
├── services/         # 共享服务层
│   ├── llm/          # LLM 编排
│   ├── rag/          # RAG 服务
│   ├── storage/      # 数据持久化
│   └── prompt/       # 提示词管理
├── tools/            # 工具实现
└── knowledge/        # 知识库管理

web/
├── app/              # Next.js 页面
├── components/       # React 组件
├── hooks/            # 自定义 Hooks
├── context/          # 全局状态
└── lib/              # 工具函数

config/
├── main.yaml         # 系统配置
└── agents.yaml       # 代理参数配置
```

## API 端点

| 路径 | 功能 |
|------|------|
| `/api/v1/chat` | WebSocket 聊天 |
| `/api/v1/question` | 问题生成 |
| `/api/v1/research` | 深度研究 |
| `/api/v1/knowledge` | 知识库管理 |
| `/api/v1/guide` | 引导式学习 |
| `/api/v1/notebook` | 笔记本 |
| `/api/v1/config` | 配置管理 |
| `/api/v1/settings` | 用户设置 |

API 文档：`http://localhost:8001/docs`

## 配置说明

### 存储后端

在 `config/main.yaml` 中配置：

```yaml
storage:
  backend: postgres  # file | sqlite | postgres
  postgres_dsn: ""
  auto_migrate: true
```

或通过环境变量：

```env
DEEPTUTOR_STORAGE_BACKEND=postgres
DEEPTUTOR_POSTGRES_DSN=postgresql://user:pass@localhost:5432/deeptutor
```

### 代理参数

在 `config/agents.yaml` 中统一管理各代理的 temperature、max_tokens 等参数。

### 研究模式预设

深度研究支持多种预设：`quick`（快速）、`medium`（中等）、`deep`（深度）、`auto`（自动）。

## 开发

```bash
# 代码检查
ruff check src/
ruff format src/

# 类型检查
mypy src/

# 测试
pytest tests/ --tb=short

# 安全扫描
bandit -r src/
```

## 许可证

[AGPL-3.0](LICENSE)
