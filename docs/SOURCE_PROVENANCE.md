# 来源与发布范围

本包于 2026-10-14 从 Paper-Agent 开发工作区提取，发布目标为独立 RSI 仓库。

- `backend/app/skills/`：完整的 12 个 Skill Pool 模块。
- `tests/skill_pool/`：完整的 6 个工程测试文件。
- `core/config.py`、`core/paths.py`、`rag/vector_store.py`、`services/model_clients.py`：复用现有配置、路径、BGE-M3/Chroma 和 Agent JSON 调用；保留原文。
- `services/llm.py`：只提取 Model Client 所需的两项上下文错误工具，定义原文保持一致；没有复制聊天、Router 或视觉客户端。
- 包初始化文件与 `docs/skill-pool.md` 保留原文。
- 设计 PDF 保持原始字节；源码与 PDF 校验值见 `source-manifest.json`。
- README、打包配置、最小依赖清单、空白环境示例和忽略规则为本次独立发布补充。

没有包含 Paper-Agent 的 Agent 执行入口、前端、HTTP API、评测数据、实验报告、用户 PDF 语料、模型权重、缓存、密钥或运行数据库。完整使用文档中的项目路径示例和 55 项既有模块回归记录属于原 Paper-Agent 开发环境；本包自身测试入口为 `python -m pytest`。
