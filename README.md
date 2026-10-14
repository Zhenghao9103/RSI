# RSI — Self-Evolving Skill Pool

从 Paper-Agent 提取的独立 Skill 自进化模块，提供 Python API 和 CLI。
支持经验触发、候选生成/导入、在线语义聚类、Master/Reference 整合，以及晋升、拆分、合并、整理、归档、索引同步和版本恢复。

本仓库包含模块源码、必要基础组件、测试和设计 PDF。模型权重、API 密钥、业务数据库、运行数据和实验结果不随仓库发布。

## 安装与离线演示

需要 Python 3.11 或更新版本。在仓库根目录执行：

```shell
python -m pip install -e .
python -m backend.app.skills --help
python -m backend.app.skills demo
```

安装后也可运行 `rsi-skill-pool demo`。离线演示使用明确注入的确定性向量与决策，SQLite、文件和 Chroma 存储均为真实实现；不调用远程模型，也不下载 BGE 权重。它覆盖同簇整合、Reference 更新、晋升、拆分、向量保留和重启一致性。

## 真实模型与 Python API

生产编码使用 BGE-M3。首次运行 `python -m backend.app.skills smoke` 时，可能从 Hugging Face 下载权重并缓存到本地；权重没有上传到本仓库。

真实候选生成和内容整合调用用户配置的 OpenAI 兼容 Agent 服务。将 `.env.example` 复制为本地 `.env`，填写 `AGENT_API_KEY`、`AGENT_BASE_URL`、`AGENT_MODEL`，然后按需运行 `python -m backend.app.skills smoke --llm`。服务调用可能产生费用。没有配置或服务不可用时明确报错，不会切换为演示模型。

```python
from backend.app.skills import SkillManager

pool = SkillManager(root="storage/skill_pool")
try:
    candidate = pool.import_candidate({
        "name": "repair-index",
        "description": "Repair stale indexes while preserving source files",
        "body": "## Applicability\nStale index.\n## Steps\n1. Inspect.\n2. Rebuild.\n3. Verify.\n## Constraints\nPreserve sources."
    })
    result = pool.integrate(candidate["id"])
    print(result)
finally:
    pool.close()
```

`integrate` 会使用真实 BGE-M3 与 Agent 模型。默认数据根目录为 `storage/skill_pool/`，自动经验积累关闭；本模块没有接入聊天、业务数据库或后台任务。

## 文档与测试

- [完整 API、CLI、默认配置及恢复说明](docs/skill-pool.md)
- [Skill 自进化框架 v1.1 PDF](docs/Skill_Self_Evolution_Framework_v1.1.pdf)
- [源码来源与发布范围](docs/SOURCE_PROVENANCE.md)

```shell
python -m pip install -r requirements-dev.txt
python -m pytest
```

工程测试使用临时目录和显式注入的模型决策，验证代码正确性，不代表真实 Skill 能力效果。开发环境已验证 95 项测试及真实 BGE-M3/Chroma 检索；真实 LLM 验证曾遇到连接错误，尚未完成真实模型端到端验证。Chroma 0.5.23 与当前 Pydantic 组合有弃用警告。

当前为 MVP：没有 A/B 效果评测、多层递归或模型训练；操作日志保存状态快照，历史文件与向量尚无自动清理。
