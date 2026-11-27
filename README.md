# RSI · 独立自进化记忆

从 Paper-Agent Memory V3（`faea50b`）提取的本地单人程序性记忆。ACE 风格反思从反馈中提炼 Skill，SQLite 保存版本和待同步操作，RSI 在线归簇并从最多 3 个簇中选取最多 4 条策略给普通问答使用。可以单独测试写入、平面检索和 RSI。

## 在服务器上运行

需要 Python 3.11 或更新版本。Linux 示例：

```bash
git clone https://github.com/Zhenghao9103/RSI.git
cd RSI
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[models,test]'
cp .env.example .env
```

在 `.env` 填写 OpenAI 兼容的 `AGENT_BASE_URL`、`AGENT_API_KEY` 和 `AGENT_MODEL`。本地服务无鉴权时，可使用示例中的占位值 `local`。Reflection、Judge 和生成使用同一个 Agent 模型。BGE-M3 默认从 Hugging Face 加载，已有离线权重时将 `BGE_MODEL` 指向模型目录。若更换向量模型，先重建索引。

```bash
rsi learn '以后解释数学概念时，先举一个数字例子，再给定义。'
rsi skills
rsi ask '什么是加权平均数？'
rsi-eval --output reports/server.json
python -m pytest
```

`rsi ask` 返回回答、实际注入的 Skill ID 与版本、token 数和降级状态。评测使用临时库，依次跑原版、ACE 仅写入、ACE + Flat、ACE + RSI。两种检索共享冻结的 Skill 快照，结果 JSON 包含逐题回答、命中和耗时。可用 `--cases` 指定与内置 [样例](rsi/general_dialogue.json) 同格式的数据。内置六题用于端到端对照。

`PROCEDURAL_MEMORY_MODE` 支持 `off`、`write`、`flat`、`rsi`。默认 `off`，示例配置为 `rsi`。数据库默认在 `data/memory.db`，可用 `rsi --db path/to/memory.db` 指定。用 `rsi retry` 重试待处理反馈和索引写入，用 `rsi rebuild` 从有效 Skill 重建索引。

Python API 可使用 `with rsi.Memory("data/memory.db") as memory:`，调用 `learn(feedback, previous_answer)`、`ask(question)` 和 `skills()`。测试通过本地临时 SQLite 和注入的模型响应运行，安装 `.[test]` 即可。

## ALFWorld 实验结果

[四组配对消融、写入成本、C/D 复测与逐题数据](experiments/alfworld/README.md)
