# Self-Evolving Skill Pool

本模块提供独立的 Python 调用接口与命令行入口，实现候选生成、在线聚类、Master/Reference 整合和局部结构演化。它没有 HTTP 服务、后台任务或聊天/Agent 钩子；只有显式调用才运行。SQLite、Skill 文件、版本和 Chroma 都使用独立数据目录，默认是仓库的 `storage/skill_pool/`。

生产运行复用项目的 BGE-M3 编码器、Chroma 客户端工厂和 `agent_json` 配置。依赖这些组件的环境和接口；执行编码会占用内存和算力，调用 LLM 会使用现有模型服务并可能产生费用。离线演示通过明确的依赖注入使用确定性决策和向量，生产默认实现不会自动切换到演示模型。

## 快速运行

在仓库根目录执行以下命令。项目 Python 为 `D:\envs_dirs\paper-agent\python.exe`，所需依赖见 `backend/requirements.txt`。

```powershell
# 查看命令；不会加载模型或创建数据目录
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --help

# 创建独立目录并查看状态；不进行编码或 LLM 调用
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' init

# 离线完整演示：真实 SQLite、文件和 Chroma；临时目录运行后清理
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills demo

# 保存演示结果时，--root 必须指向空目录
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-demo' demo

# 真实 BGE-M3 + 临时 Chroma 检索闭环；可能需要下载/加载模型
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills smoke

# 额外调用当前 Agent LLM 生成、整合候选，验证发现和按需加载
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills smoke --llm
```

`demo` 验证相似候选入同簇、第三个候选更新 Reference、晋升迁移成员、拆分保留历史向量并重算中心、重启后发现一致。演示是工程流程验证，不证明真实模型判断质量或任务效果提升。`smoke` 独立报告真实模型运行；失败时退出码为 1，不报告成功。临时目录模式在子进程运行，退出后由父进程清理目录，确保 Windows 下 Chroma 文件句柄释放；显式指定的演示目录保留数据。

## 候选与操作流程

候选支持 YAML frontmatter 的 `SKILL.md`，或具有 `name`、`description`、`body` 的 JSON 对象。名称使用小写字母、数字及单个连字符；模型生成的正文必须包含适用场景、操作步骤和约束，手动导入则检查基础字段。候选先保存为待整合项，不直接进入活跃 Skill 发现。

```markdown
---
name: repair-index
description: Repair a stale document index while preserving source files
---
## Applicability
Use when indexed document IDs or versions are stale.
## Steps
1. Inspect source and indexed document versions.
2. Rebuild only the affected index entries.
3. Verify IDs and versions match.
## Constraints
Preserve original source files and supplied evidence.
```

Trace JSON 必须包含 `task_id`、`user_query`、非空 `trace_events`、`final_answer` 和 `completion_status`。例如：

```json
{
  "task_id": "repair-1",
  "user_query": "Save the repaired indexing workflow as a reusable skill",
  "trace_events": [
    {"tool": "inspect_index", "status": "error", "error": "Stale index"},
    {"tool": "rebuild_index", "args": {"preserve_sources": true}, "status": "success"},
    {"tool": "verify_index", "status": "success", "result": "IDs and versions match"}
  ],
  "final_answer": "Rebuilt and verified the index while preserving source files.",
  "completion_status": "success"
}
```

```powershell
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' import '.\SKILL.md'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' observe '.\trace.json'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' generate '.\trace.json'
# 将下列占位符替换为上一条命令返回的真实 ID
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' integrate '<candidate_id>'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' discover 'repair stale document indexes'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' load '<skill_id>'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' load '<reference_id>' --reference
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' record-usage '<reference_id>' 'task-2' --independent
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' maintain '<cluster_id>'
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' maintain-due
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' history
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' recover
& 'D:\envs_dirs\paper-agent\python.exe' -m backend.app.skills --root '.tmp/skill-pool' rollback '<operation_id>'
```

`observe` 检测显式保存、修复经验、不同任务的重复策略和未覆盖任务类型，并记录 Trace。默认不自动生成候选。`generate` 是显式生成；LLM 或校验失败会向调用方报告。`integrate` 召回相似簇，由策略判断职责关系，依次选择忽略、更新 Reference、新增 Reference 或新建簇。

只有 `name + description` 参与编码，历史成员的归一化向量用于簇中心；当前 Master/Reference 向量用于发现。忽略和完全重复的候选不会增加成员权重。发现只返回概述，正文使用 `load` 单独读取。

`maintain` 只处理指定簇，支持 `--action PROMOTE|SPLIT|MERGE|REFACTOR|ARCHIVE|KEEP`。指定操作仍需策略判断和结构验证，不能跳过晋升的独立任务边界/完整工作流判断。晋升将原 Reference 替换为协作指引；拆分必须完整分配 Reference 与成员；晋升、拆分后的冷却期阻止反向合并。使用信号本身不会自动执行维护。

`maintain-due` 是显式维护入口，按事件和周期信号检查受影响簇，跳过仍在冷却期的簇。独立需求、Reference 更新及簇内向量发散会产生审查信号；发散阈值只是启发式，实际 SPLIT 仍由 LLM 判断。KEEP 也会记录本次审查并消费已处理信号；成员集合变化后才重新评估同簇发散。`status` 返回候选 ID、Skill/Reference 的 ID 和摘要、簇信息及待恢复操作，供后续命令使用。

## Python API

```python
from backend.app.skills import SkillManager, SkillPoolSettings

pool = SkillManager(root=r".tmp/skill-pool")
try:
    candidate = pool.import_candidate({
        "name": "repair-index",
        "description": "Repair stale indexes while preserving source files",
        "body": "## Applicability\nStale index.\n## Steps\n1. Inspect.\n2. Rebuild.\n3. Verify.\n## Constraints\nPreserve source files."
    }, source={"type": "manual"})
    integrated = pool.integrate(candidate["id"])
    hits = pool.discover("repair stale document indexes")
    if hits:
        master = pool.load_master(hits[0]["id"])
    pool.record_usage(integrated["reference_id"], "task-2", independent=True)
    status = pool.status()
finally:
    pool.close()
```

还提供 `generate_candidate(trace)`、`observe(trace)`、`maintain(cluster_id, action=None)`、`maintain_due()`、`discover_references(skill_id, query, limit=None)`、`load_reference(reference_id)`、`state()`、`history()`、`rollback(operation_id)` 和 `recover()`。可通过 `settings` 参数传入 `SkillPoolSettings`；测试/离线演示可显式注入 `embed`、`llm`、`policy` 或 `index`。导入包不会启动服务，编码器和模型调用按需使用。

## 配置与恢复

使用独立 `SKILL_POOL_` 环境变量；不修改应用 Settings，不自动读取应用 `.env`。

| 环境变量 | 默认值 |
|---|---|
| `SKILL_POOL_ROOT` | 仓库 `storage/skill_pool/` |
| `SKILL_POOL_AUTO_ACCUMULATION` | `false` |
| `SKILL_POOL_CLUSTER_THRESHOLD` | `0.80` |
| `SKILL_POOL_GENERATION_MIN_CONFIDENCE` | `0.80` |
| `SKILL_POOL_CLUSTER_LIMIT` / `SKILL_POOL_REFERENCE_LIMIT` | `3` / `3` |
| `SKILL_POOL_REPEAT_THRESHOLD` | `3` 个不同任务 |
| `SKILL_POOL_INDEPENDENT_HIT_THRESHOLD` | `3` |
| `SKILL_POOL_REFERENCE_UPDATE_THRESHOLD` | `3` |
| `SKILL_POOL_MAINTENANCE_INTERVAL` | `10` 次内容变更 |
| `SKILL_POOL_COOLDOWN_HOURS` | `24` |
| `SKILL_POOL_DISCOVERY_THRESHOLD` / `SKILL_POOL_DISCOVERY_LIMIT` | `0.65` / `3` |
| `SKILL_POOL_DIVERGENCE_SIMILARITY` | `0.75`，发散审查启发式 |
| `SKILL_POOL_LLM_MAX_TOKENS` | `4096` |

阈值是初始工程参数，不是实验得出的最优值。启用 `AUTO_ACCUMULATION` 也只处理显式提交的 Trace，不采集聊天或启动后台任务。

SQLite 保存实体、成员、来源和操作日志，`candidates/<id>/SKILL.md` 保存候选原文，`versions/` 保存不可变文件版本，`skills/` 是当前 Skill 文件的物化副本，独立 Chroma 保存版本化向量。Loader 依据已发布的 SQLite 状态读取快照，并排除未发布或过期向量。写入按数据根目录串行化；索引或文件准备失败后操作保持 pending，后续写入需先 `recover`。恢复按同一操作重试，不添加重复历史成员。Chroma 的 cosine 和旧版本默认 squared L2 距离均转换为一致的相似度。

回滚只允许最新可回滚操作；若有后续依赖，需按逆序回滚。回滚恢复结构、成员、中心、文件和发现状态。使用 `history` 和 `status` 查看操作及待恢复项；smoke 失败报告安全的执行阶段、错误类型和类别，其他外部模型异常只显示错误类型，不回显服务日志、请求正文或凭据。原生日志即使混用 GBK/UTF-8 也不会阻断父进程读取结果。不要手动修改 `versions/`；当前物化副本的手动修改也不会自动进入模块历史。

首版操作日志保存变更前后的完整状态快照；候选、历史文件版本及索引向量暂不自动清理。池增大时数据库、磁盘和状态读写成本会持续增长，适用于当前独立 MVP，尚未验证大规模性能。

## 文件与调用关系

| 文件 | 职责 |
|---|---|
| `backend/app/skills/__init__.py`、`__main__.py` | 延迟导出 Python API、CLI 命令分发 |
| `config.py`、`locking.py` | 独立配置、同目录串行写入 |
| `manager.py`、`policy.py`、`clustering.py` | 流程编排、Trigger/Generator/LLM 决策、在线聚类 |
| `evolution.py` | 局部晋升、拆分、合并、整理和归档 |
| `repository.py`、`files.py`、`index.py` | 独立 SQLite 日志、文件版本、BGE-M3/Chroma 检索 |
| `demo.py` | 离线演示、隔离子进程、真实依赖 smoke |
| `tests/skill_pool/test_manager.py`、`test_policy.py` | 整合、生成、触发和编排验收 |
| `test_evolution.py`、`test_files_local.py`、`test_storage.py` | 结构演化、候选/文件与存储恢复 |
| `test_cli.py` | 命令行、演示、临时清理和日志编码回归 |
| `docs/skill-pool.md` | 使用与验证说明 |
| `.gitignore` | 忽略 Skill 运行数据 |
| `backend/requirements.txt`、`backend/pyproject.toml` | 显式声明 PyYAML 依赖 |

生产调用链：`Python API / CLI → SkillManager → Trigger / Policy / Cluster / Evolution → Repository / FileStore / SkillIndex`。Index 按需使用现有 BGE-M3 和 Chroma 工厂，Policy 按需调用 `agent_json`；发布成功后 Loader 根据 SQLite 已发布版本读取。演示通过显式注入替换模型判断，存储链保持真实。

## 验证

```powershell
& 'D:\envs_dirs\paper-agent\python.exe' -m pytest tests/skill_pool -q
```

测试使用临时独立目录，覆盖聚类边界、内容整合、晋升/拆分/合并/归档、历史向量保留、冷却期、文件路径、真实 Chroma、发布失败恢复、完整回滚和 CLI。离线测试不依赖远程 LLM；真实 smoke 单独运行。首版没有接入现有聊天或研究流程，也没有 A/B Benchmark、多层递归和模型训练。

2026-10-11 本地验证结果：

- Skill Pool 测试：95 项通过；现有配置、模型客户端及向量存储回归：55 项通过。
- Ruff 检查和格式检查通过。Chroma 0.5.23 与当前 Pydantic 组合仍有已知弃用警告。
- 离线演示 8 项验收通过，覆盖同簇整合、Reference 更新、晋升、拆分、归档排除及重启一致性；使用真实 SQLite、文件和 Chroma，模型决策为明确注入的演示实现。
- 真实 BGE-M3/Chroma smoke 通过：1024 维，向量范数约 1，检索闭环成功，临时数据清理成功。
- 真实 LLM 生成尝试返回 `ModelClientError / connection_error`，尚未完成真实 LLM 生成与整合验证。实现保留明确错误，未使用演示策略替代生产模型。服务恢复后可显式运行 `smoke --llm` 验证完整闭环。
