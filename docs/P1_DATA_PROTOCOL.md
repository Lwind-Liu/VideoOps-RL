# P1：正式数据协议、容量预算与完整性审计

## 1. 这一阶段解决什么

P1 不追求扩大数据数量，而是先回答三个工程问题：后续每增加一个视频应该保存什么；训练集和评测集如何避免泄漏；最终离线包如何确保不超过 50 GiB。

当前 Tears of Steel 样本已经具备视频、字幕、98 个镜头、98 张关键帧和 10 条查询，但所有查询都来自同一部视频。原文件中的 `train/eval` 只能用于调试代码路径，不能测量跨视频泛化。因此 P1 把这 10 条任务统一登记为 `smoke`，正式数据必须按 `video_id` 划分 train/val/test。

## 2. 为什么必须先定义协议

如果没有稳定协议，后续常见问题包括：不同数据源时间单位不一致；任务引用不存在的关键帧；同一视频同时出现在训练和测试；原视频丢失后无法追踪来源；数据和模型把 50 GiB 离线包撑爆。

Agentic RL 会放大这些问题。例如测试视频泄漏会让策略看起来定位准确；错误时间戳会让 Reward 奖励错误轨迹；缺失证据会让 Auditor 无法判断结果是否可靠。因此数据门禁属于训练系统的一部分，不是整理文件的附属工作。

## 3. 两层 Schema

### 3.1 Dataset Manifest

`data/registry/p1_dataset_manifest.json` 描述整个数据版本：

- `dataset_id`：不可混淆的数据版本名；
- `scope`：当前数据的用途和可以声称的结论；
- `capacity`：45 GiB 目标线和 50 GiB 硬上限；
- `videos`：视频 ID、时长、许可证和核心产物哈希；
- `tasks`：任务文件、数量与 Schema；
- `split_policy`：按视频划分和 smoke 排除规则。

Manifest 不存储所有任务内容，只负责索引、约束和追踪。

### 3.2 Task Record

`data/registry/p1_smoke_tasks.jsonl` 每行是一条任务，核心字段为：

```json
{
  "schema_version": "videoops.task.v1",
  "task_id": "tos_hl_008",
  "video_id": "tears_of_steel_s1_20s_500s",
  "split": "smoke",
  "query": "Find the failed take...",
  "target_segments": [{"start_ms": 351500, "end_ms": 365000}],
  "evidence": {
    "search_terms": ["Abort", "Cut", "Nooooo"],
    "shot_ids": ["shot_0065", "shot_0066"],
    "required_modalities": ["text", "image"]
  }
}
```

所有时间统一为毫秒，区间统一为半开区间 `[start_ms, end_ms)`。`shot_ids` 不是模型答案，而是 Reward 和 EvidenceAuditor 用来核对的标准证据。

JSON Schema 位于 `schemas/`。Python 审计器另外检查跨记录关系，因为“同一视频不能跨 split”无法只靠单条 JSON Schema 表达。

## 4. 数据划分原则

正式规则是：一个 `video_id` 只能属于 train、val、test 中的一个。即使两条查询完全不同，只要来源视频相同，也不能跨 split。

原因是关键帧、人物、场景和字幕风格会泄漏。按查询随机划分可能让模型在训练时见过同一场景，测试时只是换一个问题，得到虚高结果。

`smoke` 是独立角色：用于检查代码、工具和 Reward，不进入正式指标，也不与 benchmark 结果合并。

## 5. 50 GiB 离线包预算

审计器使用二进制 GiB：`1 GiB = 1024^3 bytes`。硬上限是 50 GiB，内部目标线是 45 GiB，预留 5 GiB 给日志、LoRA adapter 和运行中间文件。

| 类别 | 目标 |
|---|---:|
| Qwen3-VL 基础权重 | 10--20 GiB |
| 精简视频/关键帧/字幕 | 10--15 GiB |
| Python wheelhouse 或容器 | 5--8 GiB |
| 特征和检索索引 | 2--4 GiB |
| 代码、配置、测试 | 小于 2 GiB |
| 预留 | 至少 5 GiB |

`package_inventory()` 会统计项目中除 `.git`、`tmp` 和 `__pycache__` 外的所有文件。服务器需要的内容不能放在这些排除目录中。

## 6. 自动审计做了什么

```powershell
python scripts/build_p1_dataset_registry.py
python scripts/audit_offline_package.py
```

当前门禁包括：

1. 离线包是否低于 50 GiB；
2. Manifest 中的路径是否仍在项目目录内；
3. 核心文件是否存在且大小一致；
4. SHA-256 是否与登记值一致；
5. Manifest 是否通过 JSON Schema；
6. 每条 Task 是否通过 JSON Schema；
7. task ID 是否重复；
8. query 是否为空；
9. 目标区间是否满足 `0 <= start < end <= duration`；
10. 是否具有镜头证据；
11. 同一视频是否跨正式 split；
12. 任务实际数量是否等于 Manifest 声明。

测试还主动构造路径逃逸、视频级数据泄漏、目标越界和临时文件误计入容量等失败案例。只有这些错误确实会被拒绝，才能说明门禁有效。

## 7. 当前真实结果与边界

P1 当前登记 1 部 480 秒开放电影、10 条 smoke 任务及对应文本和图像证据。它证明正式 Schema、哈希、容量审计和视频级划分规则可执行，但不证明跨视频泛化。

后续加入 50--100 个视频时，不修改协议，只新增视频记录和任务记录。正式评测报告必须排除 `smoke`，并注明 train/val/test 的视频数、查询数、总时长和失败文件数。

## 8. 面试时如何解释

> 我没有直接把公开视频堆进训练目录，而是先定义了版本化的数据协议。每条任务绑定视频 ID、毫秒时间段和多模态证据；数据按视频而不是按问题划分，防止同场景泄漏；所有核心文件记录大小和 SHA-256；离线包还有 50 GiB 的自动容量门禁。这样本机 smoke、服务器训练和最终评测使用的是同一套可追踪数据语义。

一个重要的独立判断是：当前 10 条查询即使写了 train/eval，也因为来自同一部视频而不能作为泛化实验。把它降级为 smoke，比保留一个好看的数字更符合工程事实。

## 9. 阅读后应能回答

1. 为什么任务不能随机按 query 划分？
2. `smoke` 与 `test` 的区别是什么？
3. 为什么 JSON Schema 之外还需要 Python 跨记录审计？
4. `shot_ids` 在训练中起什么作用？
5. 为什么容量目标是 45 GiB，而硬上限是 50 GiB？
6. 当前 P1 已经证明了什么，尚未证明什么？

你确认这些问题能够讲清楚后，再进入 P2：多模态检索工具与单 Agent baseline。
