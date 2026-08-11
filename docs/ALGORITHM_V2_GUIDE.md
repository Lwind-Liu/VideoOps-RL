# VideoOps-RL Algorithm v2：从流程 Demo 到 Agentic RL 工程

这份文档解释当前版本真正的算法内容、RL 学什么、数据怎样进入环境，以及哪些结果已经验证、哪些必须等服务器训练后才能写进简历。

## 1. 任务定义

输入自然语言查询和一段长视频，Agent 需要在有限工具预算内定位一个或多个相关时间段，并留下可审计证据。它不是单次分类，而是一个带部分观测的序列决策问题：

```text
query
  -> 选择检索模态
  -> 构造候选证据
  -> 核验候选或扩展时间上下文
  -> 判断证据是否充分
  -> 继续搜索或停止提交
```

状态包括查询、已检索候选、时序证据图、已核验节点、剩余预算和审计状态；动作是 6 类工具调用；观测是候选分数、时间位置、字幕、图像或特征证据；终止发生在 `submit` 或预算耗尽。Ground truth 只存在于环境评分端，不进入模型 prompt。

## 2. 两套互补数据

### 2.1 公开规模数据：QVHighlights

项目接入 QVHighlights 官方人工查询和时间标注：train 7,218、val 1,550、test 1,542，共 10,310 个任务、10,148 个视频，三个 split 的视频交集均为 0。每个视频使用公开的 2 秒 OpenAI CLIP 特征；原始视频不随项目重新分发。

这套数据解决“任务太少、标签真实性不足”的问题。它用于规模化 SFT/GRPO prompt、留出集时间定位评测和时序算法消融。许可证和来源边界见 `data/external/qvhighlights/qvhighlights_manifest_v1.json`。

### 2.2 原始多模态数据：三部开放电影

3 部 Blender CC BY 影片提供 306 个真实关键帧、178 条字幕和 66 个业务式查询。它保留完整的 `search_transcript -> search_visual -> inspect image -> audit` 链路，用于验证 VLM 是否真的接收到图像，以及字幕和视觉证据如何融合。

公开 benchmark 负责规模和人工标签，开放电影负责真实图片工具调用和业务流程。训练时 GRPO 以 80%/20% 采样二者，避免项目退化成只在预提特征上跑的文本 Agent。

## 3. 检索与候选信念

### 3.1 文本和视觉信号

- 字幕检索使用 BM25，不依赖 ground truth 搜索词；
- 视觉检索使用 OpenAI CLIP ViT-B/32，306 张关键帧已经离线编码为 512 维向量；
- QVHighlights 使用官方 2 秒 CLIP 视频特征，查询向量也提前离线编码，服务器 rollout 不需要联网或重复加载编码器；
- 开放电影中的少量媒体标签只占视觉分数的 0.35，是明确标注的辅助信号，不能把这套结果冒充纯零样本泛化。

### 3.2 时序证据图

每个镜头或 2 秒 clip 是一个节点，相邻时间单元由隐式时序边连接。节点保存文本、视觉和上下文三种证据。融合不是简单相加，而是 Noisy-OR：

```text
b_i = 1 - Π_m (1 - w_m s_i^m)
```

一个模态的低分不会抹掉另一个模态的强证据，多模态同时支持时，置信度会继续上升。扩展上下文时，中心节点的置信度按 `exp(-0.7 * distance)` 向相邻节点传播。该图同时服务候选排序、审计和过程奖励。

## 4. 自适应时间段解码

只返回相似度最高的 2 秒 clip 会严重低估长事件，固定窗口又无法适应 4 秒和 60 秒事件。v2 增加变量长度解码器：

1. 对 clip 相似度做 5 点移动平均，抑制孤立峰值；
2. 阈值取 72% 分位数与峰值 0.52 倍中的较大值；
3. 把超过阈值的连续区域变成 connected component；
4. 左右各扩展 2 个 clip，恢复被平滑截掉的边界；
5. 用 `0.55 * mean + 0.45 * max` 排序，生成变量长度 proposal；
6. Agent 核验 proposal anchor，再通过 `expand_context(radius=0)` 接受整个自适应区间。

这是当前训练前最明确的算法增益。在完整 QVHighlights 留出集上：

| 方法 | val mIoU | val R@1@0.5 | test mIoU | test R@1@0.5 |
|---|---:|---:|---:|---:|
| 最高分 2 秒 clip | 0.054 | 0.0% | 0.052 | 0.0% |
| 固定 10 秒窗口 | 0.235 | 10.8% | 0.227 | 10.0% |
| 自适应时序 proposal | **0.482** | **49.9%** | **0.487** | **50.1%** |

这里报告的是项目实现的单预测 R@1-style temporal IoU，不冒充 QVHighlights 官方 mAP 脚本。完整机器可读结果在 `outputs/reports/algorithm_v2_qvhighlights_eval.json`。

## 5. 多 Agent 不是角色包装

- `QueryRouter`：仅根据 query 和视频是否有对白决定检索模态与预算；
- `TimelineScout`：只允许字幕检索和上下文扩展；
- `VisionAnalyst`：只允许视觉检索和候选核验；
- `EvidenceAuditor`：只允许检查证据一致性，不能提交；
- `Coordinator`：读取共享证据图，决定核验哪些候选以及何时停止。

环境执行硬约束：未被检索返回的候选不能核验，未核验的 anchor 不能扩展，未接地的区间不能提交。这样，模型不能通过“编一个看起来合理的 shot ID”绕开奖励。

## 6. Reward 的算法结构

v2 同时使用过程奖励和终局奖励。

过程奖励覆盖：正确候选 reciprocal-rank 的提升、核验正确/错误候选、扩展后 relevant clip recall 的变化、正确/误报审计、非法动作和非法提交。它解决长工具链只在最后得到一个标量、信用分配太稀疏的问题。

终局奖励为：

```text
R_terminal = 2.00 * temporal_IoU
           + 0.35 * shot_set_F1
           + 0.30 * semantic_audit
           + 0.20 * modality_coverage
           + R_process
           - 0.025 * tool_calls
           - 0.06  * repeated_calls
           - 0.20  * invalid_calls
```

QVHighlights 另加最高 0.10 的人工 saliency 奖励。审计分数只依据候选信念、模态覆盖和时间连续性；是否真正命中标签只用于隐藏的训练 reward，不回传为观察。所有权重集中在 `configs/algorithm_v2.yaml`，便于做消融和改动审计。

## 7. SFT 到 GRPO

### SFT

规则专家运行全部任务，但只有“提交成功且审计通过”的轨迹进入 SFT，防止模型模仿失败动作。当前生成：train 3,575、val 776、test 782 条成功轨迹；训练 split 中开放电影图像轨迹被适度重复，以保留真实图像工具格式。SFT 主要学习合法 JSON、工具顺序、证据引用和停止格式。

### GRPO

GRPO train 有 9,018 条无标签泄漏采样记录，其中 7,218 条 QVHighlights、1,800 条由 40 个开放电影任务重复得到的原始图像 episode，实际比例约 80%/20%。外部 dataset 把同一个 `task_id` 传给一组 4 条 rollout，确保组内共享完全相同的初始状态，再通过相对 reward 形成优势；如果让 4 个环境各自随机抽题，组基线会混入任务难度噪声。策略学习的是：选择何种模态、搜索词、核验哪个候选、是否扩展、是否审计以及何时停止。它不是重新训练 CLIP，也不是拿 target interval 做监督回归。

直接跳过 SFT 时，大量早期输出不是合法工具调用，组内 reward 都接近失败值，GRPO 很难得到有效相对优势。因此执行顺序固定为：

```text
24 GPU LoRA SFT（8 卡机器保留兼容路径）
  -> merge LoRA 得到完整 checkpoint
  -> GPU 20-23 启动 TP=4 vLLM rollout server
  -> GPU 0-19 做 GRPO
  -> val/test 各 12 卡并行 checkpoint evaluation
```

`server/run_all.sh` 会自动在 8 卡兼容路径和 24 卡快速路径之间选择，并实现进程存活检查、vLLM health check、退出清理、确定性评测分片与结果合并。服务器训练尚未实际执行，所以不能宣称 GRPO 已提高指标。

## 8. 你应该怎样理解结果边界

已经确认：10,310 个公开任务和特征完整；无视频级 split 泄漏；3,092 个 val/test 任务全部完成算法评测；自适应 proposal 显著优于两个窗口基线；3,575 条成功 SFT 轨迹和 9,018 条 GRPO train 采样记录已生成；43 个本地测试通过。

尚未确认：SFT checkpoint 的工具合法率、GRPO 的组内 reward 方差、训练后相对 SFT 的增益、8 张 H200 的真实吞吐和峰值显存。这些必须以服务器日志和 checkpoint 报告为准。

## 9. 面试讲法

> 我把长视频查询定位建模成一个受工具预算约束的 Agentic RL 问题。底层用 BM25 和 CLIP 构建多模态候选，再用 Noisy-OR 时序证据图融合模态并传播上下文。针对单 clip 峰值无法覆盖长事件的问题，我实现了平滑、动态阈值、连通区域和 proposal ranking 的变量长度解码器，在 QVHighlights 完整 val/test 上把 R@1@0.5 从固定 10 秒窗口的约 10% 提升到约 50%。上层把检索、视觉核验、审计和协调拆成权限隔离的 Agent，并用硬接地约束防止伪造证据。训练采用成功轨迹 LoRA SFT，24 卡模式用 20 卡 GRPO、4 卡 vLLM 和 24 卡并行评测；本地已完成数据、环境、消融和离线包，训练后增益等 H200 日志产生后再报告。

## 10. 阅读和执行顺序

1. `configs/algorithm_v2.yaml`：所有算法权重；
2. `src/videoops_rl/evidence_graph.py`：候选信念和时序传播；
3. `src/videoops_rl/qv_env.py`：变量长度 proposal、约束、审计和 reward；
4. `src/videoops_rl/multivideo_env.py`：原始图像/字幕环境；
5. `src/videoops_rl/agents.py`：专业 Agent 和专家轨迹；
6. `scripts/evaluate_algorithm_v2.py`：留出集消融；
7. `scripts/build_training_data.py`：成功轨迹 SFT 与无泄漏 GRPO 数据；
8. `server/run_all.sh`：服务器唯一执行入口。
