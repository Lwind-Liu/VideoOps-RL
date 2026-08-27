# VideoOps-RL 业务等价复现与训练说明

## 1. 项目定位和事实边界

本项目从实习期间参与的长视频理解与智能媒资 Pipeline 中抽取“查询式高光定位”作为可公开复现的 Agentic RL 任务。

三类内容必须分开表述：

1. **实习业务中确认存在的链路**：长视频切分、字幕/说话人/人物/关键帧感知、时间对齐剧情理解、高光、内容解释、分龄、画质增强与媒资运营。
2. **本仓库的等价实现**：使用公开影片、QVHighlights、SRT、OpenAI CLIP 特征、关键帧和本地规则审计，复现工具协议、状态转移、训练和评测闭环。
3. **需要服务器日志确认的结果**：SFT/GRPO loss、reward、工具调用数、成功率和训练稳定性。复现结果不替代真实业务指标。

不能说“复现了公司内部工具”。准确说法是：

> 真实业务中工具来自内部媒资系统和模型服务；离职后无法继续访问，所以公开复现将内部能力抽象为统一 Tool Gateway，并用开放数据和本地后端平替，保持输入输出、证据权限、状态转移、成本和奖励计算方式可对照。

## 2. 为什么这里需要 Agentic RL

如果模型只接收整段视频并一次输出时间区间，这只是多模态定位，不构成 Agentic RL。本项目要求模型在最多 10 个工具步骤内完成连续决策：

```text
读查询和预算
  -> 选择字幕检索或视觉检索
  -> 比较候选片段
  -> 检查关键帧
  -> 判断是否扩展上下文
  -> 请求证据审计
  -> 继续补证据或提交
```

Agentic RL 的必要性来自四点：

- 每个查询需要的工具和观察深度不同；
- 工具调用会改变后续可见状态，不能近似成一次静态回答；
- 证据质量、时间边界和调用成本存在冲突；
- 模型必须学习停止，继续观察和过早提交都会付出代价。

第一轮只训练 Coordinator 的工具选择和停止策略。TimelineScout、VisionAnalyst、EvidenceAuditor 是权限分离的功能角色，不是四套独立大模型。

## 3. 从真实影片到训练环境

### 3.1 媒体预处理

```text
MP4/MKV
  -> FFmpeg 解复用和基础审计
  -> 镜头边界检测
  -> 每个镜头抽关键帧
  -> SRT/ASR 解析为带时间戳台词
  -> 可选 speaker / 人物 / 场景 / 事件标签
  -> 全部投影到统一毫秒时间轴
```

基本单位不是固定 30 秒窗口，而是 `evidence unit`：

```json
{
  "shot_id": "shot_0015",
  "start_ms": 73125,
  "end_ms": 76667,
  "keyframe_path": "keyframes/shot_0015.jpg",
  "transcript": "...",
  "speaker": "spk_02",
  "visual_tags": ["forest", "running"]
}
```

公开影片使用镜头级单元；QVHighlights 使用官方 2 秒 CLIP feature 单元。后者没有公开原视频和字幕，因此只用于规模化视觉检索与时间定位训练，不能声称在该子集上真实调用了字幕和关键帧图像服务。

### 3.2 时序证据图

字幕 BM25、CLIP 相似度、媒体元数据和相邻时间上下文分别更新证据节点。Noisy-OR 融合避免简单相加造成分数失控；时间邻接传播让高分证据扩散到相邻镜头，但按距离衰减。

每个候选保留：时间范围、证据来源、检索分数、是否检查、审计结果和调用轨迹。模型只看公开观测，环境内部标签只用于 reward。

## 4. 公司内部工具的公开平替

| 业务能力 | 本地工具 | 公开后端 | 主要输入 | 主要输出 |
|---|---|---|---|---|
| 字幕/ASR 检索 | `search_transcript` | SRT + BM25 | query, top_k | shot、台词、时间和分数 |
| 视觉向量检索 | `search_visual` | OpenAI CLIP/QV 官方特征 | query, top_k | 候选 shot 或自适应 proposal |
| 关键帧/VLM 分析 | `inspect_keyframe` | JPEG + Qwen3-VL；QV 为 feature proxy | shot_id | 图像、台词、节点证据 |
| 时间轴上下文服务 | `expand_context` | evidence graph | shot_id, radius | 相邻镜头或完整 proposal |
| 证据一致性审计 | `request_audit` | 规则 verifier | shot_ids | 语义、模态、连续性分数 |
| 结果提交与验收 | `submit` | task verifier | shot_ids | 时间段、IoU、reward 分项 |

所有调用经过 `ToolGateway`，统一返回：

```json
{
  "request_id": "task:episode:03",
  "owner": "VisionAnalyst",
  "service": "keyframe-inspection",
  "backend": "local-keyframe-store",
  "status": "ok",
  "latency_ms": 31.2,
  "cost_units": 4.0,
  "state_before": "...",
  "state_after": "...",
  "observation": {}
}
```

训练默认关闭随机故障，保证同组 rollout 从可比环境出发。可在鲁棒性评测中设置 `VIDEOOPS_TOOL_FAULT_RATE`，模拟超时或服务不可用。

## 5. Agent、权限和状态

### Coordinator

输入是查询、候选、已观察证据、审计反馈和剩余预算；输出是下一次工具调用或 `submit`。这是 SFT/GRPO 真正更新的策略模型。

### TimelineScout

只负责字幕/视觉检索和时间上下文扩展。它不能提交答案，也不能访问标签。

### VisionAnalyst

只检查检索返回的关键帧。未检索的 shot 无法直接检查，防止模型枚举 ID 或绕过检索。

### EvidenceAuditor

只对已检查或已扩展的候选做语义一致性、模态覆盖和时间连续性审计。它不能替 Coordinator 选择最终答案。

### 环境状态

```text
query / video_id
retrieved candidates
inspected shots
expanded context
searched modalities
audit score and audit scope
remaining tool budget
invalid and repeated calls
done flag
```

有效工具集合随状态变化。例如搜索前不能检查，检查前不能扩展，未接地候选不能提交。这使任务成为有约束的多轮决策过程，而不是自由文本 CoT。

## 6. SFT 训练什么

SFT 不负责让模型记住高光答案，主要训练三件事：

1. 输出合法 native tool call 和正确参数；
2. 学会基本顺序：检索、检查、扩展/审计、提交；
3. 根据工具观测继续决策，而不是忽略环境反馈。

Teacher 是无标签泄漏的规则多 Agent。只有成功且通过审计的轨迹进入主 SFT 集；正式影片轨迹适度过采样，以保证模型见过真实图像工具结果。数据按视频级拆分，避免同一影片的画风、人物和台词同时进入 train/test。

SFT 需要观察：

- train/eval loss；
- 工具 JSON/原生 function call 合法率；
- tool name 和参数 schema 准确率；
- 第一次合法调用所需 token 数；
- 轨迹是否能到达 `submit`。

## 7. GRPO 训练什么

同一个 prompt 采样 4 条轨迹。每条轨迹独立调用工具并得到总 reward，再用组内相对优势更新 Coordinator。训练的不是 CLIP、BM25 或审计器参数，而是 Qwen3-VL-2B 的工具选择、证据补充、边界扩展和停止策略。

Reward 由终局质量、过程增益和成本共同组成：

```text
终局：temporal IoU + shot F1 + audit quality + modality coverage
过程：检索排名增益 + 正确检查 + 上下文召回增益 + 正确审计
成本：tool cost + repeated call + invalid call + invalid submission
```

中间 reward 会进入整条轨迹总收益，用于缓解纯终局奖励的粗粒度 credit；当前仍是 trajectory-level GRPO，不冒充真正的 turn-level advantage 算法。

为了避免 GRPO 没有学习信号，需要同时检查：

- 组内 reward 标准差和 zero-advantage group 比例；
- submit rate、任务成功率和平均工具调用数；
- reward 各分量是否被某一项支配；
- 非法调用、重复调用和审计通过率；
- KL、clip fraction、entropy、gradient norm；
- 正确轨迹的 log probability 是否先升后崩。

`server/analyze_training_run.py` 汇总 Trainer 日志和完整工具 trace。若 Smoke 中没有提交或 reward 完全无方差，门禁会阻止自动进入 Full。

## 8. 24 张 H200 的运行划分

当前实现沿用易复现的两阶段路径：

```text
阶段 1：24 卡 ZeRO-2 LoRA SFT
阶段 2：20 卡 GRPO learner + 4 卡 vLLM rollout server
阶段 3：24 卡并行切分 val/test 评测
```

这是为了缩短多轨迹 rollout、训练和评测墙钟时间，不是因为 2B 模型必须使用 24 张 H200。Smoke 默认 0.1 epoch SFT、20 step GRPO 和 48 个评测任务；通过信号门禁后，Full 默认 3 epoch、200 step 和 300 个评测任务。

## 9. 本地准备、服务器运行和回传

服务器没有模型权重时，推荐先在 Windows 本地下载离线分片：

```powershell
cd VideoOps-RL
powershell -ExecutionPolicy Bypass -File scripts/prepare_offline_assets.ps1
```

将整个仓库连同 `offline_assets/` 上传服务器。Bootstrap 会优先校验并使用上传分片，缺失时才访问 GitHub Release。

```bash
cd VideoOps-RL
RUN_MODE=smoke bash bootstrap_server.sh
```

检查 `outputs/reports/training_signal_smoke_*.json` 后再运行 Full：

```bash
RUN_MODE=full bash bootstrap_server.sh
bash server/collect_run_bundle.sh
```

回传包包含环境信息、依赖、SFT/GRPO metrics、评测报告、训练信号报告、压缩工具 trace、checkpoint 清单和 SHA-256。Checkpoint 本体单独保留路径。

## 10. 面试中最稳妥的一句话

> 我把实习中的长视频媒资理解链路抽成查询式高光 Agent：底层用字幕、CLIP、关键帧和时间证据图模拟内部媒资工具，上层由 Qwen3-VL Coordinator 在预算内动态选择检索、核验、扩展、审计和停止动作；先用成功工具轨迹做 SFT，再用包含定位质量、证据质量、过程增益和调用成本的 GRPO 优化整条决策轨迹。公开复现保留了工具协议、状态转移和训练闭环，但不把本地指标冒充公司业务结果。
