# VideoOps-RL 项目规格

## 1. 项目问题

长视频媒资系统需要同时处理剧情理解、时间定位、内容生成、质量增强和安全约束。传统固定 Pipeline 对所有视频和片段执行相同流程，容易产生三类浪费：

1. 对低信息片段重复调用昂贵模型；
2. 在证据已经充分时继续观察；
3. 在片源质量、业务任务和风险等级变化时无法动态调整处理策略。

VideoOps-RL 的研究问题是：

> 多 Agent 能否围绕共享的时间对齐剧情证据图，通过强化学习动态选择工具、模型、片段和停止时机，在保持任务质量的同时降低处理成本？

## 2. 核心数据产品：剧情证据图

所有业务任务依赖同一份结构化数据，而不是各自重复理解视频。

### 节点

- `shot`：镜头及起止时间；
- `utterance`：台词、speaker 和时间范围；
- `character`：人物原型、别名和出现轨迹；
- `scene`：地点、环境、氛围和时间范围；
- `event`：参与者、动作、原因、结果和叙事作用；
- `risk_segment`：风险类型、等级和多模态证据；
- `quality_segment`：模糊、压缩、噪声、时序和音画状态。

### 边

- `appears_in`、`speaks_in`；
- `causes`、`precedes`、`overlaps`；
- `supported_by`；
- `first_visible_at`；
- `enhanced_from`。

每个语义结论必须保留时间范围、来源片段、证据 ID、模型版本、prompt 版本和置信度。

## 3. 系统分层

```text
输入层
  视频、字幕、业务任务、用户观看进度、处理预算

感知工具层
  解复用、镜头检测、ASR、speaker、人物、选帧、质量诊断

证据图层
  人物、台词、场景、事件、风险、质量及其时间关系

Agent 层
  Coordinator、TimelineScout、StoryAnalyst、EvidenceAuditor、MediaQualityAgent

业务任务层
  查询式高光、无剧透解释、分龄引导、增强路由、海报候选

验收层
  时间 IoU、任务正确率、证据支持率、安全违规、成本、时延和质量增益
```

## 4. Agent 与工具边界

### Coordinator

只负责策略，不直接分析所有模态。输入为任务状态、已有证据、冲突、不确定性和预算；输出为下一动作。

### TimelineScout

负责从长时间轴中缩小范围，可调用：

- `search_transcript(query)`；
- `retrieve_by_embedding(query)`；
- `inspect_shot_boundaries(range)`；
- `sample_keyframes(range, budget)`；
- `expand_context(segment, radius)`。

### StoryAnalyst

负责语义抽取，可调用 VLM/LLM，将候选片段转为严格 Schema 的人物、场景、动作和事件。

### EvidenceAuditor

检查事件是否存在画面或台词支撑、人物是否一致、时间是否越界、多个 Agent 是否冲突，并决定通过、补证据或拒答。

### MediaQualityAgent

调用质量诊断、帧筛选和增强工具，决定原片直接理解、只修复关键帧、片段 1× 修复、片段 2× 超分或不值得增强。

## 5. RL 定义

### 状态

```text
业务任务与查询
候选时间范围
已观察/未观察片段
字幕与视觉证据
事件图当前状态
证据冲突与不确定性
片源质量
剩余工具预算、视觉 token 和时延预算
```

### 动作

```json
{"type": "search_transcript", "query": "发现异常"}
{"type": "sample_keyframes", "start_ms": 120000, "end_ms": 180000, "budget": 8}
{"type": "inspect_segment", "segment_id": "seg_03"}
{"type": "expand_context", "segment_id": "seg_03", "radius_ms": 10000}
{"type": "ask_agent", "agent": "evidence_auditor"}
{"type": "enhance_segment", "segment_id": "seg_03", "mode": "repair_1x"}
{"type": "submit", "segment_ids": ["seg_03"], "evidence_ids": ["f_12", "u_08"]}
```

### 奖励

总体奖励不写死为一个不可解释的 judge 分数：

```text
R = R_task + R_grounding + R_evidence + R_quality
    - C_tool - C_latency - P_hallucination - P_safety
```

- `R_task`：业务任务是否完成；
- `R_grounding`：预测时间段与标注的 IoU；
- `R_evidence`：答案是否被引用帧/台词支持；
- `R_quality`：增强后质量和音画一致性是否改善；
- `C_tool/C_latency`：模型、帧数、GPU 时间和重试成本；
- `P_hallucination`：无证据断言、人物或因果错误；
- `P_safety`：剧透泄漏、分龄漏检或非法未来知识访问。

第一轮 RL 只训练查询式高光的 Coordinator；其他任务先作为评测和后续扩展，避免奖励相互干扰。

## 6. 第一条训练闭环：查询式高光

### 输入

- 一段未裁剪视频；
- 自然语言查询；
- 工具预算。

### 输出

- 一个或多个高光时间段；
- 每个时间段的 saliency；
- 关键帧和台词证据；
- Agent 工具调用轨迹。

### 当前数据

- 已运行：Tears of Steel 开放电影的 8 分钟样本，10 条人工复核查询（7 train / 3 eval）；
- 后续规模化：QVHighlights；
- 可选辅助：ActivityNet Captions 与 NExT-QA。

当前自建数据用于证明端到端业务闭环，不用于宣称跨视频泛化。

### 自动 reward

- moment retrieval IoU / Recall@K；
- highlight mAP / HIT@1；
- 引用证据是否位于标注区间；
- 调用次数、观察帧数和总时延。

## 7. 对照实验

1. 固定均匀采样 + 单次 VLM；
2. 固定智能 Pipeline；
3. 单 Agent 工具调用，无训练；
4. 规则多 Agent；
5. 多 Agent + RL Coordinator。

关键消融：

- 去掉 TimelineScout；
- 去掉 EvidenceAuditor；
- 去掉工具成本项；
- 固定 Chunk 与动态 Chunk；
- 原片理解与质量感知路由；
- 全局 reward 与分项 reward。

只有在相同数据 split、相同基础模型和相同预算下，才能归因于 Agent 或 RL。

## 8. 扩展业务

### 无剧透内容解释

用 `first_visible_at <= user_progress` 做数据权限硬过滤；评测答案正确率和 spoiler leakage。

### 分龄引导

先定位风险片段，再结合视觉、音频、文本、上下文、年龄和家长策略生成差异化解释；评测高风险召回、误杀和策略一致性。

### 质量诊断与增强路由

将 FlashVSR 作为昂贵工具，学习“是否增强、增强哪些片段、采用 1× 修复还是 2× 超分”，验收清晰度、时序稳定、人脸/文字真实性、音画同步和成本。

### 海报候选

使用高光与人物证据筛选素材，再接裁剪、外扩和 Top-K；现有内部依赖无法公开复现的部分只保留接口，不纳入 MVP 成功标准。

## 9. 工程与事实边界

- 不改写 TCL 原始目录，集成通过 adapter 完成；
- 所有原始数据、模型和实验均记录 manifest；
- PPT 中的目标值只作为历史方案信息，不作为个人项目结果；
- 留存代码只能证明对应模块存在，不能证明剧情、高光、无剧透和分龄已经完整上线；
- 简历结果必须来自 VideoOps-RL 自己保存的日志、轨迹和评测报告。
