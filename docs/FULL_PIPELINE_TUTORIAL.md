# VideoOps-RL 全链路教程

> 版本说明：本文保留 v1 媒体预处理和最初 POC 的形成过程，便于理解项目怎样一步步搭起来。当前算法、公开数据规模、reward、训练数据和服务器执行请以 [`ALGORITHM_V2_GUIDE.md`](ALGORITHM_V2_GUIDE.md) 为准；本文第 5--10 节中的 v1 数字不能用于描述当前版本。

这份文档回答四件事：项目解决什么问题、每层代码做了什么、RL 到底学什么、上服务器后怎样训练。建议第一次按顺序读，面试复习时直接看最后两节。

## 1. 一句话理解项目

输入一部长视频和一句需求，例如“找到兔子用绳索布置陷阱的片段”。普通大模型往往直接猜时间，VideoOps-RL 则像一个有预算的剪辑团队：先搜索候选镜头，再看关键帧，必要时扩展上下文，交给独立审计角色检查证据，最后提交 shot ID 和时间区间。

```text
视频 + 字幕 + 查询
  -> 镜头切分与统一毫秒时间轴
  -> 字幕证据 / 关键帧证据
  -> Coordinator 调度多个专业 Agent 和工具
  -> EvidenceAuditor 复核
  -> 时间区间 + 证据帧 + 可解释轨迹
```

它的实用价值不是“模型也能回答剧情问题”，而是让回答能被媒体系统消费：输出必须能定位到视频、能回看证据、能统计调用成本、失败后能重放轨迹。RL 优化的是多步处理策略，而不是重新学习视觉知识。

## 2. 为什么不直接把整部视频交给大模型

整片输入有四个工程问题：上下文和显存成本随视频增长；时间定位不稳定；模型容易在证据不足时猜测；不同查询需要的处理深度不同。工具化 Agent 把“看什么、看几次、何时停止”显式化。规则可以给出一个初始流程，但规则无法很好处理同义表达、多个候选和成本—效果权衡，因此用 SFT 学合法流程，再用 GRPO 优化最终轨迹收益。

## 3. 数据从哪里来

全部媒体来自 Blender 开放电影，许可证为 CC BY 3.0：

| split | 视频 | 时长 | 镜头/关键帧 | 字幕 | 任务 |
|---|---|---:|---:|---:|---:|
| train | Tears of Steel 样本 | 480 s | 98 | 67 | 40 |
| val | Sintel | 888 s | 90 | 111 | 15 |
| test | Big Buck Bunny | 597 s | 118 | 0 | 11 |

为什么这样分：split 的最小单位是整部电影，不是片段。同一电影的画风、角色和字幕措辞很相似，如果随机切片，会让训练和测试看见近乎相同的内容，指标会虚高。Big Buck Bunny 没有对白，专门测试视觉工具是否真的有作用。

任务不是宣称人工大规模标注的 benchmark。它是一个小型工程语料：每条包含 query、目标时间段、ground-truth shot、所需模态和难度，用于跑通训练与评测协议。正式定义见 `schemas/videoops_task_v1.schema.json`，数据总表见 `data/registry/formal_dataset_manifest_v1.json`。

## 4. 视频怎样变成 Agent 能使用的证据

第一步用 FFmpeg 检测场景切换，再为每个镜头抽取中间关键帧；第二步解析 SRT 字幕；第三步把字幕与镜头都投影到同一个毫秒时间轴，形成 evidence unit：

```json
{
  "shot_id": "shot_0015",
  "start_ms": 73125,
  "end_ms": 76667,
  "keyframe_path": "keyframes/shot_0015.jpg",
  "transcript": "..."
}
```

关键帧不是视频编码，JPEG 是图像文件；MP4/MOV 是容器，H.264 是视频编码。Agent 不需要每次解码整部视频，只需要通过 shot 索引拿到对应证据。

真实踩坑：Big Buck Bunny 的切镜结果很多，最初把全部 frame selector 拼成一条 FFmpeg 命令，命令过长且 VFR 视频会缺少名义帧。`scripts/detect_shots_ffmpeg.py` 后来改为每批 40 帧，并在批量帧数不一致时按时间戳逐帧兜底，最终得到 118 个完整镜头。这是可在面试中讲的工程排障，不是算法效果。

## 5. 环境、工具和多智能体

核心环境是 `src/videoops_rl/multivideo_env.py`。它在初始化时加载当前 task 对应的视频证据，但 `public_prompt` 只暴露 query、video ID 和工具列表，绝不暴露 target 时间或 ground-truth shot。

六个工具：

1. `search_transcript`：TimelineScout 在字幕中找候选；
2. `search_visual`：VisionAnalyst 在视觉语义索引中找候选；
3. `inspect_keyframe`：真正返回关键帧和该镜头信息；
4. `expand_context`：候选边界不完整时查看相邻镜头；
5. `request_audit`：EvidenceAuditor 检查 shot 合法、是否看过图、所需模态是否覆盖；
6. `submit`：Coordinator 停止搜索并提交最终 shot。

`src/videoops_rl/agents.py` 里保留三种可比较策略。固定基线永远选中间镜头；单 Agent 自己搜索、看图、提交；多 Agent 把时间检索、视觉核验和证据审计拆开。角色拆分不是让四个模型互相闲聊，而是权限和职责隔离：审计者不能替 Coordinator 偷看答案，Coordinator 不能把没检查过的候选包装成有证据的结果。

## 6. Reward 为什么是组合式的

最终奖励为：

```text
R = 2.0 * temporal_IoU
  + 0.35 * evidence_supported
  + 0.25 * modality_coverage
  + 0.25 * audit_passed
  - 0.025 * tool_calls
  - 0.20 * invalid_calls
```

时间 IoU 是预测区间和目标区间的交并比。只奖励 IoU，模型可能碰巧猜中却不给证据；只奖励审计，模型可能机械调用所有工具；加入工具成本，才会学习“证据足够时停止”。当前权重是工程初值，不声称理论最优。服务器训练时应先观察各分量尺度，再决定是否调权重。

这是环境 reward，不是把标签塞进 prompt。环境内部可以持有 ground truth 来评分，模型观测中不能出现。`scripts/build_training_data.py` 对 GRPO prompt 做结构化公开字段校验，并检查任务 schema、时间边界、工具参数、CLIP 特征数值和重复策略。

## 7. SFT 和 GRPO 数据怎样生成

规则多智能体先充当 teacher，运行每个任务并保存工具轨迹。只有 IoU、证据接地和审计同时通过的轨迹进入 SFT；训练只对 assistant 工具调用计算 loss，`submit` 后的隐藏评分不会写入 messages。GRPO 文件只包含公开 prompt；训练时同一 `task_id` 组成 4 条 rollout，模型自己调用工具，结束后环境计算 reward。

文件位于 `data/training/`：

- `sft_train_v2.jsonl`：3,575 条成功 teacher 轨迹，其中开放影片成功轨迹重复 5 次；
- `sft_val_v2.jsonl`：776 条验证轨迹；
- `grpo_train_v2.jsonl`：9,018 条公开 prompt，包含 7,218 条 QVHighlights 和 1,800 条开放影片 episode；
- `training_data_audit_v2.json`：构建计数、成功率、重复策略和泄漏检查；
- test 文件只用于最终评测，不被训练入口读取。

SFT 的作用是“先学会怎么行动”，GRPO 的作用是“在多条合法行动轨迹里，更偏好效果高、证据足、调用少的那条”。直接跳过 SFT 做 GRPO，早期 completion 往往连工具 JSON 都不合法，奖励大多相同，训练信号很差。

## 8. 训练前 baseline 怎么看

`scripts/evaluate_pretraining_stack.py` 已在 66 条任务上真实运行。关键结果：

| 策略 | split | mIoU | 成功率 | 审计率 |
|---|---|---:|---:|---:|
| fixed chunk | test | 0.000 | 0.0% | 0% |
| single agent | test | 0.857 | 81.8% | 0% |
| multi-agent | test | 0.857 | 81.8% | 100% |
| multi-agent | val | 0.319 | 13.3% | 100% |

这组数不能解释成“多 Agent 准确率一定更高”。目前单 Agent 和多 Agent 的候选逻辑相近，所以定位指标接近；多 Agent 的确定收益是审计覆盖。验证集低分说明词法检索和单 shot 提交仍弱，也为 SFT/GRPO 后的同义检索、候选融合、上下文扩展留下真实改进空间。完整结果在 `outputs/reports/pretraining_stack_eval_v1.json`。

## 9. 8 张 H200 上怎样训练

基础模型是项目内的 `models/Qwen3-VL-2B-Instruct`。`server/run_all.sh` 自动选择 8 卡兼容模式或 24 卡快速模式；24 卡时第一阶段全部用于 ZeRO-2 LoRA SFT，第二阶段 20 卡训练、4 卡 vLLM rollout，最后 24 卡分片评测。

```bash
python server/preflight.py
bash server/run_all.sh
```

先只跑 10-20 step smoke，检查三件事：模型能否输出可解析工具调用；关键帧是否真的注入 VLM；reward 是否随成功/失败变化。通过后再把 `--max-steps` 调到 200。这个项目目标是跑通和理解，40 条训练任务不支持宣称泛化或生产效果。

## 10. 本地到服务器的边界

本机已完成：媒体下载、预处理、数据协议、split、环境、工具、多 Agent、reward、teacher 轨迹、训练数据、泄漏审计、训练脚本、8 卡配置、离线模型和打包器。H200 上尚未执行：SFT、GRPO 和 checkpoint 评测。因此简历目前可写“构建/实现/设计并完成训练前离线闭环”，训练跑完并保存日志后才写“完成 SFT+GRPO 训练”。

离线包不替服务器猜 CUDA/PyTorch 版本。数据和模型无需下载；服务器仍需要兼容 H200 的 NVIDIA 驱动、CUDA、Python 和 PyTorch。完全断网时，wheelhouse 必须在知道服务器环境版本后单独制作。

## 11. 面试怎么讲

两分钟版本：

> 我做的是长视频查询式高光 Agent。先把视频切成镜头，抽关键帧，并把字幕、镜头和时间统一成毫秒级 evidence unit。上层不是让 VLM 直接猜时间，而是让 Coordinator 调度字幕检索、视觉检索、看图、扩展上下文和审计工具；TimelineScout、VisionAnalyst、EvidenceAuditor 分别负责不同证据。Reward 同时考虑 temporal IoU、证据覆盖、审计、工具成本和非法调用。数据用三部 CC BY 开放电影按视频级拆分，构造 66 条任务；本地已经跑通基线、SFT/GRPO 数据生成和泄漏审计，服务器用 8 张 H200 先 LoRA SFT 再 6+2 卡 GRPO/vLLM。

常见追问：

- 为什么是 RL？因为搜索、看图、扩展、审计、停止是有成本的序列决策，监督数据只能教一条示范，不能直接优化整条轨迹的效果—成本折中。
- 多 Agent 有什么必要？主要价值是职责和证据权限隔离，不是为了堆角色；当前 baseline 也诚实显示定位增益尚未出现，但审计覆盖从 0 到 100%。
- 数据真实吗？视频、字幕和关键帧都是真实公开媒体；任务标签是为工程闭环自建的小规模标注，不冒充公开 benchmark。
- 最大风险是什么？语义索引仍是小规模标签/词法检索、训练任务只有 40 条、训练后效果尚未验证；所以项目定位是完整工程 POC，不是 SOTA 论文。
- 下一步怎么扩展？替换为真实 embedding/向量索引，增加 hard negative 和多片段任务，再接无剧透解释或海报候选；底层 evidence unit 和工具协议不需要重写。

## 12. 文件地图

- `scripts/build_formal_dataset.py`：正式任务、split、manifest 和审计；
- `src/videoops_rl/multivideo_env.py`：环境、工具、reward、标签隔离；
- `src/videoops_rl/agents.py`：三类训练前策略；
- `scripts/build_training_data.py`：SFT/GRPO 数据与泄漏检查；
- `server/train_sft.py`、`server/train_llm_grpo.py`：两阶段训练；
- `server/run_all.sh`：自动选择 8/24 卡执行入口；
- `scripts/build_server_package.py`：离线打包和校验；
- `outputs/reports/`：所有机器可读审计与 baseline 结果。
