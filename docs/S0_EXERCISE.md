# S0：状态、动作、证据与 Reward

## 1. 当前任务

视频被简化成三个时间片段：

| 片段 | 时间 | 字幕 | 视觉标签 |
|---|---:|---|---|
| `seg_01` | 0—5 秒 | 主角走进房间 | person, room |
| `seg_02` | 5—10 秒 | 主角发现异常并说出真相 | person, surprise |
| `seg_03` | 10—15 秒 | 两人开始争吵 | person, conflict |

任务 query 是“发现异常”，标注目标是 `seg_02`。

## 2. 五个概念

- `state/observation`：Agent 当前知道什么，而不是环境知道的全部真相；
- `action`：Agent 对环境执行的一次操作；
- `evidence`：Agent 实际检查过、能够支持结论的信息；
- `reward`：当前一步产生的训练信号；
- `episode return`：整条轨迹所有 step reward 的和。

目标片段属于环境隐藏的 ground truth，不应直接出现在 observation 中。任务 query 必须在 observation 中，否则 policy 不知道自己要找什么。

## 3. 当前奖励表

| 条件 | Reward |
|---|---:|
| 每次普通工具调用 | -0.05 |
| 搜索质量奖励 | `+0.10 × F1` |
| 搜索完全无命中 | 额外 -0.05 |
| 首次检查目标证据 | +0.20 |
| 重复检查同一片段 | -0.10 |
| 有正确片段且检查过证据后提交 | +1.00 |
| 错误提交或没有证据就提交 | -0.50 |

提交动作使用终局 reward，不再额外扣普通工具成本。

## 4. 先手算，后运行

### 轨迹 A：efficient

```text
搜索“发现异常”
检查 seg_02
提交 seg_02
```

### 轨迹 B：duplicate

```text
搜索“发现异常”
检查 seg_02
再次检查 seg_02
提交 seg_02
```

### 轨迹 C：broad_search

```text
搜索“主角”
检查 seg_02
提交 seg_02
```

### 轨迹 D：no_evidence

```text
没有检查任何片段
直接提交 seg_02
```

请先手算三条轨迹的 episode return，再运行：

```powershell
python examples/s0_manual_episode.py efficient
python examples/s0_manual_episode.py duplicate
python examples/s0_manual_episode.py broad_search
python examples/s0_manual_episode.py no_evidence
```

## 5. 需要回答

1. 四条轨迹的总 reward 分别是多少？
2. 为什么轨迹 D 明明猜中了 `seg_02`，仍然失败？
3. 轨迹 A 和 B 最终答案相同，为什么 B 的 return 更低？
4. A 和 C 都找到目标，为什么 C 得分更低？
5. `target_segment_ids` 为什么不能直接放进 observation？

## 6. 第二任务：跨片段事件

第二任务不再只有一个正确片段：

| 片段 | 内容 |
|---|---|
| `seg_01` | 控制室设备正常运行 |
| `seg_02` | 主角发现设备冒烟 |
| `seg_03` | 主角立即关闭电源 |

问题是“主角如何发现并处理设备异常？”，完整答案需要同时引用 `seg_02` 和 `seg_03`。

```powershell
python examples/s0_multisegment_task.py complete
python examples/s0_multisegment_task.py partial_answer
python examples/s0_multisegment_task.py missing_evidence
```

三条轨迹分别检验：

- 内容完整且证据完整；
- 证据真实但只回答了一半；
- 答案猜完整了，但没有检查全部证据。

这里必须区分两个判断：

```text
correct  = 提交的目标片段是否完整正确
evidence = 提交的每个片段是否都被 Agent 实际检查
```
