# 服务器多卡 LLM-GRPO 执行说明（Algorithm v2）

本机已经验证 QVHighlights 全量 val/test 的 CLIP 检索与自适应时序 proposal，以及原始关键帧/字幕环境。`server/train_llm_grpo.py` 让 Qwen3-VL 在两套环境中按 80%/20% 混合采样，并用 TRL GRPO 优化工具轨迹。脚本通过 Python 编译和 Bash 语法检查，但尚未在 H200 上运行，因此不能写成已有训练结果。

## 建议资源

- 兼容机器：8 张 H200，SFT 8 卡，GRPO 6+2 卡；
- 快速机器：24 张 H200，SFT 24 卡；
- 24 卡 GRPO：0--19 卡训练，20--23 卡运行 TP=4 vLLM；
- 24 卡评测：val/test 同时启动，各自拆成 12 个确定性 shard；
- 首轮把 `--max-steps` 改为 20 做 smoke，通过后再跑默认 200 steps。

## 运行

```bash
pip install -r server/requirements-llm-grpo.txt
python server/preflight.py --required-gpus 24
bash server/run_all.sh
```

预检会确认 8 张 GPU、Qwen 权重、CLIP 权重、12,562 个 QV 特征文件、查询向量索引、v2 训练数据、磁盘空间和 TRL `environment_factory` 接口。正式开跑后应保存 `pip freeze`、GPU 信息、vLLM 日志和两个 checkpoint 评测报告。

## Go / No-Go

只有同时满足以下条件才进入 200-step 训练：

1. 20-step smoke test 无工具协议错误；
2. 每个任务至少生成两种不同轨迹，组内 reward 非零方差；
3. 峰值显存低于单卡容量的 90%；
4. 输出中能看到 `search/inspect/audit/submit`，且 reward 与本地环境一致；
5. eval 三条任务不进入训练采样。

若失败，优先退回 `Qwen/Qwen3-0.6B` 验证纯文本工具协议，再恢复 VLM；不要把“能启动”当成训练有效。
