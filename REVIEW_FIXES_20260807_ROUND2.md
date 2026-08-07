# 第二轮审查修改记录（2026-08-07）

本轮按审查意见逐项处理；按用户要求，不处理审查第三部分第 6 小点。

| 审查项 | 修改位置 | 结果 |
|---|---|---|
| BF16 污染坐标场 | `geometry.py`, `stage_a.py`, `model.py`, `waft.py`, `infer.py` | map/grid/displacement/residual/Jacobian/native grid 全程 FP32；特征网络仍可 BF16 |
| 512px 恒等图伪 fold | `tests/test_geometry.py`, `scripts/coordinate_precision_audit.py` | 检查 512 个唯一坐标、identity fold=0；另测 2K/4K native map 与一次原图采样 |
| 真实端到端预检 | `scripts/full_model_preflight.py` | 真实 Qwen target-Q/source-K 经过独立 DPT、Stage-A、WAFT-A2 和 renderer，并核对所有坐标 tensor dtype |
| Stage-A global pose/parity | `stage_a_audit.py` | 从旧 checkpoint 读取配置与 state，调用旧完整模型 `forward(stage="prior")`；global pose 配置/state 均为关闭，map 最大差为 0 |
| Gate 仅监督末轮 | `losses.py`, `train.py` | 全部轮次使用 sequence-weighted BCE；每轮独立输出 Brier、ECE 和 10-bin histogram |
| Bending mask 近似全开 | `losses.py` | 改为连续 `flat_weight` 加权均值，不再使用 `flat_weight > 0` |
| Line/OCR/损伤/Stage-A 基线 | `data.py`, `metrics.py`, `formal_quality.py`, `train.py` | 支持 line mask/instance 标注和真实基线拟合；实现 OCR 字符保持率；报告 high-confidence damage 及 prior fold/invalid |
| 失败模型仍写 best | `train.py`, YAML | `best.pt` 只在 geometry criteria 全部通过时写入；整阶段无合格 checkpoint 会非零退出 |
| LoRA 强度与原推理不确定 | `qwen_qk.py`, `qwen_lora_audit.py`, `full_model_preflight.py` | 解析原训练 rank/alpha 来源，直接采用原 DiffSynth `scale*(B@A)` 合并规则；对真实输入的 Q/K projection 再做同输入数值比较 |
| 原方案对 flow 输入表述过时 | `Qwen-QK-WAFT_model_architecture_training_plan.md` | 明确 current displacement 同时用于 source warp，并作为 2 通道输入送入官方 `warp_linear` |
| 2-GPU 预检尺寸过小 | `scripts/ddp_preflight.py` | 从 64×64 提升到 512×512、BF16、两 step，并断言最终 map 为 FP32 |

本地容器已完成单元/数值检查；真实 Qwen 全链、GPU 512/2K/4K 和更新后的
2-GPU 预检由 Slurm 入口自动运行并生成固定 JSON 报告。它们通过之前不会进入
正式八卡阶段。工程测试或预检通过也不等同于正式收敛、OCR 和人工视觉验收。
