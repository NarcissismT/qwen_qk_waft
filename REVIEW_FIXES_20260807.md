# 代码审查修复记录（2026-08-07）

| 审查问题 | 修改位置 | 当前行为 |
|---|---|---|
| DDP 未使用参数 | `models/dpt.py`, `models/waft.py` | 冻结三个 DPT 的 `output_conv2` 和 `refinenet4.resConfUnit1`；2 GPU 连续两 step 已通过，未开启 unused-parameter 搜索 |
| BF16 validation/inference | `train.py`, `infer.py`, `losses.py` | forward 使用与训练相同的 autocast；probability BCE 在 FP32 计算 |
| Stage-A 静默漏载 | `models/stage_a.py`, `stage_a_audit.py` | 统一 `module./model.` 前缀，要求 80/80 几何键及 shape 精确匹配，并与原始 v3.1 类做同输入 parity |
| LoRA 静默漏载 | `models/qwen_qk.py`, `qwen_lora_audit.py` | 1440 个 adapter tensor 严格匹配；探针记录每个 scale 的加载覆盖率并验证 Q/K 相对 base 发生变化 |
| 多 scale 显存生命周期 | `models/qwen_qk.py`, `probe.py` | 每个 scale 后删除 pipeline 引用、垃圾回收并清 CUDA cache |
| 坐标审计不阻断 | `data.py`, `audit.py` | 未知 flow format 直接报错；L1/PSNR/valid fraction 任一不达标即退出非零 |
| WAFT sampling/padding | `geometry.py`, `models/model.py` | feature warp 使用官方 zero padding；输入高宽必须能被 16 整除 |
| Phase B→C 不连续 | `models/model.py` | local fusion 初始化为 diffusion identity 加 zero local contribution |
| Phase C→D 不连续 | `models/waft.py` | gate 输出层初始化为常数 0.99 |
| correction 不要求改善 | `losses.py` | 错误 prior 区域加入 `required_improvement_px` margin |
| 探针 margin/分区不足 | `probe.py` | 修复 non-match margin，加入 cycle、三 seed 和五类区域指标及联合选层分数 |
| 评估/选模不足 | `metrics.py`, `train.py` | 增加 P95、结构 line、straightness、edge/corner、masked fold、Jacobian、invalid、逐轮更新、gate histogram、ECE/Brier、PSNR/SSIM，并用联合分数选 best |
| DAv2/DINOv3/A1 消融 | `configs/qwen_qk_waft.yaml` | 当前只标记为 DAv2-A2；官方 DINOv3/A1 仅在不可达的 Google Drive 发布，未把未取得权重写成已完成实验 |

## 已生成的运行证据

- `runs/qwen_qk_waft/stage_a_initialization.json`
- `runs/qwen_qk_waft/qwen_lora_initialization.json`
- `runs/qwen_qk_waft/ddp_bfloat16_preflight.json`
- `runs/qwen_qk_waft/phase0_audit.json`

这些是实现和 preflight 证据，不等同于正式 8 卡收敛、OCR 或人工视觉验收。
