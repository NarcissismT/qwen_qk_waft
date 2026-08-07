# WAFT 官方实现对齐说明

运行时实现以 WAFT 官方仓库 `waftv2` 分支为结构基准。Qwen-QK 只替换
WAFT 的双图特征编码器；循环更新器不再使用项目自写 Transformer 或自写
DPT decoder。

## 官方结构的代码位置

| 官方模块 | 本项目运行时代码 | 对齐内容 |
|---|---|---|
| `model/backbone/patch_embed.py` | `src/qwen_qk_waft/official_waft/patch_embed.py` | 官方卷积 PatchEmbed，`patch_size=8` |
| `model/backbone/vit.py` | `src/qwen_qk_waft/official_waft/vit.py` | timm `vit_small_patch16_224`，第 2/5/8/11 层，官方位置编码与 forward |
| `thirdparty/DepthAnythingV2/depth_anything_v2/dpt.py` | `src/qwen_qk_waft/official_waft/dpt.py` | 官方 DPTHead 的 projection、四级 resize、scratch、fusion 和输出头 |
| `thirdparty/DepthAnythingV2/depth_anything_v2/util/blocks.py` | `src/qwen_qk_waft/official_waft/blocks.py` | 官方 residual convolution 与 feature fusion block |
| `model/waft_a2.py` 循环核心 | `src/qwen_qk_waft/models/waft.py` | `hidden_conv`、`warp_linear`、`refine_net`、`refine_transform`、6 通道 flow/info head 和 2 倍 convex upsample |

官方 BSD 3-Clause 许可证保存在
`src/qwen_qk_waft/official_waft/LICENSE`。

## 预训练参数

训练默认读取：

```text
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/models/WAFT/vit_small_patch16_224_imagenet.safetensors
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/models/WAFT/waft_dav2_a2_zero_shot.ckpt
```

第一项是 timm 官方 `vit_small_patch16_224.augreg_in21k_ft_in1k` 权重，创建
`VisionTransformer` 时先严格加载。

该 checkpoint 是 PTLFlow 对 WAFT DAv2-A2 zero-shot 权重的 PyTorch Lightning
镜像。其 WAFT 参数名保持为官方命名。加载器在
`src/qwen_qk_waft/models/waft_checkpoint.py`，分别对以下模块执行严格加载：

- `refine_net`，包含 12 个官方 timm ViT block、PatchEmbed、位置编码和 DPTHead；
- `hidden_conv`；
- `warp_linear`；
- `refine_transform`；
- `flow_head`；
- `upsample_weight`；
- 独立的 DPT-Q 与 DPT-K 都从 `refine_net.dpt_head` 初始化。

随后完整 ViT 和 DPTHead 再从 WAFT checkpoint 严格恢复为光流预训练参数。

当前默认实验明确是 **DAv2-A2 initialization**。官方仓库没有 GitHub Release，
DINOv3-A2 与官方推荐 A1 只放在 Google Drive model zoo；当前执行主机无法连接
Google Drive，因此本次没有伪造这两组 checkpoint 消融。`waft_checkpoint` 可替换
为其他同构 A2 checkpoint，加载器会继续执行严格 key/shape 校验；A1 因 head
接口不同应使用单独实现和配置，不能直接冒充 A2 权重载入。

## 方案要求的任务适配

以下部分不是对 WAFT 官方双帧光流任务的机械复制，而是架构方案明确要求的接口：

1. `models/dpt.py`：每个选中 Qwen 层先用独立 LN/Linear 投影到 384 维，
   再进入两个互不共享、已用 WAFT DPTHead 参数初始化的 DPT-Q/DPT-K。
2. `models/model.py`：Stage-A backward map 提供非零初始位移；Qwen target-Q 和
   source-K 替代官方 DINO/DAv2 双图 encoder。
3. `models/waft.py`：第一次 `hidden_conv` 使用按 Stage-A 位移对齐后的 source
   feature；confidence gate 保留在官方 updater 外部。
4. source local encoder 在 Phase C 加入，融合卷积按
   `[identity, zero-local]` 初始化，因此切换阶段时保持 Phase B 函数不变。
5. 当前接口不复制官方图像 Padder，而是在入口明确要求高宽能被 16 整除；
   updater 内的 feature warp 已恢复官方 `padding_mode=zeros`。

除此之外，循环输入仍严格包含官方的
`[target feature, warped source feature, hidden state, current flow]`，不会删掉
官方 `warp_linear` 中的 2 通道 flow。

## Qwen MMDiT 层选择

正式训练不预先硬编码某四层。Phase 1 扫描 Qwen MMDiT 全部 60 层、候选去噪步
以及 pre/post-RoPE 版本，再将探针选出的 top-4 层写入
`runs/qwen_qk_waft/phase1_probe/selection.json`。DPTHead 固定接收这四层。

探针的 true-match margin 会先屏蔽真实位置，再与最强非匹配位置比较；同时记录
forward-backward cycle consistency、三随机种子稳定性，以及文字结构、页面边缘、
四角、内部纹理和空白背景分区结果。这里的“文字结构”来自 target 图像的固定
边缘/局部纹理规则，不冒充 OCR 标注文字行。

## 训练安全与连续性

- 三个 DPTHead 中永远不经过 intermediate forward 的 `output_conv2`，以及
  `refinenet4` 无 skip 输入时永远不调用的 `resConfUnit1` 均被冻结；因此正式
  DDP 保持 `find_unused_parameters=False`。
- validation 与 inference 和训练共用 BF16 autocast；probability BCE 单独在
  FP32 中计算。
- Phase D 的 gate 初始化为 0.99，保持 Phase C residual；错误 prior 区域的
  correction loss 使用正的 improvement margin。
- checkpoint 不再只按平均或伪 line EPE 选择，而使用 EPE、P95、文字结构、
  straightness、边缘、角点、masked fold 和 invalid rate 的配置化联合分数。
