# SDR_enhance

当前目录包含两个可选架构：逐帧的共享 ViT Global/Grid 模型，以及在共享 ViT 特征上加入时序递归融合的 Recurrent 版本。


## `models/` 里的模型

| 文件 | 主要结构 | 作用 |
|---|---|---|
| `colorfm_l.py` | DINO 风格 ViT（Patch Embed、位置编码、Transformer Blocks、LayerNorm）；同时定义传统的 `ParamGenerator` 和逐像素 MLP 应用函数。 | 作为仅使用Global的baseline |
| `colorfm_l_spatial_grid.py` | 轻量化的 Global + bilateral Grid 基础模型。包含 `SpatialAffineGridHead`、`LearnedGuideMap`、三线性 Grid 切片和局部仿射残差应用；基础 Grid Head 可融合多个 ViT 中间层特征。 | 以较轻量的结构引入 Guide、Grid 切片和局部仿射残差 |
| `colorfm_l_spatial_grid_shared_vit.py` **(当前默认baseline)**| 一个 ViT encoder 的最终 patch tokens 分别送入 Global Head 和 Grid Head。 | 统一简化Global和Grid的预测流程 |
| `colorfm_l_spatial_grid_shared_vit_recurrent.py` | 在共享 ViT 编码特征上逐帧执行 SimpleGate 递归融合，再由 Global/Grid heads 预测每帧参数。 | 时序版本


- colorfm_l_spatial_grid_shared_vit.py 的结构流：
  ```text
  输入 SDR
    ├─ resize 到 context_size × context_size（默认 256 × 256）
    │    └─ Patch Embed → ViT → 最终归一化 patch tokens [B, 256, 192]
    │           ├─ Global Head：token 平均池化 → MLP → 4 个 RGB kernels
    │           │       └─ kernels 应用于输入图像像素 → Global 图像
    │           └─ Grid Head：tokens reshape 为 [B, 192, 16, 16]
    │                   → 1×1 投影 → 卷积残差块 → 预测 3D affine Grid
    ├─ 原始 SDR → Learned Guide Map
    └─ 根据 (x, y, guide) 三线性采样 Grid，应用局部 affine 残差
            └─ Global 图像 + 局部残差 → 最终增强图像
  ```
  默认配置 `configs/colorfm_l_spatial_grid_shared_vit.yaml` 中，ViT 为 `embed_dim=192`、`depth=6`、
  `num_heads=3`，`use_cls_token=false`；因此 256×256 输入（patch size 16）得到 16×16、
  共 256 个 patch tokens。Global Head 对平均 token 预测形状为
  `[3,16]、[16,16]、[16,16]、[16,3]` 的四个 kernels。Grid 默认形状为
  `[B,12,8,16,16]`，每个 cell 对应 12 个 affine 系数（3×4），Guide 的深度维用于双线性/三线性切片。

-  Recurrent 版本的时序融合
  Recurrent 模型仍使用同一个 ViT encoder。一个 clip 内的 N 帧先合并为 `B×N` 并行编码，得到每帧特征图；之后按帧顺序递归融合当前特征 `F_t` 和前一融合状态 `H_{t-1}`。SimpleGate 模块将两者拼接，经 1×1 投影和 depthwise 3×3 卷积后拆成两支并逐元素相乘，再投影为更新量：
    ```text
    H_t = F_t + β · ProjectOut(GateA ⊙ GateB)
    ```
    `β` 是零初始化的可学习残差缩放，初始时输出等于当前帧特征；首帧以前一特征等于自身初始化，之后传递融合后的状态。Global Head 和 Grid Head 都读取递归融合后的特征。不同于参数 EMA，这里递归的是 encoder feature，不是 kernels；Grid/Guide 仍按每帧当前输入产生。
    Recurrent 训练数据必须是连续片段，默认 `train_clip_length: 10`。输入形状为 `[B,N,C,H,W]`；默认 `batch_size: 2` 表示每批 2 个 clip，即 20 帧。推理按视频时间顺序处理并延续前序状态，遇到非连续帧会重置；不应打乱帧顺序。

## 训练和推理

- 训练入口：`main_train_xxx.py`
- 模型 config：`configs/`
- 训练 solver：`solvers/`
- 单视频命令行推理：`infer_video.py`
- Source/GT 配对视频推理：`infer_sdr_video.py`
- 数据集和时间划分：`datasets/`

先按 config 配好数据路径，再在项目目录运行（训练数据默认为根目录下包含 source/ 和 target/ 目录，每个目录下包含多个视频文件对应。默认采用在线抽帧，对于非递归版本，可能有IO瓶颈）：

```bash
python main_train_xxx.py
python main_train_xxx.py --resume none
```
`--model-only` 检查结构与参数量，`--dry-run` 检查视频配对和时间划分但不训练。默认每个视频使用前段训练，末尾 300 秒作为共用 validation/test，中间留 30 秒间隔；训练按时间桶随机抽取单帧。Source/GT 必须逐帧配对。

### 单个输入视频：`infer_video.py`

适用于只有待增强 SDR 视频、没有对应调色师 GT 视频的情况。它按时间顺序逐帧推理，默认保存原始片段、Global 输出和 Global+Local 输出。支持本项目的共享 ViT 逐帧模型和 Recurrent 模型，checkpoint 必须与 config 的架构一致。

```bash
python infer_video.py \
  --config configs/xxx.yaml \
  --checkpoint runs/xxx/checkpoints/last.ckpt \
  --input "data/my_video.mp4" \
  --start 00:00:10 --end 00:00:40 \
  --output video_inference_outputs/my_video_xxx
```

--start 包含起始时刻，--end 不包含结束时刻；时间可写秒数、MM:SS 或 HH:MM:SS。默认输出 source.mp4、global.mp4、global_local.mp4；加 --no-save-source 可省略原片。

Recurrent 模型示例：
```bash
python infer_video.py \
  --config configs/colorfm_l_spatial_grid_shared_vit_recurrent_sdr.yaml \
  --checkpoint runs/colorfm_shared_vit_recurrent_simple_gate_clip10_temporal_tail/checkpoints/last.ckpt \
  --input "data/my_video.mp4" \
  --start 10 --end 40 \
  --output video_inference_outputs/my_video_recurrent
```

逐帧共享 ViT 对每帧独立预测参数，不做 kernel 邻帧平均；Recurrent 架构会按时间顺序延续 encoder 特征状态。后者是模型本身的递归融合，不等同于推理时的 kernel averaging。

### Source/GT 配对视频：`infer_sdr_video.py`

适用于需要把模型结果和调色师 GT 对齐检查的情况。需要提供配对的 Source 与逐帧对齐 GT 视频（或通过 --video 从 config 的数据目录选择一对），输出原片、GT、Global、Global+Local 四段视频。默认开启 Global kernel 邻帧平均；用 --kernel-average off 关闭。该脚本也不自动计算 PSNR/SSIM，主要用于生成同步对比视频。

视频片段推理示例：

```bash
python infer_sdr_video.py \
  --config configs/colorfm_l_spatial_grid_shared_vit.yaml \
  --checkpoint runs/shared_vit_sdr/checkpoints/last.ckpt \
  --source-video "data/source/input.mp4" \
  --gt-video "data/target/input_gt.mp4" \
  --start-sec 0 --end-sec 30 \
  --kernel-average off
```

`infer_sdr_video.py` 会根据 config 中的 `model.architecture` 选择逐帧或 Recurrent 模型。对 Recurrent 模型，帧按时间顺序递归；若要观察纯递归效果而不叠加推理阶段的 Global kernel 邻帧平均，请加 `--kernel-average off`。Checkpoint 必须与所用架构匹配。

Recurrent 推理示例：

```bash
python infer_sdr_video.py \
  --config configs/colorfm_l_spatial_grid_shared_vit_recurrent_sdr.yaml \
  --checkpoint runs/colorfm_shared_vit_recurrent_simple_gate_clip10_temporal_tail/checkpoints/last.ckpt \
  --source-video "data/source/input.mp4" \
  --gt-video "data/target/input_gt.mp4" \
  --start-sec 0 --end-sec 30 \
  --kernel-average off
```

### 训练参数

默认训练监督尺寸为 540×960，ViT context 为 256×256。Loss 由最终输出 L1、Global 输出 L1、Grid 残差幅值惩罚和 Grid Total Variation 组成，具体系数均在 YAML 的 `solver` 段配置。
