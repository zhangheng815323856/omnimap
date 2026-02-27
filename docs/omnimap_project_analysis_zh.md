# OmniMap 项目解析与自定义数据集准备指南（中文）

## 1. 项目做了什么

OmniMap 是一个**在线三维建图框架**，目标是在单一系统中同时维护三类能力：

1. **光学外观（Optical）**：高保真的可渲染外观（基于 3D Gaussian Splatting）。
2. **几何结构（Geometric）**：稳定的体素/TSDF 几何重建。
3. **语义理解（Semantic）**：开放词汇（open-vocabulary）实例语义融合。

核心思路是把 3DGS 与 TSDF 融合为一个紧耦合系统：

- `OMNI.track()` 每帧同时调用 TSDF 融合和 3DGS 更新。
- 结束阶段输出渲染评估、实例结果和网格结果。

## 2. 关键实现链路（代码级）

### 2.1 输入流与数据解码

入口在 `demo.py`：

- 读取 RGB、Depth、Pose、相机内参。
- 深度按 `depth_scale` 归一化到米。
- Pose 文件按每行 16 个数字 reshape 成 `4x4` 变换矩阵。
- 将帧打包后送入 `OMNI.track()`。

### 2.2 双后端协同

`omnimap/omni.py` 里每帧执行：

1. `TSDFBackEnd.integrate(...)`：更新 TSDF 与实例体素。
2. `GSBackEnd.process_track_data(...)`：更新高斯点、关键帧和渲染优化。

终止阶段调用：

- `eval_fast` / `eval_rendering`：输出渲染与误差评估。
- TSDF `finalize()`：导出几何与实例结果。

### 2.3 语义模块（开放词汇实例）

在 `omnimap/tsdf_backend.py`：

- YOLO-World：开放词汇检测（框）。
- TAP：基于检测框生成实例 mask 与 caption token。
- SBERT：文本特征聚合，做实例语义一致性融合。
- spaCy：NLP 预处理。

语义标签表来自 `pretrained_models/yolo_labels.txt`。

## 3. 项目实现了哪些任务

当前仓库直接支持（README + 代码路径）：

1. **在线 RGB-D 三维建图**（Replica / ScanNet）
2. **3DGS 视角渲染与评估**
3. **TSDF 网格重建与导出**
4. **实例级语义融合与可视化**（open-vocabulary）

## 4. 如果要准备自定义数据集，需要什么输入

### 4.1 最小必需输入（每个场景）

无论你是否做语义，基础建图最少都要：

1. **RGB 序列**（按时间顺序命名）
2. **Depth 序列**（与 RGB 一一对齐）
3. **每帧位姿 Pose**（`traj_w_c.txt`，每行 16 数，4x4）
4. **相机内参**（fx, fy, cx, cy，和可选 depth_scale / 畸变参数）

### 4.2 目录结构建议（对齐现有 loader）

- Replica 风格：

```text
<dataset_root>/<scene>/imap/00/rgb/*.png
<dataset_root>/<scene>/imap/00/depth/*.png
<dataset_root>/<scene>/imap/00/traj_w_c.txt
```

- ScanNet 风格：

```text
<dataset_root>/<scene>/color/*.jpg
<dataset_root>/<scene>/depth/*.png
<dataset_root>/<scene>/traj_w_c.txt
<dataset_root>/<scene>/intrinsic/intrinsic_color.txt
```

### 4.3 文件格式细节（非常关键）

1. **RGB 图像**：`cv2.imread` 可读取格式（png/jpg）。
2. **Depth 图像**：通常 16-bit 深度图；读取后会除以 `depth_scale`。
3. **Pose 文件**：
   - 每行 `16` 个浮点数；
   - 代码按 `c2w` 读取，再求逆得到 `w2c` 参与优化。
4. **内参文件**：支持两种形式：
   - `3x3` 矩阵（默认 `depth_scale=1000`）；
   - `1xN` 向量，前四项为 `fx fy cx cy`，第 5 项可作为 `depth_scale`。

### 4.4 语义分支额外依赖（如果要保留 open-vocabulary 实例）

除数据本身外，还需要在配置中正确设置模型权重路径：

- YOLO-World config + checkpoint
- TAP 两个权重
- SBERT checkpoint

这些路径在 `config/*_config.yaml` 的 `path` 段配置。

### 4.5 迁移到“你自己的数据”建议步骤

1. 先仿照 ScanNet 风格准备一个 scene。
2. 把轨迹统一成 `traj_w_c.txt`（4x4/行）。
3. 准备 `intrinsic_color.txt` 或单行 `fx fy cx cy depth_scale`。
4. 修改 `config/<your_dataset>_config.yaml` 的 `path.data_path`。
5. 在 `demo.py` 新增 `elif args.dataset == "yourset"` 路径映射。
6. 先关闭 GUI 跑通，再开 GUI 检查实例/渲染。

## 5. 结论

OmniMap 的本质是“**一套输入（RGB-D+Pose+K）驱动三种地图能力（外观+几何+语义）**”。

所以自定义数据集是否可用，关键不在“类别标签是否完整”，而在于：

- RGB-Depth 时间与分辨率是否对齐；
- 轨迹是否是稳定的每帧 4x4；
- 内参和 depth_scale 是否正确；
- 是否提供语义模型权重路径。

只要这四点满足，大多数室内 RGB-D 序列都能迁移到 OmniMap 流程。
