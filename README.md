# 陶瓷文物三维点云修复系统

这是一个基于 Flask 和深度学习点云补全模型的陶瓷文物三维修复系统，主要用于修复花瓶、碗、盘等陶瓷模型的残缺点云。

当前项目以**点云修复**为核心，前端支持点云视图、实体渲染和线框结构之间的切换。实体和线框入口仍然保留，但当前修复流程主要输出点云结果。

## 主要功能

- 上传残缺的三维模型文件
- 使用 PoinTr 对点云进行自动补全
- 支持 CPU 和 CUDA GPU 推理
- 保留原始点云，并合并修复生成的点
- 对具有明显旋转对称性的花瓶、罐类模型进行缺失角度补充
- 对浅碗、浅盘等模型进行按高度分层的轮廓补全
- 在网页中查看：
  - 点云视图
  - 实体渲染
  - 线框结构
  - 原始模型、修复结果和同步对比
- 支持点云结果导出
- 支持简单的用户登录、注册和项目记录功能

## 技术栈

- **后端**：Python、Flask、Flask-Cors
- **深度学习**：PyTorch、PoinTr
- **点云处理**：Open3D、NumPy、SciPy、scikit-learn
- **前端**：HTML、CSS、JavaScript、Three.js
- **模型辅助库**：timm、einops、PyYAML、tensorboardX

## 环境要求

- Windows、Linux 或 macOS
- Python 3.9 及以上
- 建议使用虚拟环境
- 如果使用 GPU，需要安装与本机 CUDA 版本匹配的 PyTorch
- 如果使用 CPU，可以安装 CPU 版本的 PyTorch

## 安装步骤

### 1. 获取项目

```bash
git clone https://github.com/quewww/ceramic-restoration.git
cd ceramic-restoration
```

### 2. 创建虚拟环境

Windows：

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 提供了通用的 PyTorch 版本范围，没有强制指定 CPU 或 CUDA 安装源。需要使用 GPU 时，请根据 CUDA 版本从 PyTorch 官网安装对应版本。

例如，CPU 环境可以先安装 CPU 版 PyTorch，再安装其他依赖：

```bash
pip install torch torchvision
pip install -r requirements.txt
```

如果已经安装了可用的 PyTorch，请不要为了安装项目依赖而覆盖现有的 CUDA 版本。

## 准备 PoinTr 模型

项目默认使用以下配置和权重：

```text
PoinTr/cfgs/ShapeNet55_models/PoinTr.yaml
PoinTr/ckpts/pointr_training_from_scratch_c55_best.pth
```

请确认权重文件存在。如果仓库中没有包含权重文件，需要从 PoinTr 项目或模型发布页面下载，并放入：

```text
PoinTr/ckpts/
```

如果使用 PF-Net，还需要准备相应的权重文件。当前默认模型为 PoinTr。

## 启动系统

在项目根目录执行：

```bash
python app.py
```

启动后访问：

```text
http://localhost:5002
```

应用启动时会自动创建以下目录：

```text
uploads/    # 上传的原始模型
processed/  # 修复后的点云和其他处理结果
user_data/  # 用户相关数据
```

## 使用流程

1. 打开网页首页。
2. 选择或上传一个残缺模型文件。
3. 系统会先在修复页面中显示原始模型。
4. 等待点云修复完成。
5. 在顶部选择点云、实体或线框显示方式。
6. 通过对比模式查看修复结果、原始残损模型或同步对比。
7. 使用透明度滑块观察原始模型和修复结果的对应关系。
8. 需要时导出修复后的点云。

## 支持的文件格式

上传接口支持以下格式：

```text
.ply
.pcd
.txt
.xyz
.obj
```

点云补全效果与输入数据质量有关。建议：

- 输入文件包含有效的 `x、y、z` 坐标；
- 点云分布尽量均匀；
- 不要包含大量离群点；
- 残缺区域仍保留足够的边界轮廓；
- 模型姿态尽量稳定，避免整体倾斜或坐标轴方向异常。

## 修复流程说明

### PoinTr 全局补全

对于一般模型，系统会：

1. 读取原始点云；
2. 进行异常点过滤；
3. 采样到 PoinTr 需要的输入点数；
4. 归一化点云；
5. 执行 PoinTr 推理；
6. 将输出还原到原始坐标系；
7. 对预测点进行适度去噪；
8. 与原始点合并；
9. 保存最终点云。

原始点不会被后处理流程删除或修改。

### 旋转对称补充

对于轴对称评分较高的瓶、罐类模型，系统会在 PoinTr 结果基础上补充缺失角度的轮廓点。该流程不会对所有点进行无条件旋转复制，主要用于补充检测到的缺口区域。

### 浅碗和浅盘补充

浅碗、浅盘等模型会尝试：

1. 估计旋转轴；
2. 将点云转换到局部轴向坐标系；
3. 按高度分层；
4. 在每一层统计已有角度和轮廓半径；
5. 只对缺失角度生成补点；
6. 合并原始点和新增点。

由于残缺程度、姿态和点云密度不同，几何补全结果仍可能需要进一步调整。

## 命令行推理

也可以直接调用 PoinTr 推理流水线：

```bash
python pointr_inference_pipeline.py input.ply output.ply --device cpu
```

使用 CUDA：

```bash
python pointr_inference_pipeline.py input.ply output.ply --device cuda
```

命令行推理默认使用：

```text
PoinTr/cfgs/ShapeNet55_models/PoinTr.yaml
PoinTr/ckpts/pointr_training_from_scratch_c55_best.pth
```

## 项目结构

```text
ceramic_restoration/
├── app.py                       # Flask 主应用和上传接口
├── config.py                    # 项目目录和基础配置
├── pointcloud_utils.py          # 点云读写、归一化和几何补全工具
├── pointr_inference_pipeline.py # PoinTr 推理流水线
├── enhanced_point_completion.py # 备用点云补全流程
├── point_cloud_completion.py    # 点云补全兼容模块
├── pfnet_inference.py           # PF-Net 推理模块
├── templates/                   # 前端页面模板
├── PoinTr/                     # PoinTr 模型代码、配置和权重
├── Pointnet2_PyTorch/          # 点云采样相关代码
├── PF-Net-Point-Fractal-Network-master/
│                                # PF-Net 相关代码
├── uploads/                    # 用户上传的原始文件
├── processed/                  # 系统生成的修复结果
├── user_data/                  # 用户数据
├── requirements.txt            # Python 依赖
└── README.md                  # 项目说明
```

## 常见问题

### 1. 报错：`Torch not compiled with CUDA enabled`

说明当前安装的是 CPU 版 PyTorch，但程序尝试使用 CUDA。当前项目会在检测到 CUDA 不可用时自动切换到 CPU；如果仍然报错，请确认：

- 使用的是最新项目代码；
- PoinTr 模型内部没有被改回硬编码 `.cuda()`；
- CPU 版 PyTorch 与当前 Python 环境一致。

也可以显式使用 CPU 推理：

```bash
python pointr_inference_pipeline.py input.ply output.ply --device cpu
```

### 2. 修复结果不完整

点云补全属于模型推理结果，效果会受到以下因素影响：

- 缺口面积过大；
- 缺口边缘不完整；
- 输入点云过于稀疏；
- 模型姿态倾斜；
- 点云中存在大量噪声；
- 输入模型与训练数据中的物体类别差异较大。

建议先检查原始点云是否包含完整的缺口边缘，并尝试使用点数更多、噪声更少的模型文件。

### 3. 前端无法显示修复结果

请检查：

- Flask 服务是否仍在运行；
- `processed/` 目录是否生成了修复后的文件；
- 浏览器控制台是否存在资源加载错误；
- 上传文件格式是否在支持范围内；
- PoinTr 权重文件是否存在。

### 4. 为什么实体和线框没有修复结果？

当前项目优先保证点云修复质量，实体和线框的完整修复仍处于暂缓状态。前端入口保留用于显示和后续扩展，但当前主要结果是修复后的点云。

## 清理运行产生的文件

以下内容可以安全删除，程序下次运行时会自动重新创建：

```text
__pycache__/
processed/*
```

`uploads/` 中保存的是用户上传的原始模型，删除前请确认不再需要这些文件。

## 许可证

本项目及所使用的第三方模型代码分别遵循其原始项目的许可证。使用 PoinTr、PF-Net、PointNet2 等组件时，请同时遵守对应项目的许可证和使用要求。

## 作者

quewww
