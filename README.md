# Ceramic Restoration - 点云修复系统

基于 PoinTr 预训练模型的陶瓷/文物点云修复系统。

## 项目简介

这是一个用于修复残缺点云模型的 Web 应用，使用深度学习模型（PoinTr）进行点云补全。

## 功能

- 上传残缺的点云模型（.ply, .xyz 格式）
- 使用 PoinTr 预训练模型进行自动修复
- 返回修复后的完整点云模型
- Web 界面易于使用

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/quewww/ceramic-restoration.git
cd ceramic-restoration
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载预训练模型

由于模型文件较大（~400MB），需要单独下载：

```bash
# 创建模型目录
mkdir -p PoinTr/ckpts

# 下载 PoinTr 预训练模型
# 从项目原始仓库下载：
# https://github.com/quewww/PoinTr
# 或使用官方提供的下载链接
```

## 使用

### 启动应用

```bash
python app.py
```

访问 http://localhost:5002

### 使用命令行测试

```bash
python test_repair_quality.py
```

## 项目结构

```
ceramic_restoration/
├── app.py                      # Flask Web 应用
├── config.py                   # 配置文件
├── enhanced_point_completion.py # 点云补全模块
├── point_cloud_completion.py    # 点云补全核心
├── PoinTr/                     # PoinTr 模型
│   ├── models/                 # 模型定义
│   ├── cfgs/                  # 配置文件
│   └── ckpts/                 # 预训练模型（需单独下载）
├── templates/                  # HTML 模板
├── uploads/                    # 用户上传文件
└── processed/                  # 处理后的文件
```

## 技术栈

- **后端**: Flask (Python)
- **深度学习**: PyTorch, PoinTr
- **点云处理**: Open3D, NumPy
- **前端**: HTML5, JavaScript

## 注意事项

- 预训练模型文件（~400MB）未包含在仓库中，需要单独下载
- 上传的点云文件格式支持：.ply, .xyz, .pcd
- 建议输入点云点数在 500-5000 之间以获得最佳效果

## 许可证

MIT License

## 作者

quewww
