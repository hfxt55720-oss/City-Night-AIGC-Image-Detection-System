# 城市夜景AIGC图像检测系统

城市夜景AIGC图像检测系统是一个面向城市夜景图像场景的桌面端 AIGC 图像检测工具。项目提供单张检测、批量检测、模型管理、Grad-CAM 热力图、检测记录、数据统计、数据集构建和自定义模型训练等功能。

## 功能特性

- 单张城市夜景图像 AIGC 检测。
- 批量图像检测与 CSV 结果导出。
- ResNet50、EfficientNet-B0、AIDE-Night 与自定义权重管理。
- Grad-CAM 热力图可视化。
- 检测历史记录、筛选、删除和导出。
- 数据统计图表。
- 数据集目录构建与训练集管理。
- 基于项目内相对路径的自定义模型训练。

## 项目结构

```text
AIGC_Detector/
  main.py                 # 程序入口
  ui/                     # PyQt6 界面与功能页面
  models/                 # 模型封装与模型注册
  inference/              # 单图、批量检测与 Grad-CAM 推理
  training/               # 训练脚本
  system/                 # 配置、历史记录和系统监控
  tools/                  # 辅助验证工具
  config/                 # 默认配置
  data/img/               # 图标资源
  models/weights/         # 权重放置目录，权重文件不随源码仓库提交
  datasets/               # 数据集放置目录，数据集不随源码仓库提交
  output/                 # 运行输出目录，结果不随源码仓库提交
  aide_external/          # AIDE 兼容依赖放置目录，见 THIRD_PARTY_SETUP.md
```

## 环境准备

建议使用 Python 3.10。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要 GPU 推理或训练，请按本机 CUDA 版本安装匹配的 PyTorch。

## 运行

```powershell
python main.py
```

首次运行前，请根据 `WEIGHTS.md` 放置模型权重。没有权重时，界面可以打开，但对应模型会显示不可用或无法执行检测。

## 模型权重

模型权重不随 GitHub 源码仓库提交。默认路径如下：

```text
models/weights/aide_night_best.pth
models/weights/resnet50_ai_detector.pth
models/weights/efficientnet_b0_ai_detector.pth
```

请阅读 `WEIGHTS.md` 获取权重放置说明。

## 数据集结构

自定义训练默认使用项目内数据集目录：

```text
datasets/Night-AIGC-Dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

数据集图片不随 GitHub 源码仓库提交。

## AIDE 第三方依赖

AIDE-Night 推理和训练需要 `aide_external` 中的兼容依赖源码。由于该部分来源于外部项目，公开仓库中默认不直接附带，请按 `THIRD_PARTY_SETUP.md` 准备。

## 打包

项目保留了 PyInstaller 和 NSIS 相关脚本：

```powershell
.\build_release.ps1
.\build_installer.ps1
```

打包产物会生成到 `build/`、`dist/` 或安装包目录，这些内容不应提交到 GitHub。

## 上传前检查

上传 GitHub 前请查看 `GITHUB_UPLOAD_CHECKLIST.md`。尤其要确认没有提交：

- 模型权重。
- 训练数据集。
- 检测结果、热力图和训练输出。
- 本地历史记录。
- 打包产物。
- 软著申请资料。

