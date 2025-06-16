# VGGT ComfyUI 集成

在 ComfyUI 中集成 Facebook Research 的 VGGT（Visual Geometry Grounded Transformer）模型，实现原生的多视图3D重建、相机姿态估计、深度预测和轨迹跟踪功能。

## 功能特性

- ✅ **原生VGGT算法**: 直接使用官方VGGT推理流程
- ✅ **多视图3D重建**: 支持1-200帧图像序列输入
- ✅ **相机姿态估计**: 自动估计相机内外参数
- ✅ **深度图预测**: 生成高质量密集深度图
- ✅ **点云重建**: 基于深度和姿态信息重建3D点云
- ✅ **原生结果输出**: 保留所有原生Tensor数据
- ✅ **自动模型管理**: 支持本地缓存和HuggingFace自动下载

## 安装

### 1. 自动安装（推荐）

```bash
cd custom_nodes/ComfyUI_VVL_VideoCamera_Advanced
python install.py
```

### 2. 手动安装

```bash
# 安装VGGT专用依赖
pip install -r requirements_vggt.txt

# 创建模型目录
mkdir -p models/vggt
```

## 使用方法

### 基本工作流

1. **加载模型**: 使用 `VGGT 模型加载器` 节点
   - 选择设备（auto/cuda/cpu）
   - 选择模型版本（VGGT-1B）

2. **输入图像**: 连接图像序列到 `VGGT 多输入重建` 节点
   - 支持1-200帧图像
   - 自动预处理到336x336分辨率

3. **配置参数**:
   - `confidence_threshold`: 置信度阈值（0-100%）
   - `show_cameras`: 是否显示相机位置
   - `mask_*`: 各种遮罩选项

4. **获取结果**:
   - `raw_result`: 原生VGGT推理结果（包含所有Tensor）
   - `intrinsics_json`: 相机内参JSON
   - `trajectory_preview`: 轨迹预览图像
   - `poses_json`: 相机姿态JSON
   - `model_3d_path`: 3D模型文件路径

### 节点说明

#### VGGT 模型加载器
- **输入**: 设备选择、模型版本
- **输出**: VGGT模型实例
- **功能**: 自动下载和缓存模型权重

#### VGGT 多输入重建
- **输入**: VGGT模型、图像序列、配置参数
- **输出**: 原生结果、JSON数据、预览图像、3D模型
- **功能**: 执行完整的VGGT推理流程

## 模型管理

### 支持的模型

| 模型名称 | 参数量 | 文件大小 | HuggingFace Hub |
|---------|--------|----------|-----------------|
| VGGT-1B | 1B     | ~4.7GB   | facebook/VGGT-1B |

### 模型存储

模型权重存储在 `models/vggt/` 目录：
- 自动下载：首次使用时从HuggingFace Hub下载
- 本地缓存：下载后自动缓存，避免重复下载
- 手动放置：可手动将 `vggt_1b.pt` 放入该目录

## 技术架构

### 核心组件

```
ComfyUI Framework
├── VVL_VideoCamera_Advanced/
│   ├── vggt/                    # VGGT原生代码
│   ├── vggt_native_interface.py # 原生接口封装
│   ├── vggt_model_loader.py     # 模型加载管理
│   ├── comfyui_vggt_nodes.py    # ComfyUI节点
│   └── visual_util.py           # 可视化工具
```

### 推理流程

1. **图像预处理**: 使用VGGT官方预处理函数
2. **特征聚合**: `model.aggregator(images)`
3. **多分支预测**:
   - 相机姿态: `model.camera_head()`
   - 深度图: `model.depth_head()`
   - 点云: `model.point_head()`
   - 轨迹跟踪: `model.track_head()` (可选)
4. **结果后处理**: 保留原生Tensor + 生成可视化

## 性能优化

### 内存使用

| 输入帧数 | GPU内存 | 推理时间 |
|---------|---------|----------|
| 1       | 1.88GB  | 0.04s    |
| 10      | 3.63GB  | 0.14s    |
| 50      | 11.41GB | 1.04s    |
| 100     | 21.15GB | 3.12s    |
| 200     | 40.63GB | 8.75s    |

### 优化建议

- 使用Flash Attention 3（如果可用）
- 启用混合精度推理（FP16）
- 根据GPU内存调整输入帧数
- 使用CUDA设备获得最佳性能

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件到 `models/vggt/`

2. **内存不足**
   - 减少输入图像数量
   - 使用CPU推理
   - 关闭其他GPU程序

3. **依赖缺失**
   - 运行 `python install.py` 重新安装
   - 检查Python环境和包版本

### 日志调试

启用详细日志：
```python
import logging
logging.getLogger('vvl_vggt_nodes').setLevel(logging.DEBUG)
```

## 开发说明

### 扩展节点

参考 `vggt.md` 技术规范文档，可以基于原生接口开发新的专用节点：
- 深度估计专用节点
- 相机姿态估计专用节点
- 点跟踪专用节点

### 贡献指南

1. 遵循技术规范文档
2. 保持原生VGGT API兼容性
3. 添加适当的错误处理和日志
4. 更新文档和测试

## 许可证

本项目遵循原始VGGT项目的许可证条款。

## 致谢

- [Facebook Research VGGT](https://github.com/facebookresearch/vggt)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- VGGT论文作者和贡献者 