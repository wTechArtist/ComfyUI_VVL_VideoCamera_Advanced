# ComfyUI VVL VideoCamera Advanced - 环境安装指南

这是一个集成了VGGT（Visual Geometry Grounded Transformer）和GLB点云处理功能的ComfyUI自定义节点包。

## 安装

### 方法1: 自动安装（推荐）

```bash
cd custom_nodes/ComfyUI_VVL_VideoCamera_Advanced
python install.py
```

### 方法2: 手动安装

```bash
# 1. 安装VGGT专用依赖
pip install -r requirements_vggt.txt

# 2. 创建模型目录
mkdir -p models/vggt

# 3. 安装GLB处理依赖
pip install trimesh scikit-learn
```

## 依赖要求

### VGGT核心依赖
```
torch>=2.0.0
torchvision
transformers
accelerate
huggingface_hub
numpy
pillow
```

### GLB处理依赖
```
trimesh              # GLB文件读写（必需）
scikit-learn         # 离群点检测（可选）
numpy               # 数值计算
```

## 验证安装

运行测试脚本验证安装是否成功：

```bash
python test_vggt_integration.py
```

如果看到 "🎉 所有测试通过！" 消息，说明安装成功。

## 故障排除

### 常见问题

1. **"trimesh库不可用"错误**
   ```bash
   pip install trimesh
   ```

2. **"No module named 'vggt'"错误**
   - 确保所有 `__init__.py` 文件存在
   - 重新运行 `python install.py`

3. **模型下载失败**
   - 检查网络连接
   - 模型会自动从HuggingFace下载到 `models/vggt/` 目录

4. **内存不足**
   - 减少输入图像数量
   - 使用CPU推理
   - 关闭其他GPU程序

### 模型存储

模型权重自动存储在 `ComfyUI/models/vggt/` 目录：
- 首次使用时自动从HuggingFace Hub下载
- 本地缓存避免重复下载
- 可手动放置模型文件到该目录

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（推荐，CPU也支持）
- 至少8GB内存（GPU推理需要更多显存）

安装完成后即可在ComfyUI中使用VGGT和GLB处理节点。 