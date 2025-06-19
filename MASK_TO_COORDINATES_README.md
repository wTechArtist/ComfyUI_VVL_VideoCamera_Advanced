# VVL Mask转坐标节点

## 功能描述

`MaskToCoordinates` 节点可以从输入的 mask 图像中自动提取坐标信息，生成符合 SAM2 分割节点要求的 `coordinates_positive` 和 `coordinates_negative` 参数。

- **白色区域** (值 >= threshold) → `coordinates_positive` (正样本点)
- **黑色区域** (值 < threshold) → `coordinates_negative` (负样本点)

## 输入参数

### 必需参数
- **mask** (MASK): 输入的 mask 图像
- **sample_method** (选择): 采样方法
  - `random`: 随机采样
  - `grid`: 网格采样  
  - `contour`: 轮廓采样
- **positive_points** (INT): 正样本点数量 (1-1000, 默认: 10)
- **negative_points** (INT): 负样本点数量 (0-1000, 默认: 10)
- **threshold** (FLOAT): 二值化阈值 (0.0-1.0, 默认: 0.5)

### 可选参数
- **min_distance** (INT): 点之间最小距离 (1-100, 默认: 10)
- **edge_margin** (INT): 边缘留白像素 (0-50, 默认: 5)

## 输出结果

- **coordinates_positive** (STRING): 正坐标 JSON 字符串，格式: `[{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]`
- **coordinates_negative** (STRING): 负坐标 JSON 字符串，格式同上
- **visualization** (IMAGE): 可视化图像，显示采样点位置
  - 绿色圆点: 正样本点
  - 红色圆点: 负样本点

## 使用示例

### 与 SAM2 分割节点连接

```
[Mask输入] → [MaskToCoordinates] → [Sam2Segmentation]
                    ↓
               coordinates_positive
               coordinates_negative
```

### 典型工作流程

1. 准备或生成 mask 图像
2. 连接到 `MaskToCoordinates` 节点
3. 调整采样参数 (点数量、采样方法等)
4. 将生成的坐标输出连接到 `Sam2Segmentation` 节点的对应输入
5. 查看可视化输出确认采样效果

## 采样方法说明

### Random (随机采样)
- 在有效区域内随机选择点
- 支持最小距离约束，避免点过于密集
- 适用于大多数情况

### Grid (网格采样)  
- 按网格模式均匀分布采样点
- 确保点分布均匀
- 适用于需要规整分布的场景

### Contour (轮廓采样)
- 沿着区域轮廓边缘采样
- 优先选择最大轮廓
- 适用于边界敏感的分割任务

## 技术细节

### 坐标格式
生成的坐标完全兼容 SAM2 分割节点，格式为:
```json
[
    {"x": 100, "y": 150},
    {"x": 200, "y": 250},
    ...
]
```

### 边缘处理
- `edge_margin` 参数可设置边缘留白，避免在图像边缘采样
- 防止分割时边界效应

### 性能优化
- 支持批处理 (自动处理第一帧)
- 内存高效的采样算法
- 快速的轮廓检测

## 注意事项

1. **Mask 格式**: 支持标准的 ComfyUI MASK 格式
2. **坐标范围**: 生成的坐标在 mask 图像尺寸范围内
3. **点数量**: 实际生成的点数可能少于设定值（当有效区域不足时）
4. **阈值设置**: 建议根据具体 mask 特征调整 threshold 值

## 故障排除

- **生成点数不足**: 检查 mask 是否有足够的有效区域
- **点分布不理想**: 尝试不同的采样方法或调整参数
- **边缘点过多**: 增大 `edge_margin` 值
- **坐标格式错误**: 确保输出连接到正确的 SAM2 节点输入

## 版本信息

- 版本: 1.0.0
- 兼容: ComfyUI SAM2 节点
- 依赖: PyTorch, OpenCV, NumPy 