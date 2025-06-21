# VGGT ComfyUI集成技术规范文档

## 目录
1. [项目概述](#项目概述)
2. [技术架构](#技术架构)
3. [核心模块规范](#核心模块规范)
4. [API接口设计](#api接口设计)
5. [数据流规范](#数据流规范)
6. [节点实现规范](#节点实现规范)
7. [性能优化规范](#性能优化规范)
8. [错误处理规范](#错误处理规范)
9. [测试验证规范](#测试验证规范)
10. [部署集成规范](#部署集成规范)

## 项目概述

### 1.1 项目目标
在ComfyUI框架中集成Facebook Research的VGGT（Visual Geometry Grounded Transformer）模型，实现原生的多视图3D重建、相机姿态估计、深度预测和轨迹跟踪功能。

### 1.2 核心功能需求
- **相机姿态估计**: 从图像序列自动估计相机内外参数
- **深度图预测**: 生成高质量的密集深度图
- **点云重建**: 基于深度和姿态信息重建3D点云
- **轨迹跟踪**: 跟踪指定点在图像序列中的运动
- **单视图重建**: 支持从单张图像进行3D重建
- **多视图重建**: 支持从视频或图像序列进行3D重建

### 1.3 技术约束
- 兼容ComfyUI工作流系统
- 支持CUDA和CPU推理
- 内存使用优化（支持1-200帧输入）
- 模型权重管理和缓存机制
- 支持批处理和实时预览

## 技术架构

### 2.1 系统架构图
```
ComfyUI Framework
├── VVL_VideoCamera_Advanced/
│   ├── vggt/                    # VGGT原生代码
│   │   ├── models/             # 模型定义
│   │   ├── utils/              # 工具函数
│   │   ├── heads/              # 预测头
│   │   └── layers/             # 网络层
│   ├── comfyui_vggt_nodes.py   # ComfyUI节点封装
│   ├── vggt_model_loader.py    # 模型加载管理
│   └── visual_util.py          # 可视化工具
```

### 2.2 模块依赖关系
```
VGGT原生模块 → ComfyUI适配层 → 用户界面节点
    ↓              ↓              ↓
  推理引擎      数据转换        工作流集成
    ↓              ↓              ↓
  结果输出      格式标准化      可视化展示
```

## 核心模块规范

### 3.1 VGGT模型集成规范

#### 3.1.1 模型加载器 (vggt_model_loader.py)
```python
class VGGTModelManager:
    """VGGT模型管理器，负责模型的加载、缓存和设备管理"""
    
    def __init__(self):
        self.model_cache = {}
        self.device_manager = DeviceManager()
        
    def load_model(self, model_name: str, device: str) -> torch.nn.Module:
        """加载VGGT模型"""
        """
        1. 检查缓存避免重复加载
        2. 自动从 HuggingFace Hub 下载权重
        3. 尝试启用 flash-attn 进行加速
        4. 捕获 CUDA OOM 并自动降级
        """
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        from vggt.models.vggt import VGGT
        from huggingface_hub import hf_hub_download
        import torch, os

        device = self.device_manager.get(device)
        try:
            weight_path = hf_hub_download(
                repo_id=f"facebook/{model_name}",
                filename="model.pt",
                local_dir=get_vggt_model_dir(),  # 确保权重保存在 ComfyUI/models/vggt
                cache_dir=get_vggt_model_dir(),
            )
        except Exception as e:
            raise VGGTModelNotFound(f"无法下载模型权重: {e}")

        model = VGGT()
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.to(device).eval()

        # 尝试启用 Flash-Attention
        try:
            model.enable_flash_attn()
        except Exception:
            pass

        self.model_cache[model_name] = model
        return model
        
    def get_model_info(self) -> Dict:
        """获取模型信息和状态"""
        pass
        
    def clear_cache(self):
        """清理模型缓存"""
        pass
```

#### 3.1.2 原生VGGT封装规范
```python
class VGGTNativeInterface:
    """VGGT原生功能接口封装 - 基于官方API"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model  # 原生VGGT模型实例
        self.device = device
        
    def full_inference_pipeline(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict:
        """完整的VGGT推理管道 - 遵循官方使用方式"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 添加batch维度如果需要
                if images.dim() == 4:  # (N, C, H, W)
                    images = images[None]  # (1, N, C, H, W)
                
                # 1. 特征聚合 (核心步骤)
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                
                results = {}
                
                # 2. 相机姿态预测
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = self._decode_camera_params(pose_enc, images.shape[-2:])
                results['cameras'] = {'extrinsic': extrinsic, 'intrinsic': intrinsic}
                
                # 3. 深度预测
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                results['depth'] = {'depth_map': depth_map, 'confidence': depth_conf}
                
                # 4. 点云预测
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                results['points'] = {'point_map': point_map, 'confidence': point_conf}
                
                # 5. 从深度图构建3D点 (通常更准确)
                point_map_unprojected = self._unproject_depth_to_points(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )
                results['points_from_depth'] = point_map_unprojected
                
                # 6. 可选：点跟踪
                if query_points is not None:
                    track_list, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images, ps_idx, query_points=query_points[None]
                    )
                    results['tracks'] = {
                        'track_list': track_list,
                        'visibility_score': vis_score,
                        'confidence_score': conf_score
                    }
                
                return results
    
    def _decode_camera_params(self, pose_enc: torch.Tensor, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码相机参数 - 使用VGGT官方工具函数"""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        return pose_encoding_to_extri_intri(pose_enc, image_shape)
    
    def _unproject_depth_to_points(self, depth_map: torch.Tensor, extrinsic: torch.Tensor, 
                                  intrinsic: torch.Tensor) -> torch.Tensor:
        """从深度图反投影生成3D点 - 使用VGGT官方工具函数"""
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        return unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
```

### 3.2 数据预处理规范

#### 3.2.1 图像预处理标准
```python
class VGGTImageProcessor:
    """VGGT图像预处理器 - 基于官方load_and_preprocess_images函数"""
    
    @staticmethod
    def preprocess_images(images: List[np.ndarray], target_size: Tuple[int, int] = (336, 336)) -> torch.Tensor:
        """
        图像预处理规范 - 遵循VGGT官方标准：
        1. 尺寸调整到336x336（VGGT标准输入尺寸）
        2. 归一化到[0,1]范围
        3. 转换为torch.Tensor格式
        4. 通道顺序RGB
        5. 支持masking（设置像素值为0或1来屏蔽不需要的区域）
        """
        try:
            # 使用VGGT官方预处理函数
            from vggt.utils.load_fn import load_and_preprocess_images
            
            # 如果输入是numpy数组列表，需要先保存为临时文件
            import tempfile
            import os
            from PIL import Image
            
            temp_paths = []
            temp_dir = tempfile.mkdtemp()
            
            try:
                # 保存图像到临时文件
                for i, img_array in enumerate(images):
                    if isinstance(img_array, np.ndarray):
                        # 确保是RGB格式
                        if img_array.shape[-1] == 3:
                            img_pil = Image.fromarray(img_array.astype(np.uint8))
                        else:
                            img_pil = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
                    else:
                        img_pil = img_array
                    
                    temp_path = os.path.join(temp_dir, f"temp_{i:04d}.jpg")
                    img_pil.save(temp_path)
                    temp_paths.append(temp_path)
                
                # 使用官方预处理函数
                processed_images = load_and_preprocess_images(temp_paths)
                
                return processed_images
                
            finally:
                # 清理临时文件
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                os.rmdir(temp_dir)
                
        except ImportError:
            # 如果官方函数不可用，使用备用方案
            return VGGTImageProcessor._fallback_preprocess(images, target_size)
    
    @staticmethod
    def _fallback_preprocess(images: List[np.ndarray], target_size: Tuple[int, int]) -> torch.Tensor:
        """备用图像预处理方案"""
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),  # 自动归一化到[0,1]
        ])
        
        processed = []
        for img_array in images:
            if isinstance(img_array, np.ndarray):
                # 转换为PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array).convert('RGB')
            else:
                pil_image = img_array
            
            tensor_img = transform(pil_image)
            processed.append(tensor_img)
        
        return torch.stack(processed)
    
    @staticmethod
    def apply_masking(images: torch.Tensor, mask_type: str = None) -> torch.Tensor:
        """
        应用遮罩 - VGGT支持简单的像素遮罩
        mask_type: 'black_bg', 'white_bg', 'sky' 等
        """
        if mask_type is None:
            return images
        
        masked_images = images.clone()
        
        if mask_type == 'black_bg':
            # 将黑色像素（接近0的像素）设置为0
            black_mask = (images.sum(dim=1, keepdim=True) < 0.1)  # RGB总和小于0.1
            masked_images[black_mask.expand_as(images)] = 0.0
        
        elif mask_type == 'white_bg':
            # 将白色像素（接近1的像素）设置为1
            white_mask = (images.sum(dim=1, keepdim=True) > 2.7)  # RGB总和大于2.7
            masked_images[white_mask.expand_as(images)] = 1.0
        
        # 可以添加更多遮罩类型
        
        return masked_images
```

#### 3.2.2 数据格式转换规范
```python
class VGGTDataConverter:
    """VGGT数据格式转换器"""
    
    @staticmethod
    def comfyui_to_vggt_format(comfyui_data: Any) -> torch.Tensor:
        """ComfyUI数据格式转VGGT格式"""
        pass
        
    @staticmethod  
    def vggt_to_comfyui_format(vggt_data: torch.Tensor) -> Any:
        """VGGT格式转ComfyUI格式"""
        pass
```

### 3.3 后处理规范

#### 3.3.1 结果格式化
```python
class VGGTResultProcessor:
    """VGGT结果后处理器"""
    
    @staticmethod
    def format_camera_results(extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> Dict:
        """相机参数结果格式化，同时保留原生Tensor输出"""
        return {
            "raw": {
                "extrinsic": extrinsic,   # 原生Tensor
                "intrinsic": intrinsic,
            },
            "json": {
                "extrinsic_matrices": extrinsic.cpu().numpy().tolist(),
                "intrinsic_matrices": intrinsic.cpu().numpy().tolist(),
                "format": "opencv_convention",
                "coordinate_system": "camera_from_world",
            }
        }
        
    @staticmethod
    def format_depth_results(depth_map: torch.Tensor, confidence: torch.Tensor) -> Dict:
        """深度结果格式化"""
        pass
        
    @staticmethod
    def format_point_cloud(points: torch.Tensor, colors: torch.Tensor, confidence: torch.Tensor) -> Dict:
        """点云结果格式化"""
        pass
```

## API接口设计

### 4.1 核心推理接口

#### 4.1.1 统一推理接口
```python
class VGGTInferenceEngine:
    """VGGT推理引擎"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    def full_reconstruction(self, images: torch.Tensor, **kwargs) -> Dict:
        """完整重建流程：相机+深度+点云+跟踪"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 1. 特征聚合
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                
                # 2. 相机预测
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                
                # 3. 深度预测
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                
                # 4. 点云预测
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                
                # 5. 可选：点跟踪
                track_results = None
                if 'query_points' in kwargs:
                    track_list, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images, ps_idx, 
                        query_points=kwargs['query_points']
                    )
                    track_results = {
                        'tracks': track_list,
                        'visibility': vis_score,
                        'confidence': conf_score
                    }
                
                return {
                    'cameras': {'extrinsic': extrinsic, 'intrinsic': intrinsic},
                    'depth': {'depth_map': depth_map, 'confidence': depth_conf},
                    'points': {'point_map': point_map, 'confidence': point_conf},
                    'tracks': track_results
                }
    
    def selective_reconstruction(self, images: torch.Tensor, branches: List[str], **kwargs) -> Dict:
        """选择性重建：只执行指定的分支"""
        pass
```

### 4.2 ComfyUI节点接口规范

#### 4.2.1 主要节点类型
```python
# 1. 模型加载节点
class VVLVGGTLoader:
    INPUT_TYPES = {
        "required": {
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            "vggt_model": (["VGGT-1B"], {"default": "VGGT-1B"}),
        }
    }
    RETURN_TYPES = ("VVL_VGGT_MODEL",)
    
# 2. 多输入重建节点  
class VGGTMultiInputNode:
    INPUT_TYPES = {
        "required": {
            "vggt_model": ("VVL_VGGT_MODEL",),
            "images": ("IMAGE",),
        },
        "optional": {
            "confidence_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0}),
            "show_cameras": ("BOOLEAN", {"default": True}),
            "mask_black_bg": ("BOOLEAN", {"default": False}),
            "mask_white_bg": ("BOOLEAN", {"default": False}),
            "mask_sky": ("BOOLEAN", {"default": False}),
        }
    }
    # 返回同时包含原生和格式化内容，便于后续节点灵活使用
    RETURN_TYPES = (
        "RAW_VGGT_RESULT",   # 原生VGGT字典，包含Tensor等
        "STRING",            # 相机intrinsic JSON
        "IMAGE",             # 轨迹可视化预览
        "STRING",            # 相机姿态JSON
        "STRING",            # 点云/模型文件路径
    )
    RETURN_NAMES = (
        "raw_result",
        "intrinsics_json",
        "trajectory_preview",
        "poses_json",
        "model_3d_path",
    )

# 3. 单功能专用节点
class VGGTDepthEstimation:
    """深度估计专用节点"""
    pass
    
class VGGTCameraPoseEstimation:
    """相机姿态估计专用节点"""
    pass
    
class VGGTPointTracking:
    """点跟踪专用节点"""
    pass
```

#### 4.2.2 节点注册（ComfyUI入口）
```python
NODE_CLASS_MAPPINGS = {
    "VVLVGGTLoader": VVLVGGTLoader,
    "VGGTMultiInputNode": VGGTMultiInputNode,
    "VGGTDepthEstimation": VGGTDepthEstimation,
    "VGGTCameraPoseEstimation": VGGTCameraPoseEstimation,
    "VGGTPointTracking": VGGTPointTracking,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VVLVGGTLoader": "VGGT 模型加载器",
    "VGGTMultiInputNode": "VGGT 多输入重建",
    "VGGTDepthEstimation": "VGGT 深度估计",
    "VGGTCameraPoseEstimation": "VGGT 相机姿态估计",
    "VGGTPointTracking": "VGGT 点跟踪",
}
```

## 数据流规范

### 5.1 输入数据流
```
用户输入 → 数据验证 → 格式转换 → 预处理 → VGGT推理
   ↓           ↓          ↓         ↓         ↓
图像序列   类型检查    标准格式    尺寸调整   模型推理
```

### 5.2 输出数据流  
```
VGGT输出 → 后处理 → 格式化 → 可视化 → ComfyUI输出
   ↓        ↓       ↓       ↓        ↓
原始结果   滤波/增强  JSON格式  预览图    节点输出
```

### 5.3 数据类型定义
```python
# 输入数据类型
InputVideo = Union[str, bytes, torch.Tensor]  # 视频路径或数据
InputImages = torch.Tensor  # 形状: (N, H, W, 3)
QueryPoints = torch.Tensor  # 形状: (N, 2)

# 输出数据类型
CameraParams = Dict[str, torch.Tensor]  # 相机参数
DepthMap = torch.Tensor  # 深度图
PointCloud = Dict[str, torch.Tensor]  # 点云数据
TrackResults = Dict[str, torch.Tensor]  # 跟踪结果
```

## 节点实现规范

### 6.1 节点基类规范
```python
class VGGTBaseNode:
    """VGGT节点基类，提供通用功能"""
    
    CATEGORY = "💃VVL/VGGT"
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def validate_inputs(self, **kwargs):
        """输入验证"""
        pass
        
    def handle_errors(self, func):
        """错误处理装饰器"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Node execution failed: {e}")
                raise
        return wrapper
        
    def log_performance(self, func):
        """性能监控装饰器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            self.logger.info(f"Execution time: {elapsed:.2f}s")
            return result
        return wrapper
```

### 6.2 错误处理规范
```python
class VGGTError(Exception):
    """VGGT相关错误基类"""
    pass

class VGGTModelNotFound(VGGTError):
    """模型未找到错误"""
    pass

class VGGTInsufficientMemory(VGGTError):
    """内存不足错误"""
    pass

class VGGTInvalidInput(VGGTError):
    """无效输入错误"""
    pass

# 错误处理示例
def safe_model_inference(self, *args, **kwargs):
    try:
        return self.model_inference(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        raise VGGTInsufficientMemory("GPU内存不足，请减少输入帧数或使用CPU")
    except Exception as e:
        raise VGGTError(f"模型推理失败: {str(e)}")
```

## 性能优化规范

### 7.1 内存优化
```python
class VGGTMemoryManager:
    """VGGT内存管理器"""
    
    @staticmethod
    def estimate_memory_usage(num_frames: int, image_size: Tuple[int, int]) -> float:
        """估算内存使用量（GB）"""
        # 基于VGGT基准数据的内存估算
        base_memory = 1.88  # 1帧的基准内存
        scale_factor = 0.2  # 每增加一帧的内存增量
        return base_memory + (num_frames - 1) * scale_factor
        
    @staticmethod
    def optimize_batch_size(available_memory: float, num_frames: int) -> int:
        """根据可用内存优化批处理大小"""
        pass
        
    @staticmethod
    def enable_gradient_checkpointing(model: torch.nn.Module):
        """启用梯度检查点以节省内存"""
        pass
```

### 7.2 推理优化
```python
class VGGTInferenceOptimizer:
    """VGGT推理优化器"""
    
    @staticmethod
    def enable_mixed_precision():
        """启用混合精度推理"""
        return torch.cuda.amp.autocast(dtype=torch.float16)
        
    @staticmethod
    def optimize_for_inference(model: torch.nn.Module):
        """模型推理优化"""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return torch.jit.script(model)  # 可选的JIT编译
        
    @staticmethod
    def batch_processing(images: torch.Tensor, batch_size: int = 4):
        """批处理优化"""
        pass
```

## 测试验证规范

### 8.1 单元测试规范
```python
import unittest
import torch

class TestVGGTIntegration(unittest.TestCase):
    """VGGT集成测试"""
    
    def setUp(self):
        """测试环境设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_vggt_model("VGGT-1B", self.device)
        
    def test_model_loading(self):
        """测试模型加载"""
        self.assertIsNotNone(self.model)
        
    def test_single_image_inference(self):
        """测试单图像推理"""
        dummy_image = torch.randn(1, 3, 336, 336).to(self.device)
        result = self.model.full_reconstruction(dummy_image)
        self.assertIn('cameras', result)
        self.assertIn('depth', result)
        
    def test_multi_image_inference(self):
        """测试多图像推理"""
        dummy_images = torch.randn(4, 3, 336, 336).to(self.device)
        result = self.model.full_reconstruction(dummy_images)
        self.assertEqual(result['cameras']['extrinsic'].shape[0], 4)
        
    def test_memory_usage(self):
        """测试内存使用"""
        initial_memory = torch.cuda.memory_allocated()
        dummy_images = torch.randn(10, 3, 336, 336).to(self.device)
        result = self.model.full_reconstruction(dummy_images)
        peak_memory = torch.cuda.max_memory_allocated()
        self.assertLess(peak_memory - initial_memory, 12 * 1024**3)  # <12GB
```

### 8.2 集成测试规范
- 与ComfyUI工作流的兼容性测试
- 不同输入格式的处理测试  
- 错误恢复能力测试
- 性能基准测试

## 部署集成规范

### 9.1 依赖管理
```txt
# requirements.txt - 核心依赖 (已存在)
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
huggingface_hub>=0.10.0
einops>=0.6.0
safetensors>=0.3.0

# requirements_vggt.txt - VGGT专用依赖 (需新建)
# 基于VGGT官方requirements.txt和requirements_demo.txt
trimesh>=3.15.0
scipy>=1.9.0
matplotlib>=3.5.0
gradio>=4.0.0
viser>=0.1.0
onnxruntime>=1.15.0
flash-attn>=2.4.2  # 推荐，支持Flash Attention 3
gsplat>=1.3.0  # 可选，用于Gaussian Splatting集成
omegaconf>=2.3.0
hydra-core>=1.3.0
open3d>=0.17.0  # 用于3D可视化
ttach>=0.0.4
pytorch3d>=0.7.6
pydantic<3  # VGGT 依赖 v1/v2 过渡接口
```

### 9.2 模型权重管理
```python
VGGT_MODELS = {
    "VGGT-1B": {
        "hf_model": "facebook/VGGT-1B",
        "local_file": "vggt_1b.pt", 
        "size_gb": 4.7,
        "checksum": "sha256:xxxxx"
    }
}

def download_and_verify_model(model_name: str):
    """下载并验证模型权重"""
    pass
```

### 9.3 安装配置脚本
```python
# install.py
def install_vggt_dependencies():
    """安装VGGT依赖"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_vggt.txt"])
    
def setup_model_directory():
    """设置模型目录"""
    pass
    
def verify_installation():
    """验证安装是否成功"""
    pass

if __name__ == "__main__":
    install_vggt_dependencies()
    setup_model_directory() 
    verify_installation()
    print("VGGT集成安装完成!")
```

## 代码实现清单

### 10.1 必需实现的文件

#### 核心文件
- [x] `vggt_model_loader.py` - 模型加载管理（已实现）
- [x] `comfyui_vggt_nodes.py` - ComfyUI节点（已实现）
- [ ] `vggt_native_interface.py` - 原生VGGT接口封装
- [ ] `vggt_data_processor.py` - 数据预处理和后处理
- [ ] `vggt_inference_engine.py` - 推理引擎
- [ ] `vggt_memory_manager.py` - 内存管理
- [ ] `vggt_error_handler.py` - 错误处理

#### VGGT原生模块
- [x] `vggt/` - VGGT原始代码目录（已存在）
- [ ] 需要确保所有VGGT原生功能可正常导入和使用

#### 辅助文件
- [x] `visual_util.py` - 可视化工具（已实现）
- [ ] `install.py` - 安装脚本
- [ ] `test_vggt_integration.py` - 集成测试
- [x] `requirements.txt` - 基础依赖（已实现）
- [ ] `requirements_vggt.txt` - VGGT专用依赖

### 10.2 实现优先级

#### 高优先级 (P0)
1. **VGGT原生模型集成**
   - 确保`from vggt.models.vggt import VGGT`可正常导入
   - 实现官方推理流程：`aggregator → camera_head → depth_head → point_head → track_head`