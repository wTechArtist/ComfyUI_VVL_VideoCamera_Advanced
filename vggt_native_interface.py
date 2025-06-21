"""
VGGT Native Interface
原生VGGT功能接口封装 - 基于官方API
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger('vggt_native_interface')

class VGGTNativeInterface:
    """VGGT原生功能接口封装 - 基于官方API"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model  # 原生VGGT模型实例
        self.device = device
        
    def full_inference_pipeline(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict:
        """完整的VGGT推理管道 - 基于官方最新API"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 确保输入是tensor格式，并添加batch维度如果需要
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).to(self.device)
                
                if images.dim() == 4:  # (N, C, H, W)
                    images = images[None]  # (1, N, C, H, W)
                
                # 使用官方的forward方法，获得标准的predictions字典
                predictions = self.model.forward(images, query_points)
                
                # 添加原始图像数据（官方predictions_to_glb需要）
                # 转换图像格式：从(1, N, C, H, W)到(N, H, W, C)
                if images.dim() == 5:  # (1, N, C, H, W)
                    images_formatted = images.squeeze(0).permute(0, 2, 3, 1)  # (N, H, W, C)
                else:  # (N, C, H, W)
                    images_formatted = images.permute(0, 2, 3, 1)  # (N, H, W, C)
                predictions['images'] = images_formatted
                
                # 直接返回标准的VGGT输出格式，添加相机解码结果
                if 'pose_enc' in predictions:
                    try:
                        extrinsic, intrinsic = self._decode_camera_params(
                            predictions['pose_enc'], images.shape[-2:]
                        )
                        predictions['cameras'] = {'extrinsic': extrinsic, 'intrinsic': intrinsic}
                    except Exception as e:
                        logger.warning(f"Failed to decode camera parameters: {e}")
                
                # 从深度图构建3D点云（如果可能）
                if 'depth' in predictions and 'cameras' in predictions:
                    try:
                        point_map_unprojected = self._unproject_depth_to_points(
                            predictions['depth'].squeeze(0), 
                            predictions['cameras']['extrinsic'].squeeze(0), 
                            predictions['cameras']['intrinsic'].squeeze(0)
                        )
                        predictions['points_from_depth'] = point_map_unprojected
                    except Exception as e:
                        logger.warning(f"Failed to unproject depth to points: {e}")
                
                return predictions
    
    def selective_inference(self, images: torch.Tensor, branches: List[str], **kwargs) -> Dict:
        """选择性推理：只执行指定的分支"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 添加batch维度如果需要
                if images.dim() == 4:  # (N, C, H, W)
                    images = images[None]  # (1, N, C, H, W)
                
                # 1. 特征聚合 (必须步骤)
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                
                results = {}
                
                # 2. 根据需要执行不同分支
                if 'cameras' in branches:
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = self._decode_camera_params(pose_enc, images.shape[-2:])
                    results['cameras'] = {'extrinsic': extrinsic, 'intrinsic': intrinsic}
                
                if 'depth' in branches:
                    depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                    results['depth'] = {'depth_map': depth_map, 'confidence': depth_conf}
                
                if 'points' in branches:
                    point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                    results['points'] = {'point_map': point_map, 'confidence': point_conf}
                
                if 'points_from_depth' in branches and 'cameras' in results and 'depth' in results:
                    point_map_unprojected = self._unproject_depth_to_points(
                        results['depth']['depth_map'].squeeze(0), 
                        results['cameras']['extrinsic'].squeeze(0), 
                        results['cameras']['intrinsic'].squeeze(0)
                    )
                    results['points_from_depth'] = point_map_unprojected
                
                if 'tracks' in branches and 'query_points' in kwargs:
                    track_list, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images, ps_idx, 
                        query_points=kwargs['query_points'][None]
                    )
                    results['tracks'] = {
                        'track_list': track_list,
                        'visibility_score': vis_score,
                        'confidence_score': conf_score
                    }
                
                return results
    
    def _decode_camera_params(self, pose_enc: torch.Tensor, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码相机参数 - 使用VGGT官方工具函数"""
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            return pose_encoding_to_extri_intri(pose_enc, image_shape)
        except ImportError:
            logger.error("VGGT pose_enc utils not available")
            raise
    
    def _unproject_depth_to_points(self, depth_map: torch.Tensor, extrinsic: torch.Tensor, 
                                  intrinsic: torch.Tensor) -> torch.Tensor:
        """从深度图反投影生成3D点 - 使用VGGT官方工具函数"""
        try:
            from vggt.utils.geometry import unproject_depth_map_to_point_map
            return unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        except ImportError:
            logger.error("VGGT geometry utils not available")
            raise


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


class VGGTResultProcessor:
    """VGGT结果后处理器"""
    
    @staticmethod
    def format_camera_results(extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> Dict:
        """相机参数结果格式化，同时保留原生Tensor输出"""
        # 确保输入是tensor格式
        if isinstance(extrinsic, np.ndarray):
            extrinsic = torch.from_numpy(extrinsic)
        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic)
            
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
        """深度结果格式化，同时保留原生Tensor输出"""
        # 确保输入是tensor格式
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if isinstance(confidence, np.ndarray):
            confidence = torch.from_numpy(confidence)
            
        return {
            "raw": {
                "depth_map": depth_map,
                "confidence": confidence,
            },
            "json": {
                "depth_shape": list(depth_map.shape),
                "depth_range": [float(depth_map.min()), float(depth_map.max())],
                "confidence_range": [float(confidence.min()), float(confidence.max())],
            }
        }
        
    @staticmethod
    def format_point_cloud(points: torch.Tensor, colors: torch.Tensor = None, confidence: torch.Tensor = None) -> Dict:
        """点云结果格式化，同时保留原生Tensor输出"""
        # 确保输入是tensor格式
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        if colors is not None and isinstance(colors, np.ndarray):
            colors = torch.from_numpy(colors)
        if confidence is not None and isinstance(confidence, np.ndarray):
            confidence = torch.from_numpy(confidence)
            
        result = {
            "raw": {
                "points": points,
                "colors": colors,
                "confidence": confidence,
            },
            "json": {
                "num_points": int(points.shape[0]) if points.dim() > 1 else int(points.numel() // 3),
                "point_shape": list(points.shape),
                "bounds": {
                    "min": points.min(dim=0)[0].cpu().numpy().tolist() if points.numel() > 0 else [0, 0, 0],
                    "max": points.max(dim=0)[0].cpu().numpy().tolist() if points.numel() > 0 else [0, 0, 0],
                }
            }
        }
        
        if colors is not None:
            result["json"]["has_colors"] = True
            result["json"]["color_shape"] = list(colors.shape)
        
        if confidence is not None:
            result["json"]["has_confidence"] = True
            result["json"]["confidence_range"] = [float(confidence.min()), float(confidence.max())]
        
        return result 