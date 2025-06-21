# VGGT Mask处理节点文件

import os
import json
import tempfile
from typing import List, Any, Dict
import logging
import struct

import cv2
import numpy as np
import torch
from PIL import Image

# 尝试导入 ComfyUI 的类型标记
try:
    from comfy.comfy_types import IO
except ImportError:
    class IO:
        VIDEO = "VIDEO"
        IMAGE = "IMAGE"

# 导入原生VGGT接口
try:
    from .vggt_native_interface import VGGTNativeInterface, VGGTImageProcessor, VGGTResultProcessor
    VGGT_NATIVE_AVAILABLE = True
except ImportError:
    VGGTNativeInterface = None
    VGGTImageProcessor = None
    VGGTResultProcessor = None
    VGGT_NATIVE_AVAILABLE = False

# 导入模型加载器
try:
    from .vggt_model_loader import VVLVGGTLoader
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    VVLVGGTLoader = None
    MODEL_LOADER_AVAILABLE = False

# 导入ComfyUI的路径管理
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
    FOLDER_PATHS_AVAILABLE = False

# 导入原节点的工具函数
try:
    from .comfyui_vggt_nodes import (
        _create_traj_preview, 
        _create_insufficient_data_image,
        predictions_to_glb,
        TRIMESH_AVAILABLE,
        MATPLOTLIB_AVAILABLE,
        SCIPY_AVAILABLE
    )
    VGGT_UTILS_AVAILABLE = True
except ImportError:
    _create_traj_preview = None
    _create_insufficient_data_image = None
    predictions_to_glb = None
    TRIMESH_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    SCIPY_AVAILABLE = False
    VGGT_UTILS_AVAILABLE = False

# 配置日志
logger = logging.getLogger('vvl_vggt_native_nodes')

# -----------------------------------------------------------------------------
# 原生VGGT推理函数
# -----------------------------------------------------------------------------

def run_vggt_full_native_inference(images: torch.Tensor, model_instance, device, 
                                  enable_tracks: bool = True, 
                                  grid_size: int = 8,
                                  query_region: float = 0.8) -> Dict:
    """使用原生VGGT接口进行完整推理"""
    if not VGGT_NATIVE_AVAILABLE:
        raise ImportError("VGGT Native Interface not available")
    
    try:
        # 创建原生接口
        vggt_interface = VGGTNativeInterface(model_instance, device)
        
        # 确定启用的分支
        branches = ['cameras', 'depth', 'points', 'points_from_depth']
        
        # 如果启用tracks，需要生成查询点
        query_points = None
        if enable_tracks:
            branches.append('tracks')
            # 自动生成查询点：在图像指定区域均匀采样点
            B, S, C, H, W = images.shape if images.dim() == 5 else (1, *images.shape)
            
            # 计算查询区域
            margin_h = int(H * (1 - query_region) / 2)
            margin_w = int(W * (1 - query_region) / 2)
            
            # 生成网格查询点
            y_coords = torch.linspace(margin_h, H - margin_h, grid_size)
            x_coords = torch.linspace(margin_w, W - margin_w, grid_size)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 生成查询点坐标 (N_points, 2)，格式为 (x, y)
            query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
            logger.info(f"Generated {len(query_points)} query points ({grid_size}x{grid_size} grid)")
        
        # 执行选择性推理以获取所有分支的输出
        results = vggt_interface.selective_inference(
            images, 
            branches=branches,
            query_points=query_points
        )
        
        # 添加原始图像到结果中
        if images.dim() == 5:  # (1, N, C, H, W)
            images_formatted = images.squeeze(0).permute(0, 2, 3, 1)  # (N, H, W, C)
        else:  # (N, C, H, W)
            images_formatted = images.permute(0, 2, 3, 1)  # (N, H, W, C)
        results['images'] = images_formatted.cpu().numpy()
        
        # 添加查询点信息
        if query_points is not None:
            results['query_points'] = query_points.cpu().numpy()
        
        logger.info(f"VGGT full native inference completed. Results keys: {list(results.keys())}")
        return results
        
    except Exception as e:
        logger.error(f"VGGT full native inference failed: {e}")
        raise

def preprocess_images_native(images_list: List[np.ndarray]) -> torch.Tensor:
    """使用原生VGGT预处理器处理图像"""
    if not VGGT_NATIVE_AVAILABLE:
        raise ImportError("VGGT Native Interface not available")
    
    try:
        return VGGTImageProcessor.preprocess_images(images_list)
    except Exception as e:
        logger.error(f"Native image preprocessing failed: {e}")
        raise

def apply_comprehensive_filtering(raw_results: Dict, confidence_threshold: float, 
                                mask_black_bg: bool, mask_white_bg: bool, mask_sky: bool) -> Dict:
    """对原生VGGT结果应用彻底的过滤"""
    logger.info(f"应用彻底过滤: confidence_threshold={confidence_threshold}")
    
    filtered_results = {}
    
    # 1. 首先处理图像，应用背景遮罩
    images = raw_results.get('images')
    if images is not None and (mask_black_bg or mask_white_bg or mask_sky):
        images = apply_image_masking(images, mask_black_bg, mask_white_bg, mask_sky)
    filtered_results['images'] = images
    
    # 2. 获取或创建置信度mask
    confidence_mask = create_confidence_mask(raw_results, confidence_threshold)
    
    # 3. 过滤所有点云数据
    for key in ['points_from_depth', 'points']:
        if key in raw_results:
            filtered_results[key] = apply_confidence_filtering(raw_results[key], confidence_mask)
    
    # 4. 过滤深度数据
    if 'depth' in raw_results:
        filtered_results['depth'] = apply_confidence_filtering(raw_results['depth'], confidence_mask)
    
    # 5. 其他数据直接复制
    for key in ['cameras', 'tracks', 'query_points']:
        if key in raw_results:
            filtered_results[key] = raw_results[key]
    
    # 6. 保存过滤参数到结果中
    filtered_results['_filtering_params'] = {
        'confidence_threshold': confidence_threshold,
        'mask_black_bg': mask_black_bg,
        'mask_white_bg': mask_white_bg,
        'mask_sky': mask_sky
    }
    
    logger.info("彻底过滤完成，所有点云数据已应用过滤参数")
    return filtered_results

def create_confidence_mask(raw_results: Dict, confidence_threshold: float) -> np.ndarray:
    """创建置信度mask，用于过滤低置信度点"""
    if confidence_threshold <= 0 or confidence_threshold >= 100:
        return None
    
    # 尝试从depth confidence获取
    if 'depth' in raw_results:
        depth_data = raw_results['depth']
        if isinstance(depth_data, dict) and 'confidence' in depth_data:
            conf = depth_data['confidence']
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            
            # 去掉batch维度
            if conf.ndim == 4 and conf.shape[0] == 1:
                conf = np.squeeze(conf, axis=0)
                
            threshold_val = confidence_threshold / 100.0
            conf_mask = conf >= threshold_val
            logger.info(f"从depth confidence创建mask，形状: {conf_mask.shape}")
            return conf_mask
    
    # 如果没有confidence数据，基于点云分布创建mask
    points_data = raw_results.get('points_from_depth')
    if points_data is None:
        points_data = raw_results.get('points')
    if points_data is not None:
        if isinstance(points_data, dict):
            points_data = points_data.get('point_map', points_data)
        
        if isinstance(points_data, torch.Tensor):
            points_np = points_data.cpu().numpy()
        else:
            points_np = points_data
        
        # 基于距离分布创建confidence mask
        if points_np.ndim >= 3:
            original_shape = points_np.shape
            points_flat = points_np.reshape(-1, 3) if points_np.shape[-1] == 3 else points_np.reshape(-1)
            
            # 过滤无效点
            valid_mask = ~(np.isnan(points_flat).any(axis=1) | np.isinf(points_flat).any(axis=1))
            
            if np.any(valid_mask) and points_flat.shape[-1] == 3:
                valid_points = points_flat[valid_mask]
                center = np.mean(valid_points, axis=0)
                distances = np.linalg.norm(valid_points - center, axis=1)
                distance_threshold = np.percentile(distances, confidence_threshold)
                
                # 创建完整的mask
                full_mask = np.zeros(len(points_flat), dtype=bool)
                full_mask[valid_mask] = distances <= distance_threshold
                
                # 确保返回的mask形状与数据兼容
                target_shape = original_shape[:-1]  # 去掉最后的坐标维度
                reshaped_mask = full_mask.reshape(target_shape)
                logger.info(f"创建的confidence mask形状: {reshaped_mask.shape}, 目标数据形状: {original_shape}")
                return reshaped_mask
    
    logger.warning("无法创建置信度mask，将不进行置信度过滤")
    return None

def apply_confidence_filtering(data, confidence_mask):
    """对数据应用置信度过滤"""
    if confidence_mask is None:
        return data
    
    if isinstance(data, dict):
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = apply_confidence_filtering(value, confidence_mask)
        return filtered_data
    
    if data is None:
        return data
    
    try:
        # 转换为numpy
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            was_tensor = True
            original_device = data.device
            original_dtype = data.dtype
        else:
            data_np = data
            was_tensor = False
        
        # 确保mask形状兼容
        try:
            if data_np.ndim > confidence_mask.ndim:
                # 如果数据有额外维度（如坐标维度），扩展mask
                for _ in range(data_np.ndim - confidence_mask.ndim):
                    confidence_mask = np.expand_dims(confidence_mask, axis=-1)
                confidence_mask = np.broadcast_to(confidence_mask, data_np.shape)
            elif data_np.ndim < confidence_mask.ndim:
                # 如果mask维度更多，裁剪mask
                confidence_mask = confidence_mask.reshape(data_np.shape[:confidence_mask.ndim])
        except ValueError as e:
            logger.warning(f"无法广播confidence_mask形状 {confidence_mask.shape} 到数据形状 {data_np.shape}: {e}")
            return data
        
        # 应用mask：不符合条件的点设为0（而不是NaN）
        filtered_data = np.where(confidence_mask, data_np, 0.0)
        
        # 转换回原格式
        if was_tensor:
            filtered_data = torch.from_numpy(filtered_data).to(original_device).to(original_dtype)
        
        return filtered_data
        
    except Exception as e:
        logger.warning(f"置信度过滤失败: {e}，返回原始数据")
        return data

def apply_image_masking(images_data, mask_black_bg: bool, mask_white_bg: bool, mask_sky: bool):
    """对图像应用背景遮罩"""
    if images_data is None or not (mask_black_bg or mask_white_bg or mask_sky):
        return images_data
    
    try:
        if VGGT_NATIVE_AVAILABLE:
            # 转换为tensor格式进行处理
            if isinstance(images_data, np.ndarray):
                if images_data.max() <= 1.0:
                    images_tensor = torch.from_numpy(images_data * 255).to(torch.uint8)
                else:
                    images_tensor = torch.from_numpy(images_data).to(torch.uint8)
                was_numpy = True
            else:
                images_tensor = images_data
                was_numpy = False
            
            # 确保tensor格式正确
            if images_tensor.dim() == 4:  # (S, H, W, 3) -> (S, 3, H, W)
                if images_tensor.shape[-1] == 3:
                    images_tensor = images_tensor.permute(0, 3, 1, 2)
            
            # 应用遮罩
            if mask_black_bg:
                images_tensor = VGGTImageProcessor.apply_masking(images_tensor, 'black_bg')
            if mask_white_bg:
                images_tensor = VGGTImageProcessor.apply_masking(images_tensor, 'white_bg')
            if mask_sky:
                images_tensor = VGGTImageProcessor.apply_masking(images_tensor, 'sky')
            
            # 转换回原始格式
            if images_tensor.dim() == 4 and images_tensor.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                images_tensor = images_tensor.permute(0, 2, 3, 1)
            
            if was_numpy:
                return images_tensor.cpu().numpy()
            else:
                return images_tensor
        else:
            logger.warning("VGGT原生接口不可用，跳过图像遮罩")
            return images_data
            
    except Exception as e:
        logger.warning(f"图像遮罩应用失败: {e}，返回原始图像")
        return images_data

# -----------------------------------------------------------------------------
# 完整原生输出节点
# -----------------------------------------------------------------------------

class VGGTNativeFullOutputNode:
    """VGGT 原生完整输出节点 - 输出彻底过滤后的数据"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vggt_model": ("VVL_VGGT_MODEL", {
                    "tooltip": "来自VVLVGGTLoader的VGGT模型实例"
                }),
                "images": ("IMAGE", {
                    "tooltip": "图片序列输入（必需）- 支持1-200帧图像"
                }),
            },
            "optional": {
                "enable_tracks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用点追踪分支"
                }),
                "grid_size": ("INT", {
                    "default": 8, "min": 4, "max": 16, "step": 1,
                    "tooltip": "查询点网格大小"
                }),
                "query_region": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "查询点区域比例"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 90.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "置信度阈值(%)，过滤低置信度点"
                }),
                "show_cameras": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否在3D模型中显示相机位置"
                }),
                "mask_black_bg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否过滤黑色背景点"
                }),
                "mask_white_bg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否过滤白色背景点"
                }),
                "mask_sky": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否过滤天空点"
                }),
            }
        }

    RETURN_TYPES = (
        "RAW_VGGT_RESULT",   # 彻底过滤后的VGGT结果
        "STRING",            # 结果摘要JSON  
        "STRING",            # Tracks JSON数据
        "STRING",            # 相机内参JSON
        "STRING",            # 相机姿态JSON
        "IMAGE",             # 轨迹可视化预览
        "STRING",            # 3D模型GLB文件路径
        "STRING",            # 过滤后点云PLY文件路径
    )
    RETURN_NAMES = (
        "filtered_vggt_result",
        "result_summary", 
        "tracks_json",
        "intrinsics_json",
        "poses_json",
        "trajectory_preview",
        "model_3d_glb_path",
        "filtered_pointcloud_ply_path",
    )
    OUTPUT_TOOLTIPS = [
        "彻底过滤后的VGGT结果（所有点云数据已应用置信度和背景过滤）",
        "结果摘要信息（JSON格式）",
        "点追踪数据（JSON格式）",
        "相机内参数据（JSON格式）",
        "相机位姿数据（JSON格式）",
        "相机轨迹3D可视化预览图像",
        "3D模型GLB文件路径",
        "过滤后的点云PLY文件路径"
    ]
    OUTPUT_NODE = True
    FUNCTION = "get_filtered_output"
    CATEGORY = "💃VVL/VGGT Native"

    def get_filtered_output(self, vggt_model: Dict, images, 
                           enable_tracks: bool = True, 
                           grid_size: int = 8,
                           query_region: float = 0.8,
                           confidence_threshold: float = 90.0,
                           show_cameras: bool = True,
                           mask_black_bg: bool = False,
                           mask_white_bg: bool = False,
                           mask_sky: bool = False):
        """获取彻底过滤后的VGGT输出"""
        logger.info("开始VGGT原生推理并彻底过滤")
        
        # 获取模型实例和设备
        model_instance = vggt_model.get("model")
        device = vggt_model.get("device")
        
        if model_instance is None:
            raise ValueError("无效的VGGT模型实例")
        
        # 处理图片序列输入
        image_list = []
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
            for i in range(images_np.shape[0]):
                img = images_np[i]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                image_list.append(img)
        else:
            image_list.extend(images)
        
        if not image_list:
            raise ValueError("未能提取到有效的图像数据")
        
        logger.info(f"总共处理 {len(image_list)} 张图像")
        
        # 限制图像数量
        max_frames = 200
        if len(image_list) > max_frames:
            logger.warning(f"图像数量超过限制，采样到 {max_frames} 帧")
            indices = np.linspace(0, len(image_list) - 1, max_frames, dtype=int)
            image_list = [image_list[i] for i in indices]
        
        try:
            # 使用原生VGGT预处理
            processed_images = preprocess_images_native(image_list)
            processed_images = processed_images.to(device)
            
            # 执行原生推理
            raw_results = run_vggt_full_native_inference(
                processed_images, model_instance, device, 
                enable_tracks, grid_size, query_region
            )
            
            # 🎯 关键：应用彻底的过滤
            logger.info("应用彻底过滤到所有点云数据...")
            filtered_results = apply_comprehensive_filtering(
                raw_results, confidence_threshold, mask_black_bg, mask_white_bg, mask_sky
            )
            
            # 生成结果摘要
            summary = self._generate_result_summary(filtered_results, {
                "confidence_threshold": confidence_threshold,
                "mask_black_bg": mask_black_bg,
                "mask_white_bg": mask_white_bg,
                "mask_sky": mask_sky
            })
            summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
            
            # 生成tracks JSON输出
            tracks_json = self._generate_tracks_json(filtered_results)
            
            # 生成相机参数JSON输出
            intrinsics_json, poses_json = self._generate_camera_json(filtered_results)
            
            # 生成轨迹预览
            trajectory_preview = self._create_trajectory_preview(filtered_results)
            
            # 生成3D模型GLB文件
            model_3d_glb_path = self._generate_3d_model(filtered_results, confidence_threshold, show_cameras)
            
            # 生成过滤后点云PLY文件
            pointcloud_ply_path = self._export_filtered_pointcloud(filtered_results)
            
            logger.info("VGGT彻底过滤输出完成")
            return (filtered_results, summary_json, tracks_json, intrinsics_json, poses_json, 
                   trajectory_preview, model_3d_glb_path, pointcloud_ply_path)
            
        except Exception as e:
            logger.error(f"VGGT推理失败: {e}")
            error_msg = f"VGGT推理失败: {str(e)}"
            empty_result = {"error": str(e)}
            error_json = json.dumps({"error": str(e)})
            error_image = self._create_error_image()
            return (empty_result, error_msg, error_json, error_json, error_json, 
                   error_image, "", "")
    
    def _generate_result_summary(self, results: Dict, parameters: Dict = None) -> Dict:
        """生成结果摘要"""
        summary = {
            "available_branches": list(results.keys()),
            "filtering_applied": True,
            "metadata": {
                "generated_by": "VGGTNativeFullOutputNode",
                "description": "Comprehensively filtered VGGT results"
            }
        }
        
        if parameters is not None:
            summary["metadata"]["filtering_parameters"] = parameters
        
        # 添加数据形状信息
        for key, value in results.items():
            if hasattr(value, 'shape'):
                summary[f"{key}_shape"] = list(value.shape)
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())
        
        return summary
    
    def _generate_tracks_json(self, filtered_results: Dict) -> str:
        """生成tracks JSON数据"""
        if 'tracks' not in filtered_results:
            return json.dumps({"message": "No tracks data available"})
        
        try:
            track_data = filtered_results['tracks']
            tracks_export = {}
            
            for key, value in track_data.items():
                if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                    tracks_export[key] = value.cpu().numpy().tolist()
                elif isinstance(value, torch.Tensor):
                    tracks_export[key] = value.detach().cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    tracks_export[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    # 递归处理列表中可能的tensor
                    converted_list = []
                    for item in value:
                        if hasattr(item, 'cpu'):
                            converted_list.append(item.cpu().numpy().tolist())
                        elif isinstance(item, torch.Tensor):
                            converted_list.append(item.detach().cpu().numpy().tolist())
                        else:
                            converted_list.append(item)
                    tracks_export[key] = converted_list
                else:
                    tracks_export[key] = value
            
            return json.dumps({"tracks_data": tracks_export}, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize tracks data: {e}")
            return json.dumps({"error": f"Failed to serialize tracks: {str(e)}"})
    
    def _generate_camera_json(self, filtered_results: Dict) -> tuple:
        """生成相机参数JSON"""
        intrinsics_json = ""
        poses_json = ""
        
        if 'cameras' in filtered_results:
            try:
                cameras_data = filtered_results['cameras']
                if isinstance(cameras_data, dict):
                    if 'intrinsic' in cameras_data:
                        intrinsic = cameras_data['intrinsic']
                        if hasattr(intrinsic, 'cpu'):
                            intrinsic = intrinsic.cpu().numpy()
                        intrinsics_json = json.dumps(intrinsic.tolist(), ensure_ascii=False, indent=2)
                    
                    if 'extrinsic' in cameras_data:
                        extrinsic = cameras_data['extrinsic']
                        if hasattr(extrinsic, 'cpu'):
                            extrinsic = extrinsic.cpu().numpy()
                        poses_json = json.dumps(extrinsic.tolist(), ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to generate camera JSON: {e}")
                intrinsics_json = json.dumps({"error": f"Failed to extract intrinsics: {str(e)}"})
                poses_json = json.dumps({"error": f"Failed to extract poses: {str(e)}"})
        else:
            intrinsics_json = json.dumps({"message": "No camera data available"})
            poses_json = json.dumps({"message": "No camera data available"})
        
        return intrinsics_json, poses_json
    
    def _create_error_image(self) -> torch.Tensor:
        """创建错误提示图像"""
        if VGGT_UTILS_AVAILABLE and _create_insufficient_data_image:
            return _create_insufficient_data_image()
        else:
            canvas = np.ones((400, 600, 3), dtype=np.float32) * 0.9
            return torch.from_numpy(canvas).unsqueeze(0)
    
    def _create_trajectory_preview(self, filtered_results: Dict) -> torch.Tensor:
        """生成轨迹预览"""
        if not VGGT_UTILS_AVAILABLE or not _create_traj_preview:
            return self._create_error_image()
        
        try:
            if 'cameras' in filtered_results:
                extrinsic_for_preview = filtered_results['cameras']['extrinsic']
                
                if isinstance(extrinsic_for_preview, torch.Tensor):
                    if extrinsic_for_preview.ndim == 4 and extrinsic_for_preview.shape[0] == 1:
                        extrinsic_for_preview = extrinsic_for_preview.squeeze(0)
                elif isinstance(extrinsic_for_preview, np.ndarray):
                    if extrinsic_for_preview.ndim == 4 and extrinsic_for_preview.shape[0] == 1:
                        extrinsic_for_preview = np.squeeze(extrinsic_for_preview, axis=0)
                
                return _create_traj_preview(extrinsic_for_preview)
            else:
                return self._create_error_image()
        except Exception as e:
            logger.warning(f"轨迹预览生成失败: {e}")
            return self._create_error_image()
    
    def _generate_3d_model(self, filtered_results: Dict, confidence_threshold: float, show_cameras: bool) -> str:
        """生成3D模型GLB文件"""
        if not VGGT_UTILS_AVAILABLE or not predictions_to_glb:
            logger.warning("VGGT工具函数不可用，跳过3D模型生成")
            return ""
        
        try:
            # 创建输出目录
            if FOLDER_PATHS_AVAILABLE:
                output_dir = os.path.join(folder_paths.get_output_directory(), "3d")
            else:
                output_dir = os.path.join("output", "3d")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import time
            timestamp = int(time.time())
            model_path = os.path.join(output_dir, f"vggt_filtered_{timestamp}.glb")
            
            logger.info(f"生成过滤后的3D模型: {model_path}")
            
            # 准备预测数据
            predictions_formatted = self._prepare_predictions_data(filtered_results)
            
            # 使用官方VGGT的predictions_to_glb函数
            scene_3d = predictions_to_glb(
                predictions_formatted,
                conf_thres=confidence_threshold,
                filter_by_frames="all",
                mask_black_bg=False,  # 已经在原生过滤中处理
                mask_white_bg=False,
                show_cam=show_cameras,
                mask_sky=False,
                target_dir=None,
                prediction_mode="Depthmap and Camera Branch"
            )
            
            # 导出为GLB文件
            scene_3d.export(model_path)
            
            logger.info(f"过滤后3D模型已保存: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"生成3D模型失败: {e}")
            return ""
    
    def _export_filtered_pointcloud(self, filtered_results: Dict) -> str:
        """导出过滤后的点云PLY文件"""
        try:
            # 创建输出目录
            if FOLDER_PATHS_AVAILABLE:
                output_dir = os.path.join(folder_paths.get_output_directory(), "pointclouds")
            else:
                output_dir = os.path.join("output", "pointclouds")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import time
            timestamp = int(time.time())
            ply_path = os.path.join(output_dir, f"vggt_filtered_{timestamp}.ply")
            
            # 提取点云数据
            points_data = filtered_results.get('points_from_depth') or filtered_results.get('points')
            if points_data is None:
                logger.warning("No point cloud data found")
                return ""
            
            if isinstance(points_data, dict):
                points_data = points_data.get('point_map', points_data)
            
            # 转换为numpy
            if isinstance(points_data, torch.Tensor):
                points_np = points_data.cpu().numpy()
            else:
                points_np = points_data
            
            # 去掉batch维度
            if points_np.ndim == 5 and points_np.shape[0] == 1:
                points_np = np.squeeze(points_np, axis=0)
            
            # 重塑点云数据
            if points_np.ndim == 4:  # (S, H, W, 3)
                points_np = points_np.reshape(-1, 3)
            
            # 过滤无效点（已经过滤但可能仍有0值）
            valid_mask = ~(np.isnan(points_np).any(axis=1) | np.isinf(points_np).any(axis=1))
            valid_mask = valid_mask & (np.linalg.norm(points_np, axis=1) > 1e-6)  # 排除0点
            
            if not np.any(valid_mask):
                logger.warning("No valid points after filtering")
                return ""
                
            points_np = points_np[valid_mask]
            
            # 获取颜色信息
            colors_rgb = self._extract_colors(filtered_results, valid_mask)
            
            # 写入PLY文件
            self._write_ply_file(ply_path, points_np, colors_rgb)
            
            logger.info(f"过滤后点云PLY已保存: {ply_path} ({len(points_np)} 个点)")
            return ply_path
            
        except Exception as e:
            logger.error(f"导出过滤后点云失败: {e}")
            return ""
    
    def _prepare_predictions_data(self, filtered_results: Dict) -> Dict:
        """准备predictions_to_glb需要的数据格式"""
        predictions = {}
        
        # 处理点云数据
        if 'points_from_depth' in filtered_results:
            world_points = filtered_results['points_from_depth']
            if isinstance(world_points, torch.Tensor) and world_points.ndim == 5:
                world_points = world_points.squeeze(0)
            predictions['world_points_from_depth'] = world_points
        
        # 处理深度置信度
        if 'depth' in filtered_results:
            depth_data = filtered_results['depth']
            if isinstance(depth_data, dict):
                predictions['depth_conf'] = depth_data.get('confidence')
            
        # 处理图像
        if 'images' in filtered_results:
            images = filtered_results['images']
            if isinstance(images, np.ndarray) and images.ndim == 4:
                predictions['images'] = images
        
        # 处理相机外参
        if 'cameras' in filtered_results:
            cameras = filtered_results['cameras']
            if isinstance(cameras, dict):
                extrinsic = cameras.get('extrinsic')
                if isinstance(extrinsic, torch.Tensor) and extrinsic.ndim == 4:
                    extrinsic = extrinsic.squeeze(0)
                predictions['extrinsic'] = extrinsic
        
        # 转换tensor为numpy
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy()
        
        return predictions
    
    def _extract_colors(self, filtered_results: Dict, valid_mask: np.ndarray) -> np.ndarray:
        """提取点云颜色信息"""
        try:
            if 'images' in filtered_results:
                images = filtered_results['images']
                if isinstance(images, np.ndarray) and images.ndim == 4:
                    # 重塑图像为点云颜色
                    colors_flat = images.reshape(-1, 3)
                    colors_flat = colors_flat[valid_mask]
                    
                    # 确保颜色值在0-255范围
                    if colors_flat.max() <= 1.0:
                        colors_flat = (colors_flat * 255).astype(np.uint8)
                    else:
                        colors_flat = colors_flat.astype(np.uint8)
                    
                    return colors_flat
        except Exception as e:
            logger.warning(f"提取颜色失败: {e}")
        
        # 使用默认灰色
        return np.ones((np.sum(valid_mask), 3), dtype=np.uint8) * 128
    
    def _write_ply_file(self, filepath: str, vertices: np.ndarray, colors: np.ndarray = None):
        """写入PLY格式点云文件"""
        try:
            with open(filepath, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                
                if colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                
                f.write("end_header\n")
                
                for i in range(len(vertices)):
                    x, y, z = vertices[i]
                    if colors is not None:
                        r, g, b = colors[i]
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                    else:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                        
        except Exception as e:
            logger.error(f"写入PLY文件失败: {e}")
            raise

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTNativeFullOutputNode": VGGTNativeFullOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTNativeFullOutputNode": "🔬 VGGT Native Filtered Output",
}

if MODEL_LOADER_AVAILABLE:
    NODE_CLASS_MAPPINGS["VVLVGGTLoader"] = VVLVGGTLoader
    NODE_DISPLAY_NAME_MAPPINGS["VVLVGGTLoader"] = "VVL VGGT Model Loader" 