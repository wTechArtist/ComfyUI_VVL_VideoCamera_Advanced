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
# 原生VGGT推理函数（增强版）
# -----------------------------------------------------------------------------

def run_vggt_full_native_inference(images: torch.Tensor, model_instance, device, 
                                  enable_tracks: bool = True, 
                                  grid_size: int = 8,
                                  query_region: float = 0.8) -> Dict:
    """使用原生VGGT接口进行完整推理，启用所有原生输出分支"""
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
            logger.info(f"Generated {len(query_points)} query points ({grid_size}x{grid_size} grid) in {query_region*100:.0f}% region")
        
        # 执行选择性推理以获取所有分支的输出
        results = vggt_interface.selective_inference(
            images, 
            branches=branches,
            query_points=query_points
        )
        
        # 添加原始图像到结果中（用于后续处理）
        if images.dim() == 5:  # (1, N, C, H, W)
            images_formatted = images.squeeze(0).permute(0, 2, 3, 1)  # (N, H, W, C)
        else:  # (N, C, H, W)
            images_formatted = images.permute(0, 2, 3, 1)  # (N, H, W, C)
        results['images'] = images_formatted.cpu().numpy()
        
        # 添加查询点信息
        if query_points is not None:
            results['query_points'] = query_points.cpu().numpy()
        
        logger.info(f"VGGT full native inference completed. Results keys: {list(results.keys())}")
        if 'tracks' in results:
            track_data = results['tracks']
            logger.info(f"Tracks data structure: {list(track_data.keys())}")
            if 'track_list' in track_data:
                track_shape = track_data['track_list'].shape if hasattr(track_data['track_list'], 'shape') else 'unknown'
                logger.info(f"Track list shape: {track_shape}")
        
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

# -----------------------------------------------------------------------------
# 完整原生输出节点
# -----------------------------------------------------------------------------

class VGGTNativeFullOutputNode:
    """VGGT 原生完整输出节点 - 输出所有原生分支数据，包括tracks"""

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
                    "tooltip": "是否启用点追踪分支（需要自动生成查询点）"
                }),
                "grid_size": ("INT", {
                    "default": 8, "min": 4, "max": 16, "step": 1,
                    "tooltip": "查询点网格大小（grid_size x grid_size个点）"
                }),
                "query_region": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "查询点区域比例（1.0=全图，0.8=中央80%区域）"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 90.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "置信度阈值(%)，用于过滤3D点云中的低置信度点"
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
        "RAW_VGGT_RESULT",   # 完整原生VGGT字典
        "STRING",            # 完整结果summary JSON  
        "STRING",            # Tracks JSON数据（如果有）
        "STRING",            # 相机内参JSON（与原节点兼容）
        "STRING",            # 相机姿态JSON（与原节点兼容）
        "IMAGE",             # 轨迹可视化预览
        "STRING",            # 3D模型GLB文件路径
        "STRING",            # 原生点云PLY文件路径
    )
    RETURN_NAMES = (
        "native_full_result",
        "result_summary", 
        "tracks_json",
        "intrinsics_json",
        "poses_json",
        "trajectory_preview",
        "model_3d_glb_path",
        "pointcloud_ply_path",
    )
    OUTPUT_TOOLTIPS = [
        "完整的VGGT原生推理结果（包含所有分支数据：cameras, depth, points, points_from_depth, tracks等）",
        "结果摘要信息（JSON格式，包含数据形状和结构信息）",
        "点追踪数据的JSON格式输出（如果启用了tracks分支）",
        "相机内参数据（JSON格式，与原节点输出兼容）",
        "相机位姿数据（JSON格式，与原节点输出兼容）",
        "相机轨迹3D可视化预览图像",
        "3D模型GLB文件路径（可连接到Preview3D或其他节点）",
        "原生点云PLY文件路径（VGGT原生格式，便于外部处理）"
    ]
    OUTPUT_NODE = True
    FUNCTION = "get_native_full_output"
    CATEGORY = "💃VVL/VGGT Native"

    def get_native_full_output(self, vggt_model: Dict, images, 
                              enable_tracks: bool = True, 
                              grid_size: int = 8,
                              query_region: float = 0.8,
                              confidence_threshold: float = 90.0,
                              show_cameras: bool = True,
                              mask_black_bg: bool = False,
                              mask_white_bg: bool = False,
                              mask_sky: bool = False):
        """获取VGGT完整原生输出"""
        logger.info("开始VGGT原生完整输出")
        
        # 获取模型实例和设备
        model_instance = vggt_model.get("model")
        device = vggt_model.get("device")
        
        if model_instance is None:
            raise ValueError("无效的VGGT模型实例")
        
        logger.info(f"使用设备: {device}")
        
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
            
            # 应用遮罩（如果需要）
            if VGGT_NATIVE_AVAILABLE:
                mask_types = []
                if mask_black_bg:
                    mask_types.append('black_bg')
                if mask_white_bg:
                    mask_types.append('white_bg')
                if mask_sky:
                    mask_types.append('sky')
                
                for mask_type in mask_types:
                    processed_images = VGGTImageProcessor.apply_masking(processed_images, mask_type)
            
            # 执行完整的原生推理，启用所有分支
            raw_results = run_vggt_full_native_inference(
                processed_images, model_instance, device, 
                enable_tracks, grid_size, query_region
            )
            
            # 生成结果摘要
            summary = self._generate_result_summary(raw_results, {
                "enable_tracks": enable_tracks,
                "grid_size": grid_size,
                "query_region": query_region,
                "confidence_threshold": confidence_threshold,
                "show_cameras": show_cameras,
                "mask_black_bg": mask_black_bg,
                "mask_white_bg": mask_white_bg,
                "mask_sky": mask_sky
            })
            summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
            
            # 生成tracks JSON输出
            tracks_json = ""
            if 'tracks' in raw_results and raw_results['tracks']:
                try:
                    track_data = raw_results['tracks']
                    logger.info(f"处理tracks数据，keys: {list(track_data.keys())}")
                    
                    # 将tensor转换为可序列化的格式
                    tracks_export = {}
                    for key, value in track_data.items():
                        logger.info(f"处理tracks key: {key}, type: {type(value)}")
                        if hasattr(value, 'cpu'):
                            # PyTorch tensor
                            value_np = value.cpu().numpy()
                            tracks_export[key] = value_np.tolist()
                            logger.info(f"转换tensor {key}, shape: {value_np.shape}")
                        elif isinstance(value, np.ndarray):
                            # NumPy array
                            tracks_export[key] = value.tolist()
                            logger.info(f"转换numpy {key}, shape: {value.shape}")
                        elif isinstance(value, (list, tuple)):
                            # 列表或元组，递归处理内部元素
                            tracks_export[key] = self._convert_to_serializable(value)
                            logger.info(f"转换列表 {key}, length: {len(value)}")
                        else:
                            # 其他类型直接赋值
                            tracks_export[key] = value
                            logger.info(f"直接赋值 {key}, type: {type(value)}")
                    
                    # 添加详细的元数据
                    metadata = {
                        "format": "coordinates_per_frame",
                        "description": "Point tracking data with coordinates, visibility and confidence",
                        "coordinate_format": "(x, y) pixel coordinates",
                        "data_structure": {}
                    }
                    
                    # 添加每个字段的形状信息
                    for key, value in tracks_export.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], list):
                                # 多维列表
                                shape = [len(value)]
                                temp = value[0]
                                while isinstance(temp, list) and len(temp) > 0:
                                    shape.append(len(temp))
                                    temp = temp[0] if isinstance(temp[0], list) else None
                                    if temp is None:
                                        break
                                metadata["data_structure"][key] = {
                                    "shape": shape,
                                    "description": self._get_track_field_description(key)
                                }
                            else:
                                metadata["data_structure"][key] = {
                                    "length": len(value),
                                    "description": self._get_track_field_description(key)
                                }
                        else:
                            metadata["data_structure"][key] = {
                                "type": type(value).__name__,
                                "description": self._get_track_field_description(key)
                            }
                    
                    tracks_json = json.dumps({
                        "tracks_data": tracks_export,
                        "metadata": metadata
                    }, ensure_ascii=False, indent=2)
                    
                    logger.info("tracks数据序列化成功")
                    
                except Exception as e:
                    logger.error(f"Failed to serialize tracks data: {e}")
                    import traceback
                    traceback.print_exc()
                    tracks_json = json.dumps({
                        "error": f"Failed to serialize tracks: {str(e)}",
                        "available_keys": list(raw_results['tracks'].keys()) if 'tracks' in raw_results else [],
                        "debug_info": "Check logs for detailed error information"
                    }, ensure_ascii=False, indent=2)
            else:
                tracks_json = json.dumps({"message": "No tracks data available or tracks not enabled"})
            
            # 生成与原节点兼容的相机参数JSON输出
            intrinsics_json = ""
            poses_json = ""
            if 'cameras' in raw_results:
                try:
                    cameras_data = raw_results['cameras']
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
            
            # 生成轨迹预览
            trajectory_preview = self._create_trajectory_preview(raw_results)
            
            # 生成3D模型GLB文件
            model_3d_glb_path = self._generate_3d_model_official(
                raw_results, 
                confidence_threshold, 
                show_cameras,
                mask_black_bg,
                mask_white_bg,
                mask_sky
            )
            
            # 生成原生点云PLY文件
            pointcloud_ply_path = self._export_native_pointcloud(raw_results, confidence_threshold)
            
            logger.info("VGGT原生完整输出完成")
            return (raw_results, summary_json, tracks_json, intrinsics_json, poses_json, 
                   trajectory_preview, model_3d_glb_path, pointcloud_ply_path)
            
        except Exception as e:
            logger.error(f"VGGT原生完整推理失败: {e}")
            error_msg = f"VGGT原生完整推理失败: {str(e)}"
            empty_result = {"error": str(e)}
            error_json = json.dumps({"error": str(e)})
            error_image = self._create_error_image()
            return (empty_result, error_msg, error_json, error_json, error_json, 
                   error_image, "", "")
    
    def _generate_result_summary(self, results: Dict, parameters: Dict = None) -> Dict:
        """生成详细的结果摘要信息"""
        summary = {
            "available_branches": list(results.keys()),
            "data_info": {},
            "metadata": {
                "generated_by": "VGGTNativeFullOutputNode",
                "description": "Complete VGGT native inference results with all branches"
            }
        }
        
        # 添加处理参数到元数据
        if parameters is not None:
            summary["metadata"]["processing_parameters"] = parameters
        
        for key, value in results.items():
            if key == "images":
                if isinstance(value, np.ndarray):
                    summary["data_info"][key] = {
                        "type": "numpy.ndarray",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "description": "Input images in (N, H, W, C) format"
                    }
            elif key == "cameras":
                if isinstance(value, dict):
                    summary["data_info"][key] = {
                        "type": "dict",
                        "keys": list(value.keys()),
                        "description": "Camera parameters (extrinsic and intrinsic matrices)"
                    }
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            summary["data_info"][key][f"{sub_key}_shape"] = list(sub_value.shape)
            elif key == "depth":
                if isinstance(value, dict):
                    summary["data_info"][key] = {
                        "type": "dict",
                        "keys": list(value.keys()),
                        "description": "Depth maps and confidence"
                    }
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            summary["data_info"][key][f"{sub_key}_shape"] = list(sub_value.shape)
            elif key == "tracks":
                if isinstance(value, dict):
                    summary["data_info"][key] = {
                        "type": "dict",
                        "keys": list(value.keys()),
                        "description": "Point tracking data (coordinates, visibility, confidence)"
                    }
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            summary["data_info"][key][f"{sub_key}_shape"] = list(sub_value.shape)
                            if sub_key == 'track_list' and hasattr(sub_value, 'shape') and len(sub_value.shape) >= 2:
                                summary["data_info"][key]["n_points"] = sub_value.shape[-3] if len(sub_value.shape) >= 3 else "unknown"
                                summary["data_info"][key]["n_frames"] = sub_value.shape[-2] if len(sub_value.shape) >= 2 else "unknown"
            elif key == "query_points":
                if isinstance(value, np.ndarray):
                    summary["data_info"][key] = {
                        "type": "numpy.ndarray",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "description": "Query points used for tracking (N_points, 2) format (x, y)",
                        "n_points": value.shape[0] if len(value.shape) >= 1 else 0
                    }
            elif hasattr(value, 'shape'):
                summary["data_info"][key] = {
                    "type": type(value).__name__,
                    "shape": list(value.shape),
                    "description": f"Raw {key} data from VGGT"
                }
            elif isinstance(value, dict):
                summary["data_info"][key] = {
                    "type": "dict",
                    "keys": list(value.keys()),
                    "description": f"Structured {key} data"
                }
            else:
                summary["data_info"][key] = {
                    "type": type(value).__name__,
                    "description": f"Raw {key} data from VGGT"
                }
        
        return summary
    
    def _create_error_image(self) -> torch.Tensor:
        """创建错误提示图像"""
        if VGGT_UTILS_AVAILABLE and _create_insufficient_data_image:
            return _create_insufficient_data_image()
        else:
            # 创建简单的错误图像
            canvas = np.ones((400, 600, 3), dtype=np.float32) * 0.9
            # 这里可以添加文字，但为了简单起见直接返回灰色图像
            return torch.from_numpy(canvas).unsqueeze(0)
    
    def _create_trajectory_preview(self, raw_results: Dict) -> torch.Tensor:
        """生成轨迹预览"""
        if not VGGT_UTILS_AVAILABLE or not _create_traj_preview:
            return self._create_error_image()
        
        try:
            if 'cameras' in raw_results:
                extrinsic_for_preview = raw_results['cameras']['extrinsic']
                
                # 应用与GLB生成相同的维度处理逻辑
                if isinstance(extrinsic_for_preview, torch.Tensor):
                    if extrinsic_for_preview.ndim == 4 and extrinsic_for_preview.shape[0] == 1:
                        extrinsic_for_preview = extrinsic_for_preview.squeeze(0)  # (1, S, 3, 4) -> (S, 3, 4)
                elif isinstance(extrinsic_for_preview, np.ndarray):
                    if extrinsic_for_preview.ndim == 4 and extrinsic_for_preview.shape[0] == 1:
                        extrinsic_for_preview = np.squeeze(extrinsic_for_preview, axis=0)  # (1, S, 3, 4) -> (S, 3, 4)
                
                logger.info(f"轨迹预览外参形状: {extrinsic_for_preview.shape}")
                return _create_traj_preview(extrinsic_for_preview)
            else:
                return self._create_error_image()
        except Exception as e:
            logger.warning(f"轨迹预览生成失败: {e}")
            return self._create_error_image()
    
    def _generate_3d_model_official(self, raw_results: Dict, confidence_threshold: float, show_cameras: bool, 
                                   mask_black_bg: bool = False, mask_white_bg: bool = False, mask_sky: bool = False) -> str:
        """使用官方VGGT的predictions_to_glb函数生成高质量3D模型"""
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
            model_path = os.path.join(output_dir, f"vggt_native_{timestamp}.glb")
            
            logger.info(f"开始生成原生VGGT质量的3D模型: {model_path}")
            
            # 准备官方格式的预测数据
            world_points_from_depth = raw_results.get('points_from_depth')
            
            # depth_conf: 兼容多种返回格式
            depth_conf = None
            if 'depth' in raw_results:
                if isinstance(raw_results['depth'], dict):
                    depth_conf = raw_results['depth'].get('confidence')
                else:
                    depth_conf = raw_results.get('depth_conf')
            
            # images
            images_tensor = raw_results.get('images')
            
            # extrinsic
            extrinsic_mat = None
            if 'cameras' in raw_results:
                if isinstance(raw_results['cameras'], dict):
                    extrinsic_mat = raw_results['cameras'].get('extrinsic')
                else:
                    extrinsic_mat = raw_results.get('extrinsic')
            
            # 特别处理 extrinsic: 去掉 batch 维 (1, S, 3, 4) -> (S, 3, 4)
            if isinstance(extrinsic_mat, torch.Tensor):
                if extrinsic_mat.ndim == 4 and extrinsic_mat.shape[0] == 1:
                    extrinsic_mat = extrinsic_mat.squeeze(0)
            elif isinstance(extrinsic_mat, np.ndarray):
                if extrinsic_mat.ndim == 4 and extrinsic_mat.shape[0] == 1:
                    extrinsic_mat = np.squeeze(extrinsic_mat, axis=0)
            
            # 维度兼容处理
            images_tensor_proc = images_tensor
            if isinstance(images_tensor_proc, torch.Tensor):
                if images_tensor_proc.ndim == 5 and images_tensor_proc.shape[0] == 1:
                    images_tensor_proc = images_tensor_proc.squeeze(0)
            elif isinstance(images_tensor_proc, np.ndarray):
                if images_tensor_proc.ndim == 5 and images_tensor_proc.shape[0] == 1:
                    images_tensor_proc = np.squeeze(images_tensor_proc, axis=0)
            
            wpfd_proc = world_points_from_depth
            if isinstance(wpfd_proc, torch.Tensor):
                if wpfd_proc is not None and wpfd_proc.ndim == 5 and wpfd_proc.shape[0] == 1:
                    wpfd_proc = wpfd_proc.squeeze(0)
            elif isinstance(wpfd_proc, np.ndarray):
                if wpfd_proc is not None and wpfd_proc.ndim == 5 and wpfd_proc.shape[0] == 1:
                    wpfd_proc = np.squeeze(wpfd_proc, axis=0)
            
            predictions_formatted = {
                'world_points_from_depth': wpfd_proc,
                'depth_conf': depth_conf,
                'images': images_tensor_proc,
                'extrinsic': extrinsic_mat,
            }
            
            # 如果没有depth-based points，尝试使用原始点云
            if predictions_formatted['world_points_from_depth'] is None:
                if 'points' in raw_results:
                    logger.info("Using point_map as fallback for world_points")
                    if isinstance(raw_results['points'], dict):
                        predictions_formatted['world_points'] = raw_results['points']['point_map']
                        predictions_formatted['world_points_conf'] = raw_results['points']['confidence']
                    else:
                        predictions_formatted['world_points'] = raw_results['points']
            
            # 确保数据格式正确（转换为numpy）
            for key, value in predictions_formatted.items():
                if value is not None and isinstance(value, torch.Tensor):
                    predictions_formatted[key] = value.cpu().numpy()
            
            # 使用官方VGGT的predictions_to_glb函数
            scene_3d = predictions_to_glb(
                predictions_formatted,
                conf_thres=confidence_threshold,
                filter_by_frames="all",
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cameras,
                mask_sky=mask_sky,
                target_dir=None,
                prediction_mode="Depthmap and Camera Branch"
            )
            
            # 导出为GLB文件
            scene_3d.export(model_path)
            
            logger.info(f"原生VGGT质量3D模型已保存到: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"生成3D模型失败: {e}")
            return ""
    
    def _export_native_pointcloud(self, raw_results: Dict, confidence_threshold: float) -> str:
        """导出原生格式的点云PLY文件"""
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
            ply_path = os.path.join(output_dir, f"vggt_native_{timestamp}.ply")
            
            # 从原生结果提取点云数据
            if 'points_from_depth' in raw_results:
                points = raw_results['points_from_depth']
                logger.info("使用 points_from_depth 生成PLY")
            elif 'points' in raw_results:
                if isinstance(raw_results['points'], dict):
                    points = raw_results['points']['point_map']
                else:
                    points = raw_results['points']
                logger.info("使用 points 生成PLY")
            else:
                logger.warning("No point cloud data found in raw results")
                return ""
            
            # 转换为numpy数组
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = points
            
            # 去掉batch维度
            if points_np.ndim == 5 and points_np.shape[0] == 1:
                points_np = np.squeeze(points_np, axis=0)
            
            # 重塑点云数据 (S, H, W, 3) -> (S*H*W, 3)
            if points_np.ndim == 4:
                points_np = points_np.reshape(-1, 3)
            elif points_np.ndim == 3:
                points_np = points_np.reshape(-1, 3)
            
            # 过滤无效点
            valid_mask = ~np.isnan(points_np).any(axis=1) & ~np.isinf(points_np).any(axis=1)
            points_np = points_np[valid_mask]
            
            if len(points_np) == 0:
                logger.warning("No valid points found")
                return ""
            
            # 获取颜色信息
            colors_rgb = None
            if 'images' in raw_results:
                images = raw_results['images']
                if isinstance(images, torch.Tensor):
                    images_np = images.cpu().numpy()
                else:
                    images_np = images
                
                # 去掉batch维度
                if images_np.ndim == 5 and images_np.shape[0] == 1:
                    images_np = np.squeeze(images_np, axis=0)
                
                # 处理图像格式 (S, H, W, 3) or (S, 3, H, W)
                if images_np.ndim == 4:
                    if images_np.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                        images_np = np.transpose(images_np, (0, 2, 3, 1))
                    
                    # 重塑为点云颜色 (S*H*W, 3)
                    colors_rgb = images_np.reshape(-1, 3)
                    colors_rgb = colors_rgb[valid_mask]  # 应用相同的mask
                    
                    # 确保颜色值在0-255范围
                    if colors_rgb.max() <= 1.0:
                        colors_rgb = (colors_rgb * 255).astype(np.uint8)
                    else:
                        colors_rgb = colors_rgb.astype(np.uint8)
            
            # 基于置信度过滤
            if confidence_threshold > 0 and len(points_np) > 100:
                # 使用距离作为置信度的简单替代
                center = np.mean(points_np, axis=0)
                distances = np.linalg.norm(points_np - center, axis=1)
                threshold = np.percentile(distances, confidence_threshold)
                conf_mask = distances <= threshold
                points_np = points_np[conf_mask]
                if colors_rgb is not None:
                    colors_rgb = colors_rgb[conf_mask]
            
            # 写入PLY文件
            self._write_ply_file(ply_path, points_np, colors_rgb)
            
            logger.info(f"原生点云PLY已保存到: {ply_path} ({len(points_np)} 个点)")
            return ply_path
            
        except Exception as e:
            logger.error(f"导出点云PLY失败: {e}")
            raise
    
    def _write_ply_file(self, filepath: str, vertices: np.ndarray, colors: np.ndarray = None):
        """写入PLY格式点云文件"""
        try:
            with open(filepath, 'w') as f:
                # PLY header
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
                
                # 写入顶点数据
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
    
    def _convert_to_serializable(self, obj):
        """递归转换对象为JSON可序列化格式"""
        if hasattr(obj, 'cpu'):
            # PyTorch tensor
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            # NumPy array
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            # 列表或元组，递归处理每个元素
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            # 字典，递归处理每个值
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        else:
            # 其他类型直接返回
            return obj
    
    def _get_track_field_description(self, field_name: str) -> str:
        """获取tracks字段的描述信息"""
        descriptions = {
            "track_list": "Point coordinates per frame - shape: (N_points, N_frames, 2) where last dim is (x, y)",
            "visibility_score": "Visibility scores per point per frame - shape: (N_points, N_frames)",
            "confidence_score": "Confidence scores per point per frame - shape: (N_points, N_frames)",
            "coords": "Alternative coordinate format",
            "vis": "Alternative visibility format",
            "conf": "Alternative confidence format"
        }
        return descriptions.get(field_name, f"Data field: {field_name}")

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTNativeFullOutputNode": VGGTNativeFullOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTNativeFullOutputNode": "🔬 VGGT Native Full Output",
}

# 如果模型加载器可用，添加到映射中
if MODEL_LOADER_AVAILABLE:
    NODE_CLASS_MAPPINGS["VVLVGGTLoader"] = VVLVGGTLoader
    NODE_DISPLAY_NAME_MAPPINGS["VVLVGGTLoader"] = "VVL VGGT Model Loader" 