# VGGT Mask处理节点文件

import os
import json
import tempfile
from typing import List, Any, Dict, Tuple
import logging
from collections import defaultdict
import time

import cv2
import numpy as np
import torch
from PIL import Image

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
        predictions_to_glb,
        TRIMESH_AVAILABLE,
        MATPLOTLIB_AVAILABLE,
        SCIPY_AVAILABLE
    )
    VGGT_UTILS_AVAILABLE = True
except ImportError:
    predictions_to_glb = None
    TRIMESH_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    SCIPY_AVAILABLE = False
    VGGT_UTILS_AVAILABLE = False

# 配置日志
logger = logging.getLogger('vvl_vggt_mask_processor')

# -----------------------------------------------------------------------------
# 核心交集算法
# -----------------------------------------------------------------------------

def compute_pointcloud_mask_intersection(filtered_vggt_result: Dict, mask_sequence: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算过滤后点云与mask白色区域的交集"""
    logger.info("🎯 开始计算点云与mask的交集")
    
    # 1. 获取过滤后的点云数据
    points_data = filtered_vggt_result.get('points_from_depth')
    if points_data is None:
        points_data = filtered_vggt_result.get('points')
    if points_data is None:
        raise ValueError("过滤后的VGGT结果中没有点云数据")
    
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
    
    original_shape = points_np.shape
    logger.info(f"过滤后点云形状: {original_shape}")
    
    # 2. 验证点云格式并重塑
    if len(original_shape) != 4:  # 必须是 (S, H, W, 3)
        raise ValueError(f"点云形状必须是(S, H, W, 3)，但得到: {original_shape}")
    
    S, H, W, _ = original_shape
    points_flat = points_np.reshape(-1, 3)  # (S*H*W, 3)
    
    # 3. 获取对应的图像颜色
    colors_flat = extract_colors_from_filtered_result(filtered_vggt_result, original_shape)
    
    # 4. 处理mask序列
    logger.info(f"处理mask序列: {len(mask_sequence)} 张")
    
    # 检查mask分辨率
    first_mask = mask_sequence[0]
    mask_h, mask_w = first_mask.shape
    logger.info(f"Mask分辨率: {mask_w}x{mask_h}, 点云分辨率: {W}x{H}")
    
    # 如果分辨率不匹配，缩放mask
    if W != mask_w or H != mask_h:
        logger.info(f"缩放mask从{mask_w}x{mask_h}到{W}x{H}")
        resized_masks = []
        for mask in mask_sequence:
            resized_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized_mask)
        mask_sequence = resized_masks
    
    # 5. 确定mask阈值（白色区域）
    mask_threshold = determine_white_threshold(mask_sequence)
    logger.info(f"使用白色区域阈值: {mask_threshold}")
    
    # 6. 计算交集mask
    intersection_mask = np.zeros(len(points_flat), dtype=bool)
    
    n_frames = min(S, len(mask_sequence))
    logger.info(f"处理 {n_frames} 帧进行交集计算")
    
    for frame_idx in range(n_frames):
        frame_mask = mask_sequence[frame_idx]  # (H, W)
        
        # 计算当前帧在扁平化数组中的索引范围
        frame_start = frame_idx * H * W
        frame_end = frame_start + H * W
        
        # 展平mask
        mask_flat = frame_mask.flatten()  # (H*W,)
        
        # 找到白色区域
        white_pixels = mask_flat > mask_threshold
        
        # 过滤掉已经被VGGT过滤掉的点（值为0的点）
        frame_points = points_flat[frame_start:frame_end]
        valid_points = np.linalg.norm(frame_points, axis=1) > 1e-6  # 非零点
        
        # 交集：既在白色区域又是有效点
        frame_intersection = white_pixels & valid_points
        
        # 更新总交集mask
        intersection_mask[frame_start:frame_end] = frame_intersection
        
        frame_intersection_count = np.sum(frame_intersection)
        frame_white_count = np.sum(white_pixels)
        frame_valid_count = np.sum(valid_points)
        
        if frame_idx < 3:  # 显示前3帧的统计
            logger.info(f"  帧{frame_idx}: 白色像素={frame_white_count}, 有效点={frame_valid_count}, 交集={frame_intersection_count}")
    
    # 7. 提取交集结果
    intersected_points = points_flat[intersection_mask]
    intersected_colors = colors_flat[intersection_mask]
    
    total_intersection = np.sum(intersection_mask)
    total_points = len(points_flat)
    intersection_ratio = total_intersection / total_points * 100 if total_points > 0 else 0
    
    logger.info(f"🎯 交集计算完成:")
    logger.info(f"  原始过滤点云: {total_points} 个点")
    logger.info(f"  交集结果: {total_intersection} 个点 ({intersection_ratio:.1f}%)")
    logger.info(f"  方法: 过滤后点云 ∩ mask白色区域")
    
    return intersected_points, intersected_colors, intersection_mask

def extract_colors_from_filtered_result(filtered_vggt_result: Dict, original_shape: tuple) -> np.ndarray:
    """从过滤后的VGGT结果中提取颜色信息"""
    try:
        if 'images' in filtered_vggt_result:
            images = filtered_vggt_result['images']
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            else:
                images_np = images
            
            if images_np.ndim == 5 and images_np.shape[0] == 1:
                images_np = np.squeeze(images_np, axis=0)
            
            # 确保颜色格式正确: (S, H, W, 3)
            if images_np.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                images_np = np.transpose(images_np, (0, 2, 3, 1))
            
            # 展平颜色数据
            if images_np.shape[:3] == original_shape[:3]:  # (S, H, W)
                colors_flat = images_np.reshape(-1, 3)
                if colors_flat.max() <= 1.0:
                    colors_flat = (colors_flat * 255).astype(np.uint8)
                else:
                    colors_flat = colors_flat.astype(np.uint8)
                return colors_flat
    except Exception as e:
        logger.warning(f"提取颜色失败: {e}")
    
    # 使用默认灰色
    total_points = np.prod(original_shape[:3])  # S * H * W
    return np.ones((total_points, 3), dtype=np.uint8) * 128

def determine_white_threshold(mask_sequence: List[np.ndarray]) -> float:
    """确定mask中白色区域的阈值"""
    all_values = set()
    for mask in mask_sequence[:3]:  # 只检查前3帧
        all_values.update(np.unique(mask))
    
    logger.info(f"Mask唯一值: {sorted(list(all_values))}")
    
    max_val = max(all_values) if all_values else 255
    if max_val <= 1.0:
        # 0-1范围的mask
        return 0.5
    elif len(all_values) == 2 and 0 in all_values:
        # 二值mask
        non_zero_vals = [v for v in all_values if v > 0]
        return non_zero_vals[0] / 2 if non_zero_vals else 127
    else:
        # 0-255范围的mask
        return 127

# -----------------------------------------------------------------------------
# 主要节点实现
# -----------------------------------------------------------------------------

class VGGTMaskProcessorNode:
    """VGGT Mask处理节点 - 计算过滤后点云与mask白色区域的交集"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filtered_vggt_result": ("RAW_VGGT_RESULT", {
                    "tooltip": "来自VGGTNativeFullOutputNode的过滤后结果"
                }),
                "mask_sequence": ("MASK", {
                    "tooltip": "mask图像序列，白色区域表示目标物体"
                }),
            },
            "optional": {
                "export_format": (["PLY", "GLB", "BOTH"], {
                    "default": "PLY",
                    "tooltip": "导出格式：PLY(点云)、GLB(网格)或两者都导出"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",            # 交集统计信息JSON
        "STRING",            # PLY文件路径
        "STRING",            # GLB文件路径  
        "STRING",            # 处理报告
    )
    RETURN_NAMES = (
        "intersection_stats",
        "intersected_ply_path",
        "intersected_glb_path", 
        "processing_report",
    )
    OUTPUT_TOOLTIPS = [
        "交集统计信息（JSON格式）",
        "交集点云PLY文件路径",
        "交集3D模型GLB文件路径",
        "详细的处理报告（JSON格式）"
    ]
    OUTPUT_NODE = True
    FUNCTION = "process_intersection"
    CATEGORY = "💃VVL/VGGT Mask"

    def process_intersection(self, filtered_vggt_result: Dict, mask_sequence,
                           export_format: str = "PLY"):
        """处理点云与mask的交集"""
        logger.info("开始VGGT点云与mask交集处理")
        
        try:
            # 处理mask序列输入
            mask_list = self._process_mask_sequence(mask_sequence)
            
            # 计算交集
            intersected_points, intersected_colors, intersection_mask = compute_pointcloud_mask_intersection(
                filtered_vggt_result, mask_list
            )
            
            # 生成统计信息
            total_points = len(intersection_mask)
            intersection_count = len(intersected_points)
            intersection_ratio = (intersection_count / total_points * 100) if total_points > 0 else 0.0
            
            stats = {
                "total_intersection_points": int(intersection_count),
                "total_points": int(total_points),
                "method": "filtered_pointcloud_intersect_mask_white_regions",
                "intersection_ratio": float(intersection_ratio)
            }
            stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
            
            # 创建输出目录
            if FOLDER_PATHS_AVAILABLE:
                output_dir = os.path.join(folder_paths.get_output_directory(), "intersected_models")
            else:
                output_dir = os.path.join("output", "intersected_models")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # 导出文件
            ply_path = ""
            glb_path = ""
            
            if len(intersected_points) == 0:
                logger.warning("交集为空，无法导出文件")
            else:
                if export_format in ["PLY", "BOTH"]:
                    ply_path = self._export_intersection_ply(
                        intersected_points, intersected_colors, output_dir, timestamp
                    )
                
                if export_format in ["GLB", "BOTH"]:
                    glb_path = self._export_intersection_glb(
                        intersected_points, intersected_colors, output_dir, timestamp,
                        filtered_vggt_result
                    )
            
            # 生成处理报告
            report = {
                "processing_method": "pointcloud_mask_intersection",
                "input_source": "filtered_vggt_result",
                "intersection_algorithm": "pixel_level_correspondence",
                "statistics": stats,
                "output_files": {
                    "ply_file": ply_path,
                    "glb_file": glb_path
                },
                "processing_time": time.time(),
                "status": "success" if (ply_path or glb_path) else "empty_intersection"
            }
            report_json = json.dumps(report, ensure_ascii=False, indent=2)
            
            logger.info("VGGT点云与mask交集处理完成")
            return (stats_json, ply_path, glb_path, report_json)
            
        except Exception as e:
            logger.error(f"交集处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"交集处理失败: {str(e)}"
            error_json = json.dumps({"error": str(e)})
            return (error_json, "", "", error_msg)
    
    def _process_mask_sequence(self, mask_sequence):
        """处理mask序列输入"""
        mask_list = []
        if isinstance(mask_sequence, torch.Tensor):
            mask_np = mask_sequence.cpu().numpy()
            logger.info(f"Tensor mask形状: {mask_np.shape}")
            
            for i in range(mask_np.shape[0]):
                mask_img = mask_np[i]
                
                # 如果是RGB图像，转换为单通道
                if mask_img.ndim == 3:
                    if mask_img.shape[2] >= 3:
                        r, g, b = mask_img[:,:,0], mask_img[:,:,1], mask_img[:,:,2]
                        if np.allclose(r, g) and np.allclose(g, b):
                            mask_img = r  # 使用第一个通道
                        else:
                            mask_img = np.mean(mask_img, axis=2)
                    else:
                        mask_img = mask_img[:,:,0]
                
                # 处理值范围
                if mask_img.max() <= 1.0:
                    mask_img = (mask_img * 255).astype(np.uint8)
                else:
                    mask_img = mask_img.astype(np.uint8)
                
                mask_list.append(mask_img)
        else:
            mask_list = list(mask_sequence)
        
        logger.info(f"处理了 {len(mask_list)} 张mask图像")
        return mask_list
    
    def _export_intersection_ply(self, points: np.ndarray, colors: np.ndarray, 
                               output_dir: str, timestamp: int) -> str:
        """导出交集点云PLY文件"""
        try:
            filename = f"vggt_intersection_{timestamp}.ply"
            ply_path = os.path.join(output_dir, filename)
            
            self._write_ply_file(ply_path, points, colors)
            
            logger.info(f"交集PLY已保存: {ply_path} ({len(points)} 个点)")
            return ply_path
            
        except Exception as e:
            logger.error(f"导出交集PLY失败: {e}")
            return ""
    
    def _export_intersection_glb(self, points: np.ndarray, colors: np.ndarray, 
                               output_dir: str, timestamp: int,
                               filtered_vggt_result: Dict) -> str:
        """导出交集GLB文件"""
        if not VGGT_UTILS_AVAILABLE or not predictions_to_glb:
            logger.warning("GLB导出功能不可用")
            return ""
        
        try:
            filename = f"vggt_intersection_{timestamp}.glb"
            glb_path = os.path.join(output_dir, filename)
            
            logger.info(f"生成交集GLB文件: {glb_path}")
            
            # 重新组织交集点云为GLB兼容格式
            predictions_formatted = self._create_intersection_predictions(
                points, colors, filtered_vggt_result
            )
            
            # 使用官方VGGT的predictions_to_glb函数
            scene_3d = predictions_to_glb(
                predictions_formatted,
                conf_thres=0.0,  # 不再过滤，因为已经是交集结果
                filter_by_frames="all",
                mask_black_bg=False,
                mask_white_bg=False,
                show_cam=False,  # 不显示相机
                mask_sky=False,
                target_dir=None,
                prediction_mode="Depthmap and Camera Branch"
            )
            
            # 导出为GLB文件
            scene_3d.export(glb_path)
            
            logger.info(f"交集GLB已保存: {glb_path}")
            return glb_path
            
        except Exception as e:
            logger.error(f"导出交集GLB失败: {e}")
            return ""
    
    def _create_intersection_predictions(self, points: np.ndarray, colors: np.ndarray,
                                       filtered_vggt_result: Dict) -> Dict:
        """为GLB导出创建交集预测数据"""
        logger.info(f"为GLB导出创建交集预测数据，点数: {len(points)}")
        
        # 计算合适的网格尺寸
        n_points = len(points)
        if n_points < 100:
            h = w = 16
        elif n_points < 1000:
            h = w = 32
        else:
            side = int(np.ceil(np.sqrt(n_points)))
            h = w = max(side, 16)
        
        target_points = h * w
        
        # 如果需要填充点
        if n_points < target_points:
            indices = np.random.choice(n_points, target_points - n_points, replace=True)
            padding_points = points[indices]
            padding_colors = colors[indices]
            
            padded_points = np.vstack([points, padding_points])
            padded_colors = np.vstack([colors, padding_colors])
        elif n_points > target_points:
            indices = np.random.choice(n_points, target_points, replace=False)
            padded_points = points[indices]
            padded_colors = colors[indices]
        else:
            padded_points = points
            padded_colors = colors
        
        # 重塑为GLB需要的格式
        world_points = padded_points.reshape(1, h, w, 3)
        
        # 处理颜色
        if padded_colors.max() > 1.0:
            colors_normalized = padded_colors / 255.0
        else:
            colors_normalized = padded_colors
        
        images_reshaped = colors_normalized.reshape(1, h, w, 3)
        
        # 创建置信度数据（交集点给予高置信度）
        depth_conf = np.ones((1, h, w), dtype=np.float32)
        
        # 获取相机参数（如果有的话）
        extrinsic = None
        if 'cameras' in filtered_vggt_result:
            cameras = filtered_vggt_result['cameras']
            if isinstance(cameras, dict):
                extrinsic = cameras.get('extrinsic')
                if isinstance(extrinsic, torch.Tensor):
                    extrinsic = extrinsic.cpu().numpy()
                if extrinsic is not None and extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
                    extrinsic = np.squeeze(extrinsic, axis=0)
        
        if extrinsic is None:
            extrinsic = np.eye(4)[None, :3, :]  # 默认单位矩阵
        
        # 构建预测结果
        predictions = {
            'world_points_from_depth': world_points.astype(np.float32),
            'depth_conf': depth_conf,
            'images': images_reshaped.astype(np.float32),
            'extrinsic': extrinsic,
        }
        
        logger.info(f"交集GLB预测数据创建完成: points={world_points.shape}")
        return predictions
    
    def _write_ply_file(self, filepath: str, vertices: np.ndarray, colors: np.ndarray = None):
        """写入PLY格式文件"""
        try:
            with open(filepath, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"comment VGGT filtered pointcloud intersect mask white regions\n")
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
    "VGGTMaskProcessorNode": VGGTMaskProcessorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTMaskProcessorNode": "🎭 VGGT Intersection Processor",
} 