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
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以查看详细信息

# -----------------------------------------------------------------------------
# Mask处理核心函数
# -----------------------------------------------------------------------------

def process_tracks_with_masks(tracks_data: Dict, mask_sequence: List[np.ndarray], 
                             confidence_threshold: float = 0.5) -> Dict[int, List[int]]:
    """
    使用tracks数据和mask序列进行点分割
    
    Args:
        tracks_data: VGGT的tracks数据，包含track_list, visibility_score, confidence_score
        mask_sequence: mask图像序列，每张图像包含不同的分割ID
        confidence_threshold: 置信度阈值，低于此值的点会被忽略
    
    Returns:
        point_labels: {point_id: [object_ids]} 每个点投票得到的物体ID列表
    """
    logger.info(f"开始处理tracks数据进行mask分割，使用固定mask阈值: 128")
    
    # 提取tracks数据
    track_list = tracks_data.get('track_list', [])
    visibility_score = tracks_data.get('visibility_score', [])
    confidence_score = tracks_data.get('confidence_score', [])
    
    if track_list is None or (isinstance(track_list, (list, tuple)) and len(track_list) == 0) or (isinstance(track_list, np.ndarray) and track_list.size == 0):
        logger.warning("No track_list found in tracks_data")
        return {}
    
    # 转换为numpy数组 - 更强力的tensor检测
    logger.info(f"track_list类型: {type(track_list)}, 是否有cpu方法: {hasattr(track_list, 'cpu')}")
    
    if hasattr(track_list, 'cpu') and hasattr(track_list, 'numpy'):
        # PyTorch tensor
        track_coords = track_list.cpu().numpy()
        logger.info(f"从tensor转换track_coords, shape: {track_coords.shape}")
    elif isinstance(track_list, np.ndarray):
        track_coords = track_list
        logger.info(f"使用numpy track_coords, shape: {track_coords.shape}")
    elif hasattr(track_list, 'cpu'):  # 可能是tensor但没有numpy方法
        track_coords = track_list.cpu().detach().numpy()
        logger.info(f"强制转换tensor track_coords, shape: {track_coords.shape}")
    else:
        try:
            track_coords = np.array(track_list)
            logger.info(f"数组转换track_coords, shape: {track_coords.shape}")
        except Exception as e:
            logger.error(f"track_list转换失败: {e}, type: {type(track_list)}")
            # 尝试递归处理列表中的tensor元素
            if isinstance(track_list, (list, tuple)) and len(track_list) > 0:
                converted_list = []
                for item in track_list:
                    if hasattr(item, 'cpu'):
                        converted_list.append(item.cpu().numpy())
                    else:
                        converted_list.append(item)
                track_coords = np.array(converted_list)
                logger.info(f"递归转换track_coords成功, shape: {track_coords.shape}")
            else:
                raise ValueError(f"无法转换track_list, type: {type(track_list)}")
    
    if hasattr(visibility_score, 'cpu'):
        vis_scores = visibility_score.cpu().numpy()
    elif isinstance(visibility_score, np.ndarray):
        vis_scores = visibility_score
    else:
        vis_scores = np.array(visibility_score) if visibility_score is not None else None
    
    if hasattr(confidence_score, 'cpu'):
        conf_scores = confidence_score.cpu().numpy()
    elif isinstance(confidence_score, np.ndarray):
        conf_scores = confidence_score
    else:
        conf_scores = np.array(confidence_score) if confidence_score is not None else None
    
    # 重塑vis_scores和conf_scores以匹配track_coords - 需要在重塑track_coords之后
    pass  # 先重塑track_coords，然后再处理score
    
    logger.info(f"Track坐标形状: {track_coords.shape}")
    logger.info(f"Mask序列长度: {len(mask_sequence)}")
    
    # 处理不同的track_coords格式
    if track_coords.ndim == 5:  # (batch, 1, frames, points, 2) 
        logger.info("检测到5D格式，重塑为标准格式")
        if track_coords.shape[1] == 1:  # 去掉多余的维度
            track_coords = track_coords[:, 0]  # (batch, frames, points, 2)
        # 合并batch和points维度: (batch*points, frames, 2)
        batch_size, n_frames, n_points_per_batch, coord_dim = track_coords.shape
        track_coords = track_coords.reshape(-1, n_frames, coord_dim)  # (batch*points, frames, 2)
        # 转置为 (points, frames, 2)
        track_coords = track_coords.swapaxes(0, 1).swapaxes(0, 1)
        logger.info(f"重塑后形状: {track_coords.shape}")
    elif track_coords.ndim == 4:  # (batch, frames, points, 2) 或 (1, frames, points, 2)
        if track_coords.shape[0] == 1:
            track_coords = track_coords.squeeze(0)  # 去掉batch维度: (frames, points, 2)
            track_coords = track_coords.swapaxes(0, 1)  # 转置为 (points, frames, 2)
        else:
            # 多个batch的情况
            batch_size, n_frames, n_points_per_batch, coord_dim = track_coords.shape
            track_coords = track_coords.reshape(-1, n_frames, coord_dim)
    elif track_coords.ndim == 3:  # 已经是 (points, frames, 2) 或 (frames, points, 2)
        if track_coords.shape[2] == 2:  # (frames, points, 2) -> (points, frames, 2)
            track_coords = track_coords.swapaxes(0, 1)
    
    # 确保最终格式是 (N_points, N_frames, 2)
    if track_coords.shape[2] != 2:
        logger.error(f"意外的坐标维度: {track_coords.shape}")
        raise ValueError(f"期望最后一维为2（x,y坐标），但得到: {track_coords.shape}")
    
    n_points, n_frames, _ = track_coords.shape
    logger.info(f"处理 {n_points} 个点，{n_frames} 帧")
    
    # 现在重塑vis_scores和conf_scores以匹配重塑后的track_coords
    if vis_scores is not None:
        logger.info(f"原始visibility_score形状: {vis_scores.shape}")
        if vis_scores.ndim == 3 and vis_scores.shape[0] == 1:  # (1, frames, points_per_batch)
            vis_scores = vis_scores.squeeze(0)  # (frames, points_per_batch)
            vis_scores = vis_scores.T  # 转置为 (points_per_batch, frames)
            # 如果有多个batch，需要复制数据以匹配总点数
            if n_points > vis_scores.shape[0]:
                batch_count = n_points // vis_scores.shape[0]
                vis_scores = np.tile(vis_scores, (batch_count, 1))  # 复制到所有batch
        elif vis_scores.ndim == 2:  # (frames, points) 或 (points, frames)
            if vis_scores.shape[0] == n_frames:  # (frames, points) -> (points, frames)
                vis_scores = vis_scores.T
            # 检查是否需要扩展到匹配点数
            if vis_scores.shape[0] != n_points:
                batch_count = n_points // vis_scores.shape[0]
                vis_scores = np.tile(vis_scores, (batch_count, 1))
        logger.info(f"重塑后visibility_score形状: {vis_scores.shape}")
    
    if conf_scores is not None:
        logger.info(f"原始confidence_score形状: {conf_scores.shape}")
        if conf_scores.ndim == 3 and conf_scores.shape[0] == 1:  # (1, frames, points_per_batch)
            conf_scores = conf_scores.squeeze(0)  # (frames, points_per_batch)
            conf_scores = conf_scores.T  # 转置为 (points_per_batch, frames)
            # 如果有多个batch，需要复制数据以匹配总点数
            if n_points > conf_scores.shape[0]:
                batch_count = n_points // conf_scores.shape[0]
                conf_scores = np.tile(conf_scores, (batch_count, 1))  # 复制到所有batch
        elif conf_scores.ndim == 2:  # (frames, points) 或 (points, frames)
            if conf_scores.shape[0] == n_frames:  # (frames, points) -> (points, frames)
                conf_scores = conf_scores.T
            # 检查是否需要扩展到匹配点数
            if conf_scores.shape[0] != n_points:
                batch_count = n_points // conf_scores.shape[0]
                conf_scores = np.tile(conf_scores, (batch_count, 1))
        logger.info(f"重塑后confidence_score形状: {conf_scores.shape}")
    
    # 对每个点进行投票分割 - 添加详细调试
    point_labels = {}
    
    # 调试信息：检查mask图像的值范围和物体覆盖率
    logger.info("检查mask图像信息:")
    for i, mask_img in enumerate(mask_sequence[:3]):  # 只检查前3帧
        unique_values = np.unique(mask_img)
        object_pixels = np.sum(mask_img > 128)
        total_pixels = mask_img.size
        coverage = object_pixels / total_pixels * 100
        logger.info(f"  Frame {i}: 图像形状={mask_img.shape}, 值范围={unique_values}, 物体覆盖率={coverage:.1f}%")
    
    # 调试：检查更多点的坐标范围，包括所有点
    sample_coords = track_coords[:, :3]  # 所有点，前3帧
    logger.info(f"所有点坐标范围 (前3帧): x=[{sample_coords[:,:,0].min():.1f}, {sample_coords[:,:,0].max():.1f}], y=[{sample_coords[:,:,1].min():.1f}, {sample_coords[:,:,1].max():.1f}]")
    
    # 检查前几个点在第一帧的mask值
    first_mask = mask_sequence[0]
    logger.info("前20个点在第一帧的mask采样:")
    for i in range(min(20, n_points)):
        x, y = track_coords[i, 0]
        x_int = max(0, min(int(round(x)), first_mask.shape[1] - 1))
        y_int = max(0, min(int(round(y)), first_mask.shape[0] - 1))
        mask_val = first_mask[y_int, x_int]
        logger.info(f"  点{i}: ({x:.0f},{y:.0f}) -> mask={mask_val}")
        if i == 9:  # 分两行显示
            break
    
    for i in range(10, min(20, n_points)):
        x, y = track_coords[i, 0]
        x_int = max(0, min(int(round(x)), first_mask.shape[1] - 1))
        y_int = max(0, min(int(round(y)), first_mask.shape[0] - 1))
        mask_val = first_mask[y_int, x_int]
        logger.info(f"  点{i}: ({x:.0f},{y:.0f}) -> mask={mask_val}")
    
    valid_votes_count = 0
    filtered_by_visibility = 0
    filtered_by_confidence = 0
    out_of_bounds_count = 0
    
    for point_idx in range(n_points):
        votes = defaultdict(int)  # {object_id: vote_count}
        total_votes = 0
        point_valid_votes = 0
        
        for frame_idx in range(min(n_frames, len(mask_sequence))):
            # 获取当前帧的坐标
            x, y = track_coords[point_idx, frame_idx]
            
            # 检查可见性和置信度
            is_visible = True
            vis_score = 1.0
            conf_score = 1.0
            
            if vis_scores is not None:
                vis_score = vis_scores[point_idx, frame_idx] if vis_scores.ndim >= 2 else vis_scores[point_idx]
                if vis_score <= 0.5:
                    is_visible = False
                    filtered_by_visibility += 1
            
            if conf_scores is not None:
                conf_score = conf_scores[point_idx, frame_idx] if conf_scores.ndim >= 2 else conf_scores[point_idx]
                if conf_score <= confidence_threshold:
                    is_visible = False
                    filtered_by_confidence += 1
            
            if not is_visible:
                continue
                
            # 获取mask图像
            mask_img = mask_sequence[frame_idx]
            h, w = mask_img.shape[:2]
            
            # 检查坐标是否在范围内
            if x < 0 or x >= w or y < 0 or y >= h:
                out_of_bounds_count += 1
                continue
            
            # 确保坐标在图像范围内
            x_int = max(0, min(int(round(x)), w - 1))
            y_int = max(0, min(int(round(y)), h - 1))
            
            # 获取该位置的物体ID
            if mask_img.ndim == 2:
                raw_id = mask_img[y_int, x_int]
            else:
                raw_id = mask_img[y_int, x_int, 0]  # 假设第一个通道是ID
            
            # 处理mask值：255 -> 1, 0 -> 0 (将255映射为物体ID 1)
            if raw_id > 128:  # 假设超过阈值的值都是物体
                object_id = 1  # 简化为单一物体
            else:
                object_id = 0  # 背景
            
            # 调试信息：记录前几个点的详细投票过程
            if point_idx < 3 and frame_idx < 3:
                logger.info(f"    点{point_idx} 帧{frame_idx}: 坐标=({x:.1f},{y:.1f}) -> ({x_int},{y_int}), vis={vis_score:.3f}, conf={conf_score:.3f}, raw_mask={raw_id}, object_id={object_id}")
            
            # 投票（包括背景ID=0）
            votes[int(object_id)] += 1  # 确保使用int类型
            total_votes += 1
            point_valid_votes += 1
            
            if object_id > 0:
                valid_votes_count += 1
        
        # 记录投票结果
        if total_votes > 0:
            # 按投票数排序，取最高票的物体ID
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            point_labels[point_idx] = [obj_id for obj_id, count in sorted_votes if count >= total_votes * 0.3]  # 至少30%的票
        else:
            point_labels[point_idx] = [0]  # 未分类
        
        # 调试：记录前几个点的投票结果
        if point_idx < 5:
            logger.info(f"  点{point_idx}投票结果: 总票数={total_votes}, 有效票数={point_valid_votes}, 最终标签={point_labels[point_idx]}")
    
    logger.info(f"投票统计: 有效非背景票数={valid_votes_count}, 被可见性过滤={filtered_by_visibility}, 被置信度过滤={filtered_by_confidence}, 坐标越界={out_of_bounds_count}")
    
    # 如果没有找到物体点，尝试基于mask生成虚拟tracks
    if valid_votes_count == 0:
        logger.warning("所有查询点都落在背景区域，尝试基于mask生成虚拟物体点")
        # 注释掉虚拟点生成，因为我们现在使用更好的完整点云分割方法
        # virtual_points = generate_virtual_object_points(mask_sequence, n_virtual_points=50)
        
        # if virtual_points:
        #     logger.info(f"生成了 {len(virtual_points)} 个虚拟物体点")
        #     # 为虚拟点添加到point_labels
        #     for i, (frame_idx, x, y, object_id) in enumerate(virtual_points):
        #         virtual_point_idx = n_points + i  # 避免与现有点冲突
        #         point_labels[virtual_point_idx] = [object_id]
        #         logger.info(f"  虚拟点{virtual_point_idx}: 帧{frame_idx} 坐标({x},{y}) -> 物体{object_id}")
        # else:
        #     logger.error("无法生成虚拟物体点，mask中可能没有物体区域")
        logger.info("将使用完整点云分割方法来查找物体点")
    
    logger.info(f"完成点分割，{len(point_labels)} 个点被分类")
    
    # 统计分割结果
    object_counts = defaultdict(int)
    for labels in point_labels.values():
        for label in labels:
            object_counts[label] += 1
    
    logger.info(f"分割统计: {dict(object_counts)}")
    
    return point_labels

def generate_virtual_object_points(mask_sequence: List[np.ndarray], n_virtual_points: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    基于mask序列生成虚拟物体点
    
    Args:
        mask_sequence: mask图像序列
        n_virtual_points: 要生成的虚拟点数量
    
    Returns:
        List of (frame_idx, x, y, object_id) tuples
    """
    virtual_points = []
    
    for frame_idx, mask_img in enumerate(mask_sequence):
        # 找到物体像素位置
        object_pixels = np.where(mask_img > 128)
        
        if len(object_pixels[0]) > 0:
            # 随机采样物体像素位置
            n_samples = min(max(1, n_virtual_points // len(mask_sequence)), len(object_pixels[0]))
            
            if n_samples > 0:
                indices = np.random.choice(len(object_pixels[0]), n_samples, replace=False)
                
                for idx in indices:
                    y = int(object_pixels[0][idx])
                    x = int(object_pixels[1][idx])
                    object_id = 1  # 简化为单一物体
                    virtual_points.append((frame_idx, x, y, object_id))
                
                # 只处理前几帧就够了
                if len(virtual_points) >= n_virtual_points:
                    break
    
    return virtual_points[:n_virtual_points]

def create_mask_based_segmentation(raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                 target_object_id: int = None) -> np.ndarray:
    """
    基于mask序列对完整点云进行分割（正确的方法）
    
    Args:
        raw_vggt_result: VGGT原生结果
        mask_sequence: mask图像序列
        target_object_id: 目标物体ID，None表示所有非背景物体
    
    Returns:
        segment_mask: 布尔掩码数组，指示哪些点属于目标物体
    """
    # 使用固定的初始阈值，然后通过自动检测调整
    mask_threshold = 128
    logger.info(f"开始基于mask的真正点云分割，初始mask阈值: {mask_threshold}")
    
    # 强制检查mask序列
    logger.info(f"🔍 强制检查mask序列: type={type(mask_sequence)}, len={len(mask_sequence) if hasattr(mask_sequence, '__len__') else 'N/A'}")
    if hasattr(mask_sequence, '__len__') and len(mask_sequence) > 0:
        first_mask = mask_sequence[0]
        logger.info(f"🔍 第一个mask: type={type(first_mask)}")
        if hasattr(first_mask, 'shape'):
            logger.info(f"🔍 第一个mask形状: {first_mask.shape}")
        if hasattr(first_mask, 'dtype'):
            logger.info(f"🔍 第一个mask dtype: {first_mask.dtype}")
        if hasattr(first_mask, 'min'):
            logger.info(f"🔍 第一个mask值范围: [{first_mask.min()}, {first_mask.max()}]")
        if hasattr(first_mask, 'unique'):
            unique_vals = np.unique(first_mask) if hasattr(np, 'unique') else 'N/A'
            logger.info(f"🔍 第一个mask唯一值: {unique_vals}")
    
    # 首先分析mask序列的值范围，自动调整阈值
    logger.info("分析mask序列的值分布...")
    all_unique_values = set()
    max_val_in_sequence = 0
    min_val_in_sequence = float('inf')
    
    for i, mask in enumerate(mask_sequence[:5]):  # 只检查前5帧
        unique_vals = np.unique(mask)
        all_unique_values.update(unique_vals)
        max_val_in_sequence = max(max_val_in_sequence, mask.max())
        min_val_in_sequence = min(min_val_in_sequence, mask.min())
        
        if i < 3:  # 详细显示前3帧
            logger.info(f"  Mask {i}: shape={mask.shape}, dtype={mask.dtype}, "
                       f"range=[{mask.min()}, {mask.max()}], unique_count={len(unique_vals)}")
    
    logger.info(f"序列总体: min={min_val_in_sequence}, max={max_val_in_sequence}, "
               f"unique_values={sorted(list(all_unique_values))[:20]}...")
    
    # 自动调整mask阈值
    original_threshold = mask_threshold
    if max_val_in_sequence <= 1.0:
        # 0-1范围的mask，调整阈值
        mask_threshold = 0.5
        logger.warning(f"检测到0-1范围的mask，自动调整阈值: {original_threshold} -> {mask_threshold}")
    elif max_val_in_sequence < original_threshold:
        # 最大值小于阈值，使用最大值的一半
        mask_threshold = max_val_in_sequence / 2
        logger.warning(f"Mask最大值({max_val_in_sequence})小于原阈值({original_threshold})，"
                      f"自动调整阈值为: {mask_threshold}")
    elif len(all_unique_values) == 2 and 0 in all_unique_values:
        # 二值mask，使用两个值的中间值
        non_zero_vals = [v for v in all_unique_values if v > 0]
        if non_zero_vals:
            mask_threshold = non_zero_vals[0] / 2
            logger.info(f"检测到二值mask (0, {non_zero_vals[0]})，调整阈值为: {mask_threshold}")
    
    logger.info(f"最终使用的mask阈值: {mask_threshold}")
    
    # 获取完整点云数据
    if 'points_from_depth' in raw_vggt_result:
        points = raw_vggt_result['points_from_depth']
        logger.info("使用 points_from_depth 进行分割")
    elif 'points' in raw_vggt_result:
        if isinstance(raw_vggt_result['points'], dict):
            points = raw_vggt_result['points']['point_map']
        else:
            points = raw_vggt_result['points']
        logger.info("使用 points 进行分割")
    else:
        raise ValueError("No point cloud data found")
    
    # 转换为numpy
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    # 去掉batch维度并重塑
    if points_np.ndim == 5 and points_np.shape[0] == 1:
        points_np = np.squeeze(points_np, axis=0)
    
    # 记录原始形状以便重建
    original_shape = points_np.shape
    logger.info(f"点云原始形状: {original_shape}")
    
    # 检查点云分辨率
    if len(original_shape) == 4:  # (S, H, W, 3)
        S, H, W, _ = original_shape
        logger.info(f"点云分辨率: {W}x{H}, 帧数: {S}")
        if len(mask_sequence) > 0:
            mask_h, mask_w = mask_sequence[0].shape
            logger.info(f"Mask分辨率: {mask_w}x{mask_h}")
            if W != mask_w or H != mask_h:
                logger.warning(f"⚠️ 点云分辨率({W}x{H})与Mask分辨率({mask_w}x{mask_h})不匹配！")
    
    # 重塑为 (N_points, 3)
    if points_np.ndim == 4:  # (S, H, W, 3)
        S, H, W, _ = points_np.shape
        points_flat = points_np.reshape(-1, 3)
    elif points_np.ndim == 3:  # (N, H, W) or similar
        points_flat = points_np.reshape(-1, 3)
    else:
        points_flat = points_np
    
    # 过滤无效点
    valid_mask = ~np.isnan(points_flat).any(axis=1) & ~np.isinf(points_flat).any(axis=1)
    valid_points = points_flat[valid_mask]
    logger.info(f"有效点数: {len(valid_points)}/{len(points_flat)}")
    
    # 获取相机参数
    cameras = raw_vggt_result.get('cameras', {})
    extrinsic = cameras.get('extrinsic')
    intrinsic = cameras.get('intrinsic')
    
    # 检查是否可以使用直接像素对应方法
    USE_PIXEL_CORRESPONDENCE = False
    if len(original_shape) == 4:  # (S, H, W, 3)
        S, H, W, _ = original_shape
        if len(mask_sequence) > 0:
            mask_h, mask_w = mask_sequence[0].shape
            if W == mask_w and H == mask_h:
                logger.info("✅ 点云和Mask分辨率匹配，使用直接像素对应方法")
                USE_PIXEL_CORRESPONDENCE = True
    
    # 临时强制使用简单分割来测试
    USE_SIMPLE_SEGMENTATION = False  # 设为True来测试简单分割
    
    if USE_PIXEL_CORRESPONDENCE:
        # 使用直接像素对应方法（最简单最准确）
        return create_pixel_correspondence_segmentation(
            points_flat, mask_sequence, valid_mask, original_shape
        )
    elif extrinsic is None or intrinsic is None or USE_SIMPLE_SEGMENTATION:
        logger.warning("缺少相机参数或强制使用简单分割")
        return create_simple_spatial_segmentation(valid_points, mask_sequence, valid_mask)
    
    # 转换相机参数
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    
    # 去掉batch维度
    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = np.squeeze(extrinsic, axis=0)
    if intrinsic.ndim == 3 and intrinsic.shape[0] == 1:
        intrinsic = np.squeeze(intrinsic, axis=0)
    
    logger.info(f"相机外参形状: {extrinsic.shape}, 内参形状: {intrinsic.shape}")
    logger.info(f"Mask序列长度: {len(mask_sequence)}")
    
    # 检查是否需要缩放mask（在循环之前决定）
    need_resize_mask = False
    if len(mask_sequence) > 0:
        original_mask_h, original_mask_w = mask_sequence[0].shape
        if original_mask_w == 1024 and original_mask_h == 896:
            # 检查内参是否是针对518x448分辨率的
            scale_x = original_mask_w / 518.0
            scale_y = original_mask_h / 448.0
            if abs(scale_x - 2.0) < 0.1 and abs(scale_y - 2.0) < 0.1:
                need_resize_mask = True
                logger.info("将在投影过程中缩放mask从1024x896到518x448以匹配点云分辨率")
    
    # 使用更大的批次处理，加快速度
    n_points = len(valid_points)
    batch_size = 500000  # 增大到50万个点，减少批次数
    n_frames = min(len(mask_sequence), extrinsic.shape[0])
    
    logger.info(f"处理 {n_frames} 帧进行点云投影分割，总点数: {n_points}")
    
    # 初始化点标签和置信度
    point_labels = np.zeros(n_points, dtype=np.uint8)
    confidence_scores = np.zeros(n_points, dtype=np.float32)
    
    # 添加调试统计
    total_valid_projections = 0
    total_object_hits = 0
    
    # 采样处理以加速 - 如果点太多，只处理一部分
    if n_points > 1000000:  # 超过100万点时采样
        sample_rate = 0.2  # 采样20%的点
        sample_indices = np.random.choice(n_points, int(n_points * sample_rate), replace=False)
        sampled_points = valid_points[sample_indices]
        logger.info(f"点云过大，采样处理 {len(sampled_points)} 个点 ({sample_rate*100:.0f}%)")
    else:
        sampled_points = valid_points
        sample_indices = np.arange(n_points)
    
    # 分批处理采样后的点云
    total_batches = (len(sampled_points) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(range(0, len(sampled_points), batch_size)):
        batch_end = min(batch_start + batch_size, len(sampled_points))
        batch_points = sampled_points[batch_start:batch_end]
        batch_votes = np.zeros((len(batch_points), 256), dtype=np.int16)
        
        # 显示进度（减少日志输出）
        if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
        
        # 第一批添加详细调试
        if batch_idx == 0:
            logger.info(f"第一批点云范围: X=[{batch_points[:, 0].min():.2f}, {batch_points[:, 0].max():.2f}], "
                       f"Y=[{batch_points[:, 1].min():.2f}, {batch_points[:, 1].max():.2f}], "
                       f"Z=[{batch_points[:, 2].min():.2f}, {batch_points[:, 2].max():.2f}]")
        
        # 只处理关键帧以加速（每隔3帧处理一次）
        frame_step = 3
        processed_frames = list(range(0, n_frames, frame_step))
        
        for frame_idx in processed_frames:
            if frame_idx >= len(mask_sequence):
                break
                
            mask = mask_sequence[frame_idx]
            
            # 如果需要缩放mask到518x448
            if need_resize_mask:
                original_h, original_w = mask.shape
                if original_w == 1024 and original_h == 896:
                    import cv2
                    mask = cv2.resize(mask, (518, 448), interpolation=cv2.INTER_NEAREST)
                    if batch_idx == 0 and frame_idx == 0:
                        logger.info(f"缩放mask从{original_w}x{original_h}到518x448以匹配点云分辨率")
            
            # 获取当前帧的相机参数
            if extrinsic.shape[0] > frame_idx:
                extr = extrinsic[frame_idx]  # (3, 4) or (4, 4)
                if extr.shape[0] == 3:
                    # 补充为4x4矩阵
                    extr_4x4 = np.eye(4)
                    extr_4x4[:3, :] = extr
                    extr = extr_4x4
            else:
                extr = np.eye(4)
            
            # 处理内参矩阵的不同格式
            if intrinsic.ndim == 4:  # (1, N, 3, 3)
                intr = intrinsic[0, frame_idx] if frame_idx < intrinsic.shape[1] else intrinsic[0, 0]
            elif intrinsic.ndim == 3:  # (N, 3, 3)
                intr = intrinsic[frame_idx] if frame_idx < intrinsic.shape[0] else intrinsic[0]
            elif intrinsic.ndim == 2:  # (3, 3)
                intr = intrinsic
            else:
                logger.warning(f"内参矩阵形状异常: {intrinsic.shape}，使用默认值")
                intr = np.eye(3)
                intr[0, 0] = 1000  # fx
                intr[1, 1] = 1000  # fy
                intr[0, 2] = 512   # cx
                intr[1, 2] = 448   # cy
            
            # 在第一帧打印内参矩阵信息（在获取mask尺寸之前）
            if batch_idx == 0 and frame_idx == 0:
                logger.info(f"内参矩阵:\n{intr}")
                logger.info(f"  fx={intr[0,0]:.1f}, fy={intr[1,1]:.1f}")
                logger.info(f"  cx={intr[0,2]:.1f}, cy={intr[1,2]:.1f}")
                # 获取原始mask尺寸
                original_mask_h, original_mask_w = mask_sequence[0].shape
                logger.info(f"  原始mask尺寸: {original_mask_w}x{original_mask_h}")
                logger.info(f"  是否需要缩放mask: {need_resize_mask}")
                logger.info(f"保持内参不变: fx={intr[0,0]:.1f}, fy={intr[1,1]:.1f}, "
                           f"cx={intr[0,2]:.1f}, cy={intr[1,2]:.1f}")
            
            # 投影3D点到2D图像
            # 检查外参矩阵是否接近单位矩阵
            is_identity = np.allclose(extr[:3, :3], np.eye(3), atol=0.01)
            if is_identity:
                # 如果外参接近单位矩阵，使用简化投影（假设点云已在相机坐标系）
                projected_points = project_3d_to_2d_simple(batch_points, intr)
                if batch_idx == 0 and frame_idx == 0:
                    logger.info("使用简化投影（点云已在相机坐标系）")
            else:
                # 否则使用完整投影
                projected_points = project_3d_to_2d(batch_points, extr, intr)
                if batch_idx == 0 and frame_idx == 0:
                    logger.info("使用完整投影（需要外参变换）")
            
            # 获取当前mask的尺寸
            h, w = mask.shape
            
            # 第一批第一帧的调试信息
            if batch_idx == 0 and frame_idx == 0:
                logger.info(f"第一批第一帧投影结果:")
                logger.info(f"  投影点范围: X=[{projected_points[:, 0].min():.1f}, {projected_points[:, 0].max():.1f}], "
                           f"Y=[{projected_points[:, 1].min():.1f}, {projected_points[:, 1].max():.1f}]")
                logger.info(f"  图像尺寸: {w}x{h}")
                logger.info(f"  外参矩阵是否接近单位矩阵: {np.allclose(extr[:3, :3], np.eye(3), atol=0.01)}")
                
                # 重要：详细检查mask图像
                logger.info(f"  Mask图像详细信息:")
                logger.info(f"    - dtype: {mask.dtype}")
                logger.info(f"    - shape: {mask.shape}")
                logger.info(f"    - 值范围: [{mask.min()}, {mask.max()}]")
                logger.info(f"    - 唯一值: {np.unique(mask)[:20]}...")  # 只显示前20个
                logger.info(f"    - 值>mask_threshold({mask_threshold})的像素数: {np.sum(mask > mask_threshold)}")
                logger.info(f"    - 值==255的像素数: {np.sum(mask == 255)}")
                logger.info(f"    - 值==1的像素数: {np.sum(mask == 1)}")
                logger.info(f"    - 值==0的像素数: {np.sum(mask == 0)}")
                
                # 检查前10个点的投影结果和对应的mask值
                for i in range(min(10, len(projected_points))):
                    logger.info(f"  点{i}: 3D({batch_points[i, 0]:.2f}, {batch_points[i, 1]:.2f}, {batch_points[i, 2]:.2f}) "
                               f"-> 2D({projected_points[i, 0]:.1f}, {projected_points[i, 1]:.1f})")
                    # 检查是否在图像范围内
                    in_range = (0 <= projected_points[i, 0] < w) and (0 <= projected_points[i, 1] < h)
                    if in_range:
                        x_int = int(projected_points[i, 0])
                        y_int = int(projected_points[i, 1])
                        mask_val = mask[y_int, x_int]
                        # 检查周围的mask值
                        y_start, y_end = max(0, y_int-1), min(h, y_int+2)
                        x_start, x_end = max(0, x_int-1), min(w, x_int+2)
                        neighbor_vals = mask[y_start:y_end, x_start:x_end]
                        logger.info(f"    -> 在图像内({x_int},{y_int}), mask值={mask_val}, 周围3x3={neighbor_vals.flatten()}")
                    else:
                        logger.info(f"    -> 超出图像范围")
                
                # 如果mask全是0，检查是否应该调整阈值
                if mask.max() <= 1.0:
                    logger.warning(f"  ⚠️  检测到mask值范围[0,1]，可能需要将mask_threshold调整为更小的值！")
                    logger.info(f"    建议尝试: mask_threshold=0.5 (对于0-1范围的mask)")
                elif mask.max() < mask_threshold:
                    logger.warning(f"  ⚠️  Mask最大值({mask.max()})小于阈值({mask_threshold})，所有像素都会被视为背景！")
                    logger.info(f"    建议将mask_threshold调整为: {mask.max() // 2}")
            
            # 检查有效投影（不是inf且在图像范围内）
            not_inf = (projected_points[:, 0] != float('inf')) & (projected_points[:, 1] != float('inf'))
            valid_u = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w)
            valid_v = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            valid_proj = not_inf & valid_u & valid_v
            
            if np.any(valid_proj):
                valid_indices = np.where(valid_proj)[0]
                valid_coords = projected_points[valid_proj].astype(np.int32)
                
                # 批量查询mask值
                object_ids = mask[valid_coords[:, 1], valid_coords[:, 0]]
                
                # 统计
                total_valid_projections += len(valid_indices)
                object_mask = object_ids > mask_threshold  # 前景物体
                total_object_hits += np.sum(object_mask)
                
                # 更新投票 - 将mask值转换为物体ID
                for idx, raw_id in zip(valid_indices, object_ids):
                    if raw_id > mask_threshold:  # 前景
                        obj_id = 1  # 简化为单一物体ID
                        batch_votes[idx, obj_id] += 1
                    else:  # 背景
                        batch_votes[idx, 0] += 1
        
        # 基于投票结果分配标签（批次内）
        # 使用更宽松的阈值：只要有投票就算
        for i in range(len(batch_votes)):
            max_votes = np.max(batch_votes[i])
            if max_votes > 0:
                # 找到得票最多的类别
                label = np.argmax(batch_votes[i])
                # 映射回原始索引
                original_idx = sample_indices[batch_start + i]
                # 如果是前景类别，即使只有1票也接受
                if label > 0 and batch_votes[i, label] >= 1:
                    point_labels[original_idx] = label
                    confidence_scores[original_idx] = batch_votes[i, label] / len(processed_frames)
                elif batch_votes[i, 0] > batch_votes[i, label]:  # 背景票数更多
                    point_labels[original_idx] = 0
                    confidence_scores[original_idx] = batch_votes[i, 0] / len(processed_frames)
        
        # 定期报告进度
        if (batch_idx + 1) % 5 == 0:
            current_object_points = np.sum(point_labels > 0)
            logger.info(f"  当前已找到 {current_object_points} 个物体点")
    
    # 统计最终结果
    logger.info(f"投影统计: 总有效投影={total_valid_projections}, 命中物体={total_object_hits}")
    logger.info(f"投影命中率: {total_object_hits/total_valid_projections*100:.2f}%" if total_valid_projections > 0 else "无有效投影")
    
    # 创建目标物体的mask
    if target_object_id is None:
        # 所有非背景物体
        segment_mask = point_labels > 0
    else:
        # 特定物体
        segment_mask = point_labels == target_object_id
    
    object_counts = np.bincount(point_labels)
    logger.info(f"分割统计: 背景={object_counts[0]}, 物体1={object_counts[1] if len(object_counts) > 1 else 0}")
    logger.info(f"目标物体点数: {np.sum(segment_mask)}")
    
    # 将结果映射回原始点云索引
    full_segment_mask = np.zeros(len(points_flat), dtype=bool)
    full_segment_mask[valid_mask] = segment_mask
    
    return full_segment_mask

def project_3d_to_2d_simple(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """简化的投影函数，假设点云已经在相机坐标系中"""
    # points_3d: (N, 3) - 已经在相机坐标系中
    # intrinsic: (3, 3) - 相机内参矩阵
    
    # 过滤掉Z<=0的点
    valid_z = points_3d[:, 2] > 1e-6
    projected = np.full((len(points_3d), 2), float('inf'))  # 使用inf标记无效投影
    
    if np.any(valid_z):
        valid_points = points_3d[valid_z]
        
        # 应用内参矩阵
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # 直接使用相机坐标投影（不需要归一化，因为内参已经包含了焦距）
        # u = fx * X/Z + cx
        # v = fy * Y/Z + cy
        u = fx * valid_points[:, 0] / valid_points[:, 2] + cx
        v = fy * valid_points[:, 1] / valid_points[:, 2] + cy
        
        # 如果Y坐标大量为负，说明Y轴方向相反
        if np.median(v) < 0:
            # 翻转Y坐标并重新计算
            v = fy * (-valid_points[:, 1]) / valid_points[:, 2] + cy
            logger.debug("检测到Y轴需要翻转")
        
        projected[valid_z, 0] = u
        projected[valid_z, 1] = v
    
    return projected

def project_3d_to_2d(points_3d: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """将3D点投影到2D图像坐标"""
    # points_3d: (N, 3)
    # extrinsic: (4, 4) - 世界坐标到相机坐标变换矩阵
    # intrinsic: (3, 3) - 相机内参矩阵
    
    # 转换为齐次坐标
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # (N, 4)
    
    # 世界坐标 -> 相机坐标
    # 注意：VGGT可能使用不同的坐标系，可能需要取逆
    # 尝试使用外参矩阵的逆
    try:
        extrinsic_inv = np.linalg.inv(extrinsic)
        camera_points = (extrinsic_inv @ points_homo.T).T  # (N, 4)
    except:
        # 如果求逆失败，使用原始矩阵
        camera_points = (extrinsic @ points_homo.T).T  # (N, 4)
    
    camera_points_3d = camera_points[:, :3]  # (N, 3)
    
    # 相机坐标 -> 图像坐标
    # 投影到图像平面 (避免除零)
    valid_z = camera_points_3d[:, 2] > 1e-6
    projected = np.zeros((len(points_3d), 2))
    
    if np.any(valid_z):
        valid_camera_points = camera_points_3d[valid_z]
        
        # 投影到图像平面
        # 注意：这里直接应用内参矩阵
        projected_homo = np.hstack([valid_camera_points, np.ones((len(valid_camera_points), 1))])
        image_coords = (intrinsic @ projected_homo[:, :3].T).T  # (N, 3)
        
        # 归一化齐次坐标
        image_coords[:, :2] /= image_coords[:, 2:3]
        projected[valid_z] = image_coords[:, :2]
    
    return projected

def create_pixel_correspondence_segmentation(points_flat: np.ndarray, mask_sequence: List[np.ndarray], 
                                           valid_mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    """使用直接像素对应进行分割（当点云和mask分辨率匹配时）"""
    logger.info("使用直接像素对应分割方法")
    
    if len(original_shape) != 4:
        raise ValueError(f"需要4D点云形状，但得到: {original_shape}")
    
    S, H, W, _ = original_shape
    logger.info(f"点云形状: S={S}, H={H}, W={W}")
    
    # 初始化分割结果
    full_segment_mask = np.zeros(len(points_flat), dtype=bool)
    
    # 对每一帧进行处理
    for frame_idx in range(min(S, len(mask_sequence))):
        mask = mask_sequence[frame_idx]
        
        # 计算当前帧在扁平化数组中的索引范围
        frame_start = frame_idx * H * W
        frame_end = frame_start + H * W
        
        # 获取当前帧的有效点索引
        frame_valid_mask = valid_mask[frame_start:frame_end]
        
        # 将mask扁平化
        mask_flat = mask.flatten()
        
        # 应用mask到有效点
        # 注意：需要考虑valid_mask的影响
        valid_indices = np.where(frame_valid_mask)[0]
        for idx in valid_indices:
            pixel_idx = idx  # 在当前帧内的像素索引
            if pixel_idx < len(mask_flat):
                if mask_flat[pixel_idx] > 127:  # 使用固定阈值127
                    full_segment_mask[frame_start + idx] = True
    
    object_count = np.sum(full_segment_mask)
    total_valid = np.sum(valid_mask)
    logger.info(f"像素对应分割完成: {object_count}/{total_valid} 个点被标记为物体 "
               f"({object_count/total_valid*100:.1f}%)")
    
    return full_segment_mask

def create_simple_spatial_segmentation(points: np.ndarray, mask_sequence: List[np.ndarray], 
                                     valid_mask: np.ndarray) -> np.ndarray:
    """基于空间位置的简单分割（当缺少相机参数时）"""
    logger.info("使用简单空间分割策略")
    
    # 计算mask的空间覆盖率
    total_coverage = 0
    for mask in mask_sequence:
        coverage = np.sum(mask > 0) / mask.size
        total_coverage += coverage
    
    avg_coverage = total_coverage / len(mask_sequence)
    logger.info(f"平均mask覆盖率: {avg_coverage*100:.1f}%")
    
    # 基于覆盖率估算应该分割的点数
    n_object_points = int(len(points) * avg_coverage)
    
    # 使用Z坐标进行简单分割（假设物体在特定深度范围）
    z_coords = points[:, 2]
    z_sorted_indices = np.argsort(z_coords)
    
    # 选择中等深度的点作为物体点（避免太近或太远的点）
    start_idx = len(points) // 4
    end_idx = start_idx + n_object_points
    object_indices = z_sorted_indices[start_idx:end_idx]
    
    segment_mask = np.zeros(len(points), dtype=bool)
    segment_mask[object_indices] = True
    
    # 映射回完整点云
    full_mask = np.zeros(len(valid_mask), dtype=bool)
    full_mask[valid_mask] = segment_mask
    
    logger.info(f"简单分割选择了 {np.sum(segment_mask)} 个点")
    return full_mask

def create_segmented_pointcloud_with_spatial_preservation(raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                                      target_object_id: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    创建保持原始空间对应关系的分割点云
    
    Args:
        raw_vggt_result: VGGT原生结果
        mask_sequence: mask图像序列
        target_object_id: 目标物体ID，None表示保留所有非背景点
    
    Returns:
        original_points: 完整的原始点云 (N, 3) - 保持原始顺序和坐标
        segmented_points: 分割后的物体点云 (M, 3) - M是物体点数量
        segment_mask: 布尔掩码 (N,) - 指示哪些原始点属于目标物体
        original_shape: 原始点云的形状信息
    """
    logger.info(f"创建保持空间对应关系的分割点云，目标物体ID: {target_object_id}")
    
    # 获取点云数据
    if 'points_from_depth' in raw_vggt_result:
        points = raw_vggt_result['points_from_depth']
        logger.info("使用 points_from_depth")
    elif 'points' in raw_vggt_result:
        if isinstance(raw_vggt_result['points'], dict):
            points = raw_vggt_result['points']['point_map']
        else:
            points = raw_vggt_result['points']
        logger.info("使用 points")
    else:
        raise ValueError("No point cloud data found in raw_vggt_result")
    
    # 转换为numpy并保存原始形状
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    # 记录原始形状
    original_shape = points_np.shape
    logger.info(f"原始点云形状: {original_shape}")
    
    # 去掉batch维度但保持空间结构
    if points_np.ndim == 5 and points_np.shape[0] == 1:
        points_np = np.squeeze(points_np, axis=0)  # 现在是 (S, H, W, 3)
        original_shape = points_np.shape
    
    # 保存原始点云的完整副本（保持原始坐标和顺序）
    if points_np.ndim == 4:  # (S, H, W, 3)
        S, H, W, _ = points_np.shape
        original_points_flat = points_np.reshape(-1, 3)  # 展平但保持顺序
    elif points_np.ndim == 3:
        original_points_flat = points_np.reshape(-1, 3)
    else:
        original_points_flat = points_np
    
    logger.info(f"原始点云总数: {len(original_points_flat)}")
    
    # 获取分割mask（这个mask对应扁平化后的点云）
    segment_mask = create_mask_based_segmentation(raw_vggt_result, mask_sequence, target_object_id)
    
    # 确保mask长度匹配
    if len(segment_mask) != len(original_points_flat):
        logger.warning(f"分割mask长度({len(segment_mask)})与点云长度({len(original_points_flat)})不匹配")
        min_len = min(len(segment_mask), len(original_points_flat))
        segment_mask = segment_mask[:min_len]
        original_points_flat = original_points_flat[:min_len]
    
    # 应用分割mask获取物体点
    segmented_points = original_points_flat[segment_mask]
    
    # 统计信息
    total_points = len(original_points_flat)
    object_points = len(segmented_points)
    object_ratio = object_points / total_points * 100 if total_points > 0 else 0
    
    logger.info(f"空间保持分割结果: 总点数={total_points}, 物体点数={object_points}, 物体占比={object_ratio:.1f}%")
    
    return original_points_flat, segmented_points, segment_mask, original_shape

def create_segmented_pointcloud_new(raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                   target_object_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于mask序列创建分割点云（新的正确方法）
    现在使用空间保持方法来确保坐标对应
    
    Args:
        raw_vggt_result: VGGT原生结果
        mask_sequence: mask图像序列
        target_object_id: 目标物体ID，None表示保留所有非背景点
    
    Returns:
        vertices: 点云顶点坐标 (N, 3) - 保持原始坐标系
        colors: 点云颜色 (N, 3)
    """
    logger.info(f"使用空间保持方法创建分割点云，目标物体ID: {target_object_id}")
    
    # 使用新的空间保持方法
    original_points, segmented_points, segment_mask, original_shape = create_segmented_pointcloud_with_spatial_preservation(
        raw_vggt_result, mask_sequence, target_object_id
    )
    
    # 获取颜色数据
    colors_np = None
    if 'images' in raw_vggt_result:
        images = raw_vggt_result['images']
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        if images_np.ndim == 5 and images_np.shape[0] == 1:
            images_np = np.squeeze(images_np, axis=0)
        
        if images_np.ndim == 4:
            if images_np.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                images_np = np.transpose(images_np, (0, 2, 3, 1))
            
            # 展平颜色数据（保持与点云相同的顺序）
            colors_flat = images_np.reshape(-1, 3)
            if colors_flat.max() <= 1.0:
                colors_flat = (colors_flat * 255).astype(np.uint8)
            else:
                colors_flat = colors_flat.astype(np.uint8)
            
            # 确保颜色数据长度匹配
            if len(colors_flat) != len(original_points):
                logger.warning(f"颜色数据长度({len(colors_flat)})与点云长度({len(original_points)})不匹配")
                min_len = min(len(colors_flat), len(original_points))
                colors_flat = colors_flat[:min_len]
            
            # 应用同样的分割mask获取对应的颜色
            if len(colors_flat) >= len(segment_mask):
                colors_np = colors_flat[segment_mask]
            else:
                logger.warning("颜色数据不足，使用默认颜色")
                colors_np = None
    
    # 如果没有颜色数据或颜色数据处理失败，生成默认颜色
    if colors_np is None:
        if target_object_id is not None and target_object_id > 0:
            # 为不同物体分配不同颜色
            colors = [
                [255, 0, 0],    # 红色
                [0, 255, 0],    # 绿色  
                [0, 0, 255],    # 蓝色
                [255, 255, 0],  # 黄色
                [255, 0, 255],  # 洋红
                [0, 255, 255],  # 青色
                [255, 128, 0],  # 橙色
                [128, 0, 255],  # 紫色
            ]
            color = colors[target_object_id % len(colors)]
            colors_np = np.tile(color, (len(segmented_points), 1)).astype(np.uint8)
        else:
            colors_np = np.ones((len(segmented_points), 3), dtype=np.uint8) * 128
    
    logger.info(f"空间保持分割完成: 保留 {len(segmented_points)} 个点，坐标系保持不变")
    return segmented_points, colors_np

def create_segmented_pointcloud(raw_vggt_result: Dict, point_labels: Dict[int, List[int]], 
                               target_object_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据分割结果创建点云数据（兼容旧接口）
    
    Args:
        raw_vggt_result: VGGT原生结果
        point_labels: 点分割标签（已弃用，仅为兼容性保留）
        target_object_id: 目标物体ID，None表示保留所有分割的点
    
    Returns:
        vertices: 点云顶点坐标 (N, 3)
        colors: 点云颜色 (N, 3)
    """
    logger.warning("使用旧的create_segmented_pointcloud接口，建议改用create_segmented_pointcloud_new")
    
    # 由于旧方法问题很大，这里简单返回少量点作为兼容
    logger.info(f"创建分割点云，目标物体ID: {target_object_id}")
    
    # 获取点云数据
    if 'points_from_depth' in raw_vggt_result:
        points = raw_vggt_result['points_from_depth']
        logger.info("使用 points_from_depth")
    elif 'points' in raw_vggt_result:
        if isinstance(raw_vggt_result['points'], dict):
            points = raw_vggt_result['points']['point_map']
        else:
            points = raw_vggt_result['points']
        logger.info("使用 points")
    else:
        raise ValueError("No point cloud data found in raw_vggt_result")
    
    # 转换为numpy
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    # 去掉batch维度
    if points_np.ndim == 5 and points_np.shape[0] == 1:
        points_np = np.squeeze(points_np, axis=0)
    
    # 重塑为点云 (S*H*W, 3)
    if points_np.ndim == 4:
        points_np = points_np.reshape(-1, 3)
    elif points_np.ndim == 3:
        points_np = points_np.reshape(-1, 3)
    
    # 过滤无效点
    valid_mask = ~np.isnan(points_np).any(axis=1) & ~np.isinf(points_np).any(axis=1)
    points_np = points_np[valid_mask]
    
    # 简单选择前1000个点作为示例
    n_sample = min(1000, len(points_np))
    indices = np.random.choice(len(points_np), n_sample, replace=False)
    sample_points = points_np[indices]
    
    # 生成默认颜色
    if target_object_id is not None and target_object_id > 0:
        colors = [
            [255, 0, 0],    # 红色
            [0, 255, 0],    # 绿色  
            [0, 0, 255],    # 蓝色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 洋红
            [0, 255, 255],  # 青色
            [255, 128, 0],  # 橙色
            [128, 0, 255],  # 紫色
        ]
        color = colors[target_object_id % len(colors)]
        colors_np = np.tile(color, (len(sample_points), 1)).astype(np.uint8)
    else:
        colors_np = np.ones((len(sample_points), 3), dtype=np.uint8) * 128
    
    logger.info(f"旧方法兼容模式，保留 {len(sample_points)} 个点")
    return sample_points, colors_np

def create_segmented_pointcloud_glb_quality(raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                           target_object_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建与GLB质量一致的分割点云
    使用predictions_to_glb函数进行高质量过滤，然后应用分割mask
    
    Args:
        raw_vggt_result: VGGT原生结果
        mask_sequence: mask图像序列
        target_object_id: 目标物体ID，None表示保留所有非背景点
    
    Returns:
        vertices: 高质量分割点云顶点坐标 (N, 3)
        colors: 对应的颜色 (N, 3)
    """
    logger.info(f"创建与GLB质量一致的分割点云，目标物体ID: {target_object_id}")
    
    # 检查是否有predictions_to_glb函数
    if not VGGT_UTILS_AVAILABLE or not predictions_to_glb:
        logger.warning("predictions_to_glb不可用，回退到原方法")
        return create_segmented_pointcloud_new(raw_vggt_result, mask_sequence, target_object_id)
    
    try:
        # 先使用predictions_to_glb生成高质量的完整点云
        logger.info("第一步：使用predictions_to_glb生成高质量完整点云")
        
        # 准备与GLB相同的数据格式
        world_points_from_depth = raw_vggt_result.get('points_from_depth')
        
        # depth_conf: 兼容多种返回格式
        depth_conf = None
        if 'depth' in raw_vggt_result:
            if isinstance(raw_vggt_result['depth'], dict):
                depth_conf = raw_vggt_result['depth'].get('confidence')
            else:
                depth_conf = raw_vggt_result.get('depth_conf')
        
        # images
        images_tensor = raw_vggt_result.get('images')
        
        # extrinsic
        extrinsic_mat = None
        if 'cameras' in raw_vggt_result:
            if isinstance(raw_vggt_result['cameras'], dict):
                extrinsic_mat = raw_vggt_result['cameras'].get('extrinsic')
            else:
                extrinsic_mat = raw_vggt_result.get('extrinsic')
        
        # 维度处理
        if isinstance(extrinsic_mat, torch.Tensor):
            if extrinsic_mat.ndim == 4 and extrinsic_mat.shape[0] == 1:
                extrinsic_mat = extrinsic_mat.squeeze(0)
        elif isinstance(extrinsic_mat, np.ndarray):
            if extrinsic_mat.ndim == 4 and extrinsic_mat.shape[0] == 1:
                extrinsic_mat = np.squeeze(extrinsic_mat, axis=0)
        
        # 处理其他数据
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
        
        # 回退到原始点云
        if predictions_formatted['world_points_from_depth'] is None:
            if 'points' in raw_vggt_result:
                logger.info("Using point_map as fallback for world_points")
                if isinstance(raw_vggt_result['points'], dict):
                    predictions_formatted['world_points'] = raw_vggt_result['points']['point_map']
                    predictions_formatted['world_points_conf'] = raw_vggt_result['points']['confidence']
                else:
                    predictions_formatted['world_points'] = raw_vggt_result['points']
        
        # 转换为numpy
        for key, value in predictions_formatted.items():
            if value is not None and isinstance(value, torch.Tensor):
                predictions_formatted[key] = value.cpu().numpy()
        
        # 使用predictions_to_glb生成高质量3D场景
        scene_3d = predictions_to_glb(
            predictions_formatted,
            conf_thres=90.0,  # 使用高置信度阈值保证质量
            filter_by_frames="all",
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=False,  # 只要点云
            mask_sky=False,
            target_dir=None,
            prediction_mode="Depthmap and Camera Branch"
        )
        
        # 从3D场景提取高质量点云
        vertices_list = []
        colors_list = []
        
        for geometry in scene_3d.geometry.values():
            if hasattr(geometry, 'vertices') and hasattr(geometry, 'visual'):
                vertices = np.array(geometry.vertices)
                vertices_list.append(vertices)
                
                if hasattr(geometry.visual, 'vertex_colors'):
                    colors = np.array(geometry.visual.vertex_colors)[:, :3]
                    colors_list.append(colors)
                elif hasattr(geometry.visual, 'face_colors'):
                    face_colors = np.array(geometry.visual.face_colors)[:, :3]
                    vertex_colors = np.tile(face_colors[0] if len(face_colors) > 0 else [128, 128, 128], 
                                          (len(vertices), 1))
                    colors_list.append(vertex_colors)
                else:
                    default_colors = np.ones((len(vertices), 3), dtype=np.uint8) * 128
                    colors_list.append(default_colors)
        
        if not vertices_list:
            logger.warning("没有从3D场景提取到点云，回退到原方法")
            return create_segmented_pointcloud_new(raw_vggt_result, mask_sequence, target_object_id)
        
        # 合并高质量点云
        high_quality_vertices = np.vstack(vertices_list)
        high_quality_colors = np.vstack(colors_list)
        
        # 确保颜色格式正确
        if high_quality_colors.max() <= 1.0:
            high_quality_colors = (high_quality_colors * 255).astype(np.uint8)
        else:
            high_quality_colors = high_quality_colors.astype(np.uint8)
        
        logger.info(f"第一步完成：获得 {len(high_quality_vertices)} 个高质量点")
        
        # 第二步：应用mask分割到高质量点云
        logger.info("第二步：对高质量点云应用mask分割")
        
        # 需要将高质量点云映射回原始图像空间进行分割
        # 这里使用投影方法或空间对应关系
        
        # 获取分割mask（使用原始方法）
        segment_mask_full = create_mask_based_segmentation(raw_vggt_result, mask_sequence, target_object_id)
        
        # 获取原始点云用于空间对应
        if 'points_from_depth' in raw_vggt_result:
            original_points_raw = raw_vggt_result['points_from_depth']
        elif 'points' in raw_vggt_result:
            if isinstance(raw_vggt_result['points'], dict):
                original_points_raw = raw_vggt_result['points']['point_map']
            else:
                original_points_raw = raw_vggt_result['points']
        else:
            logger.warning("无法获取原始点云进行对应，回退到原方法")
            return create_segmented_pointcloud_new(raw_vggt_result, mask_sequence, target_object_id)
        
        # 转换原始点云
        if isinstance(original_points_raw, torch.Tensor):
            original_points_np = original_points_raw.cpu().numpy()
        else:
            original_points_np = original_points_raw
        
        if original_points_np.ndim == 5 and original_points_np.shape[0] == 1:
            original_points_np = np.squeeze(original_points_np, axis=0)
        
        if original_points_np.ndim == 4:
            original_points_flat = original_points_np.reshape(-1, 3)
        else:
            original_points_flat = original_points_np.reshape(-1, 3)
        
        # 过滤无效原始点
        valid_original_mask = ~np.isnan(original_points_flat).any(axis=1) & ~np.isinf(original_points_flat).any(axis=1)
        valid_original_points = original_points_flat[valid_original_mask]
        
        # 调整segment_mask长度匹配
        if len(segment_mask_full) != len(original_points_flat):
            min_len = min(len(segment_mask_full), len(original_points_flat))
            segment_mask_full = segment_mask_full[:min_len]
            valid_original_mask = valid_original_mask[:min_len]
        
        # 获取分割后的原始点
        segment_mask_valid = segment_mask_full[valid_original_mask]
        segmented_original_points = valid_original_points[segment_mask_valid]
        
        if len(segmented_original_points) == 0:
            logger.warning("分割后没有找到物体点，返回空结果")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        logger.info(f"原始分割找到 {len(segmented_original_points)} 个物体点")
        
        # 第三步：在高质量点云中找到最接近的点
        logger.info("第三步：在高质量点云中匹配分割点")
        
        # 使用KDTree进行快速最近邻搜索
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(high_quality_vertices)
            distances, indices = tree.query(segmented_original_points, k=1)
            
            # 过滤距离过远的点（可能是噪声）
            distance_threshold = np.percentile(distances, 95)  # 使用95百分位作为阈值
            valid_matches = distances <= distance_threshold
            
            matched_indices = indices[valid_matches]
            final_vertices = high_quality_vertices[matched_indices]
            final_colors = high_quality_colors[matched_indices]
            
            logger.info(f"第三步完成：匹配到 {len(final_vertices)} 个高质量分割点")
            logger.info(f"平均匹配距离: {np.mean(distances[valid_matches]):.4f}")
            
            return final_vertices, final_colors
            
        except ImportError:
            logger.warning("scipy不可用，使用简化匹配方法")
            # 简化方法：直接使用空间位置筛选
            
            # 计算分割区域的边界
            min_coords = np.min(segmented_original_points, axis=0)
            max_coords = np.max(segmented_original_points, axis=0)
            
            # 在高质量点云中找到在边界内的点
            mask_x = (high_quality_vertices[:, 0] >= min_coords[0]) & (high_quality_vertices[:, 0] <= max_coords[0])
            mask_y = (high_quality_vertices[:, 1] >= min_coords[1]) & (high_quality_vertices[:, 1] <= max_coords[1])
            mask_z = (high_quality_vertices[:, 2] >= min_coords[2]) & (high_quality_vertices[:, 2] <= max_coords[2])
            
            region_mask = mask_x & mask_y & mask_z
            
            final_vertices = high_quality_vertices[region_mask]
            final_colors = high_quality_colors[region_mask]
            
            logger.info(f"简化匹配完成：找到 {len(final_vertices)} 个区域内高质量点")
            
            return final_vertices, final_colors
        
    except Exception as e:
        logger.error(f"GLB质量分割失败: {e}")
        import traceback
        traceback.print_exc()
        logger.info("回退到原分割方法")
        return create_segmented_pointcloud_new(raw_vggt_result, mask_sequence, target_object_id)

# -----------------------------------------------------------------------------
# 主要节点实现
# -----------------------------------------------------------------------------

class VGGTMaskProcessorNode:
    """VGGT Mask处理节点 - 使用tracks数据和mask序列生成分割的3D模型"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "native_full_result": ("RAW_VGGT_RESULT", {
                    "tooltip": "来自VGGTNativeFullOutputNode的完整原生结果"
                }),
                "tracks_json": ("STRING", {
                    "tooltip": "来自VGGTNativeFullOutputNode的tracks JSON数据"
                }),
                "mask_sequence": ("MASK", {
                    "tooltip": "mask图像序列，每张图像包含分割标识（不同像素值代表不同物体）"
                }),
            },
            "optional": {
                "target_object_id": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "目标物体ID（0=导出所有分割物体，>0=导出指定物体）"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "点追踪置信度阈值，低于此值的点会被忽略"
                }),
                "export_format": (["PLY", "GLB", "BOTH"], {
                    "default": "PLY",
                    "tooltip": "导出格式：PLY(点云)、GLB(网格)或两者都导出"
                }),
                "use_full_pointcloud": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用完整点云分割（推荐）。如果关闭，仅使用tracks点进行分割"
                }),
                "preserve_spatial_correspondence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否保持与原始点云的空间对应关系（强烈推荐），确保分割后的点云坐标与原点云完全一致"
                }),
                "export_full_pointcloud_with_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否同时导出包含分割mask信息的完整点云文件"
                }),
                "use_glb_quality_filtering": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用与GLB模型相同的高质量过滤算法（强烈推荐），确保分割点云质量与GLB输出一致"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",            # 分割统计信息JSON
        "STRING",            # PLY文件路径
        "STRING",            # GLB文件路径  
        "STRING",            # 分割报告
        "STRING",            # 完整点云PLY路径（带mask信息）
    )
    RETURN_NAMES = (
        "segmentation_stats",
        "segmented_ply_path",
        "segmented_glb_path", 
        "segmentation_report",
        "full_pointcloud_with_mask_path",
    )
    OUTPUT_TOOLTIPS = [
        "分割统计信息（JSON格式，包含每个物体的点数统计）",
        "分割后的PLY点云文件路径",
        "分割后的GLB 3D模型文件路径",
        "详细的分割处理报告（JSON格式）",
        "包含分割mask信息的完整原始点云PLY文件路径"
    ]
    OUTPUT_NODE = True
    FUNCTION = "process_mask_segmentation"
    CATEGORY = "💃VVL/VGGT Mask"

    def process_mask_segmentation(self, native_full_result: Dict, tracks_json: str, mask_sequence,
                                target_object_id: int = 0,
                                confidence_threshold: float = 0.5,
                                export_format: str = "PLY",
                                use_full_pointcloud: bool = True,
                                preserve_spatial_correspondence: bool = True,
                                export_full_pointcloud_with_mask: bool = False,
                                use_glb_quality_filtering: bool = True):
        """处理mask分割"""
        logger.info("开始VGGT Mask分割处理")
        logger.info(f"接收到tracks_json类型: {type(tracks_json)}, 长度: {len(tracks_json) if tracks_json else 0}")
        
        tracks_data = None
        
        try:
            # 首先尝试从native_full_result直接获取tracks数据
            if 'tracks' in native_full_result and native_full_result['tracks']:
                logger.info("直接从native_full_result获取tracks数据")
                raw_tracks = native_full_result['tracks']
                logger.info(f"从native_full_result获取的tracks keys: {list(raw_tracks.keys())}")
                
                # 检查并转换tensor格式的数据
                converted_tracks = {}
                for key, value in raw_tracks.items():
                    if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                        # PyTorch tensor
                        converted_tracks[key] = value.cpu().numpy()
                        logger.info(f"转换tensor {key}, shape: {value.shape}")
                    elif isinstance(value, np.ndarray):
                        converted_tracks[key] = value
                        logger.info(f"保持numpy {key}, shape: {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        # 列表或元组，可能包含tensor
                        try:
                            # 尝试转换为numpy数组（如果元素是tensor）
                            if len(value) > 0 and hasattr(value[0], 'cpu'):
                                # 列表中包含tensor
                                converted_list = [item.cpu().numpy() if hasattr(item, 'cpu') else item for item in value]
                                converted_tracks[key] = np.array(converted_list)
                                logger.info(f"转换tensor列表 {key}, 元素数量: {len(value)}")
                            else:
                                # 普通列表
                                converted_tracks[key] = np.array(value) if isinstance(value, list) else value
                                logger.info(f"转换列表 {key}, 元素数量: {len(value)}")
                        except Exception as e:
                            logger.warning(f"转换列表 {key} 失败: {e}, 保持原格式")
                            converted_tracks[key] = value
                    else:
                        # 其他类型，可能是隐藏的tensor
                        try:
                            # 尝试作为tensor处理
                            if hasattr(value, 'cpu'):
                                converted_tracks[key] = value.cpu().numpy()
                                logger.info(f"强制转换tensor {key}, type: {type(value)}")
                            else:
                                converted_tracks[key] = value
                                logger.info(f"保持原格式 {key}, type: {type(value)}")
                        except Exception as e:
                            logger.warning(f"处理 {key} 时出错: {e}")
                            converted_tracks[key] = value
                
                tracks_data = converted_tracks
                logger.info("成功从native_full_result获取tracks数据")
            
            # 如果直接获取失败且有tracks_json，尝试JSON解析
            elif tracks_json and tracks_json.strip():
                logger.info("备用方案：从tracks_json解析数据")
                logger.info(f"tracks_json前200字符: {tracks_json[:200]}")
                parsed_data = json.loads(tracks_json)
                logger.info(f"成功解析JSON，顶级keys: {list(parsed_data.keys())}")
                tracks_data = parsed_data.get('tracks_data', {})
                if tracks_data:
                    logger.info("成功从tracks_json获取tracks数据")
            
            # 检查是否成功获取了tracks数据
            if not tracks_data:
                raise ValueError("无法从native_full_result或tracks_json获取有效的tracks数据")
            
            # 处理mask序列 - 改进版本
            mask_list = []
            logger.info(f"开始处理mask序列，输入类型: {type(mask_sequence)}")
            
            if isinstance(mask_sequence, torch.Tensor):
                mask_np = mask_sequence.cpu().numpy()
                logger.info(f"Tensor mask形状: {mask_np.shape}, dtype: {mask_np.dtype}, 值范围: [{mask_np.min()}, {mask_np.max()}]")
                
                for i in range(mask_np.shape[0]):
                    mask_img = mask_np[i]
                    logger.info(f"  处理mask {i}: shape={mask_img.shape}, dtype={mask_img.dtype}, range=[{mask_img.min()}, {mask_img.max()}]")
                    
                    # 如果是RGB图像，转换为单通道
                    if mask_img.ndim == 3:
                        # 检查是否所有通道都相同（灰度图的RGB表示）
                        if mask_img.shape[2] >= 3:
                            r, g, b = mask_img[:,:,0], mask_img[:,:,1], mask_img[:,:,2]
                            if np.allclose(r, g) and np.allclose(g, b):
                                mask_img = r  # 使用第一个通道
                                logger.info(f"    检测到灰度RGB，转换为单通道")
                            else:
                                # RGB图像，可能需要转换为灰度
                                mask_img = np.mean(mask_img, axis=2)
                                logger.info(f"    转换RGB为灰度")
                        else:
                            mask_img = mask_img[:,:,0]
                    
                    # 处理值范围
                    if mask_img.max() <= 1.0 and mask_img.dtype in [np.float32, np.float64]:
                        # 0-1范围的浮点mask，转换为0-255
                        if np.all(np.isin(mask_img, [0.0, 1.0])):
                            # 严格的0-1二值mask
                            mask_img = (mask_img * 255).astype(np.uint8)
                            logger.info(f"    转换0-1二值mask为0-255")
                        else:
                            # 可能是0-1范围的多值mask
                            mask_img = (mask_img * 255).astype(np.uint8)
                            logger.info(f"    转换0-1浮点mask为0-255")
                    else:
                        # 已经是整数范围，确保类型正确
                        mask_img = mask_img.astype(np.uint8)
                        logger.info(f"    保持整数mask，转换为uint8")
                    
                    logger.info(f"    最终mask {i}: shape={mask_img.shape}, dtype={mask_img.dtype}, range=[{mask_img.min()}, {mask_img.max()}], unique={np.unique(mask_img)}")
                    mask_list.append(mask_img)
                    
            else:
                # 非tensor输入，直接使用
                mask_list = list(mask_sequence)
                logger.info(f"使用非tensor mask序列，长度: {len(mask_list)}")
                if len(mask_list) > 0:
                    first_mask = mask_list[0]
                    logger.info(f"第一个mask: type={type(first_mask)}, shape={getattr(first_mask, 'shape', 'N/A')}")
                    if hasattr(first_mask, 'dtype'):
                        logger.info(f"  dtype={first_mask.dtype}, range=[{first_mask.min()}, {first_mask.max()}]")
            
            logger.info(f"Mask序列处理完成，共 {len(mask_list)} 张图像")
            
            # 🔍 强制检查处理后的mask_list
            logger.info(f"🔍 强制检查处理后的mask_list:")
            for i in range(min(3, len(mask_list))):
                mask = mask_list[i]
                logger.info(f"🔍 mask_list[{i}]: type={type(mask)}, shape={getattr(mask, 'shape', 'N/A')}")
                if hasattr(mask, 'dtype'):
                    logger.info(f"🔍   dtype={mask.dtype}, min={mask.min()}, max={mask.max()}")
                if hasattr(mask, '__len__') and len(mask.shape) == 2:
                    unique_vals = np.unique(mask)
                    white_pixels = np.sum(mask == 255)
                    black_pixels = np.sum(mask == 0)
                    logger.info(f"🔍   unique_values={unique_vals}, white_pixels={white_pixels}, black_pixels={black_pixels}")
            
            # 调试tracks_data的详细信息
            logger.info(f"传递给process_tracks_with_masks的tracks_data:")
            for key, value in tracks_data.items():
                logger.info(f"  {key}: type={type(value)}, hasattr_cpu={hasattr(value, 'cpu')}")
                if hasattr(value, 'shape'):
                    logger.info(f"    shape: {value.shape}")
                elif hasattr(value, '__len__'):
                    logger.info(f"    length: {len(value)}")
            
            # 使用tracks数据和mask进行分割
            point_labels = process_tracks_with_masks(
                tracks_data, mask_list, confidence_threshold
            )
            
            # 统计分割结果
            stats = self._generate_segmentation_stats(point_labels)
            stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
            
            # 创建输出目录
            if FOLDER_PATHS_AVAILABLE:
                output_dir = os.path.join(folder_paths.get_output_directory(), "segmented_models")
            else:
                output_dir = os.path.join("output", "segmented_models")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # 导出文件
            ply_path = ""
            glb_path = ""
            full_pointcloud_path = ""
            
            if export_format in ["PLY", "BOTH"]:
                if use_full_pointcloud:
                    # 使用完整点云分割（推荐）
                    logger.info("使用完整点云分割方法")
                    if preserve_spatial_correspondence:
                        # 新的空间保持方法
                        logger.info("启用空间对应关系保持")
                        if use_glb_quality_filtering:
                            logger.info("使用GLB质量过滤")
                            ply_path = self._export_segmented_ply_glb_quality(
                                native_full_result, mask_list, target_object_id, 
                                output_dir, timestamp
                            )
                        else:
                            ply_path = self._export_segmented_ply_with_spatial_preservation(
                                native_full_result, mask_list, target_object_id, 
                                output_dir, timestamp
                            )
                    else:
                        # 原来的方法
                        ply_path = self._export_segmented_ply_new(
                            native_full_result, mask_list, target_object_id, 
                            output_dir, timestamp
                        )
                else:
                    # 仅使用tracks点分割（可能无效）
                    logger.info("使用tracks点分割方法（仅使用查询点）")
                    if stats.get("unique_objects", 0) == 0:
                        logger.warning("tracks点分割未找到物体，建议启用use_full_pointcloud选项")
                    ply_path = self._export_segmented_ply(
                        native_full_result, point_labels, target_object_id,
                        output_dir, timestamp
                    )
            
            # 导出包含mask信息的完整点云
            if export_full_pointcloud_with_mask:
                logger.info("导出包含分割mask信息的完整点云")
                full_pointcloud_path = self._export_full_pointcloud_with_mask(
                    native_full_result, mask_list, target_object_id,
                    output_dir, timestamp
                )
            
            if export_format in ["GLB", "BOTH"]:
                glb_path = self._export_segmented_glb(
                    native_full_result, point_labels, target_object_id,
                    output_dir, timestamp
                )
            
            # 生成报告
            report = self._generate_segmentation_report(
                stats, ply_path, glb_path, target_object_id, confidence_threshold,
                full_pointcloud_path, preserve_spatial_correspondence, use_glb_quality_filtering
            )
            report_json = json.dumps(report, ensure_ascii=False, indent=2)
            
            logger.info("VGGT Mask分割处理完成")
            return (stats_json, ply_path, glb_path, report_json, full_pointcloud_path)
            
        except Exception as e:
            logger.error(f"Mask分割处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"分割处理失败: {str(e)}"
            error_json = json.dumps({"error": str(e)})
            return (error_json, "", "", error_msg, "")
    
    def _generate_segmentation_stats(self, point_labels: Dict[int, List[int]]) -> Dict:
        """生成分割统计信息"""
        object_counts = defaultdict(int)
        total_points = len(point_labels)
        
        for labels in point_labels.values():
            for label in labels:
                # 确保key是标准int类型
                label_int = int(label) if hasattr(label, 'item') else int(label)
                object_counts[label_int] += 1
        
        return {
            "total_points": int(total_points),
            "object_point_counts": {str(k): int(v) for k, v in object_counts.items()},  # key转为字符串
            "unique_objects": len([obj_id for obj_id in object_counts.keys() if obj_id > 0]),
            "background_points": int(object_counts.get(0, 0))
        }
    
    def _export_segmented_ply_new(self, raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出分割后的PLY文件（新方法）"""
        try:
            # 首先检查mask序列的有效性
            logger.info(f"检查mask序列: {len(mask_sequence)} 张图像")
            for i in range(min(3, len(mask_sequence))):
                mask = mask_sequence[i]
                unique_vals = np.unique(mask)
                object_ratio = np.sum(mask > 0) / mask.size * 100
                logger.info(f"  Mask {i}: shape={mask.shape}, unique_values={unique_vals}, object_ratio={object_ratio:.1f}%")
            
            vertices, colors = create_segmented_pointcloud_new(
                raw_vggt_result, mask_sequence, 
                target_object_id if target_object_id > 0 else None
            )
            
            if len(vertices) == 0:
                logger.warning("分割后没有点，无法导出PLY文件")
                return ""
            
            if target_object_id > 0:
                filename = f"vggt_segmented_obj{target_object_id}_{timestamp}.ply"
            else:
                filename = f"vggt_segmented_all_{timestamp}.ply"
            
            ply_path = os.path.join(output_dir, filename)
            self._write_ply_file(ply_path, vertices, colors)
            
            logger.info(f"新方法分割PLY已保存: {ply_path} ({len(vertices)} 个点)")
            return ply_path
            
        except Exception as e:
            logger.error(f"导出分割PLY失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _export_segmented_ply(self, raw_vggt_result: Dict, point_labels: Dict[int, List[int]], 
                            target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出分割后的PLY文件（旧方法，兼容性保留）"""
        try:
            vertices, colors = create_segmented_pointcloud(
                raw_vggt_result, point_labels, 
                target_object_id if target_object_id > 0 else None
            )
            
            if target_object_id > 0:
                filename = f"vggt_segmented_obj{target_object_id}_{timestamp}.ply"
            else:
                filename = f"vggt_segmented_all_{timestamp}.ply"
            
            ply_path = os.path.join(output_dir, filename)
            self._write_ply_file(ply_path, vertices, colors)
            
            logger.info(f"分割PLY已保存: {ply_path} ({len(vertices)} 个点)")
            return ply_path
            
        except Exception as e:
            logger.error(f"导出分割PLY失败: {e}")
            return ""
    
    def _export_segmented_glb(self, raw_vggt_result: Dict, point_labels: Dict[int, List[int]], 
                            target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出分割后的GLB文件"""
        if not VGGT_UTILS_AVAILABLE or not predictions_to_glb:
            logger.warning("GLB导出功能不可用")
            return ""
        
        try:
            # 由于分割后的点数量很少，GLB导出容易失败
            # 暂时只支持PLY导出，避免复杂的3D网格生成
            logger.info("分割后点数量较少，暂时跳过GLB导出，请使用PLY文件")
            return ""
            
        except Exception as e:
            logger.error(f"导出分割GLB失败: {e}")
            return ""
    
    def _create_segmented_predictions(self, raw_vggt_result: Dict, point_labels: Dict[int, List[int]], 
                                    target_object_id: int) -> Dict:
        """创建分割版本的预测数据"""
        # 创建分割mask
        vertices, colors = create_segmented_pointcloud(
            raw_vggt_result, point_labels, 
            target_object_id if target_object_id > 0 else None
        )
        
        # 获取原始相机参数
        cameras = raw_vggt_result.get('cameras', {})
        
        # 处理相机外参 - 使用与原生节点相同的简单格式
        extrinsic = cameras.get('extrinsic', None)
        if extrinsic is not None:
            if isinstance(extrinsic, torch.Tensor):
                extrinsic = extrinsic.cpu().numpy()
        else:
            # 使用单位矩阵作为默认值 (S, 3, 4) 格式
            extrinsic = np.eye(4)[None, :3, :]
        
        # 确保形状正确
        if extrinsic.ndim == 3:
            if extrinsic.shape[-2] == 4 and extrinsic.shape[-1] == 4:
                # (S, 4, 4) -> (S, 3, 4)
                extrinsic = extrinsic[:, :3, :]
        
        # 重塑点云数据为简单的3D格式 - 参考原生节点的格式
        n_points = len(vertices)
        h = int(np.sqrt(n_points))
        w = n_points // h
        if h * w != n_points:
            # 无法完美重塑为方形，使用接近方形的布局
            h = int(np.ceil(np.sqrt(n_points)))
            # 填充到方形
            pad_size = h * h - n_points
            if pad_size > 0:
                # 用最后一个点填充
                last_vertex = vertices[-1:].repeat(pad_size, axis=0)
                last_color = colors[-1:].repeat(pad_size, axis=0)
                vertices = np.vstack([vertices, last_vertex])
                colors = np.vstack([colors, last_color])
            w = h
        
        # 重塑为 (1, H, W, 3) 格式（与原生节点兼容）
        world_points = vertices.reshape(1, h, w, 3)
        images_color = colors.reshape(1, h, w, 3) / 255.0  # 归一化到[0,1]
        
        # 构建与原生节点兼容的格式
        result = {
            'world_points_from_depth': world_points,
            'depth_conf': np.ones((1, h, w)),  # 假设所有点置信度为1
            'images': images_color,
            'extrinsic': extrinsic,
        }
        
        return result
    
    def _write_ply_file(self, filepath: str, vertices: np.ndarray, colors: np.ndarray = None):
        """写入PLY格式文件"""
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
    
    def _generate_segmentation_report(self, stats: Dict, ply_path: str, glb_path: str,
                                    target_object_id: int, confidence_threshold: float,
                                    full_pointcloud_path: str, preserve_spatial_correspondence: bool,
                                    use_glb_quality_filtering: bool) -> Dict:
        """生成详细的分割报告"""
        return {
            "processing_parameters": {
                "target_object_id": target_object_id,
                "confidence_threshold": confidence_threshold,
                "preserve_spatial_correspondence": preserve_spatial_correspondence,
                "use_glb_quality_filtering": use_glb_quality_filtering
            },
            "segmentation_statistics": stats,
            "output_files": {
                "ply_file": ply_path,
                "glb_file": glb_path,
                "full_pointcloud_with_mask_path": full_pointcloud_path
            },
            "processing_time": time.time(),
            "status": "success" if (ply_path or glb_path or full_pointcloud_path) else "failed"
        }

    def _export_segmented_ply_with_spatial_preservation(self, raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                                      target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出保持空间对应关系的分割PLY文件"""
        try:
            # 使用空间保持方法获取分割结果
            original_points, segmented_points, segment_mask, original_shape = create_segmented_pointcloud_with_spatial_preservation(
                raw_vggt_result, mask_sequence, target_object_id if target_object_id > 0 else None
            )
            
            if len(segmented_points) == 0:
                logger.warning("分割后没有点，无法导出PLY文件")
                return ""
            
            # 获取对应的颜色信息
            colors = self._get_colors_for_segmented_points(raw_vggt_result, segment_mask, target_object_id)
            
            # 生成文件名
            if target_object_id > 0:
                filename = f"vggt_segmented_spatial_preserved_obj{target_object_id}_{timestamp}.ply"
            else:
                filename = f"vggt_segmented_spatial_preserved_all_{timestamp}.ply"
            
            ply_path = os.path.join(output_dir, filename)
            self._write_ply_file(ply_path, segmented_points, colors)
            
            # 计算与原始点云的对应关系统计
            total_original = len(original_points)
            segmented_count = len(segmented_points)
            preservation_ratio = segmented_count / total_original * 100 if total_original > 0 else 0
            
            logger.info(f"空间保持分割PLY已保存: {ply_path}")
            logger.info(f"  原始点云: {total_original} 个点")
            logger.info(f"  分割点云: {segmented_count} 个点 ({preservation_ratio:.1f}%)")
            logger.info(f"  坐标系: 与原始点云完全一致")
            logger.info(f"  原始形状: {original_shape}")
            
            return ply_path
            
        except Exception as e:
            logger.error(f"导出空间保持分割PLY失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _export_full_pointcloud_with_mask(self, raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                        target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出包含分割mask信息的完整点云文件"""
        try:
            # 获取完整的空间对应信息
            original_points, segmented_points, segment_mask, original_shape = create_segmented_pointcloud_with_spatial_preservation(
                raw_vggt_result, mask_sequence, target_object_id if target_object_id > 0 else None
            )
            
            if len(original_points) == 0:
                logger.warning("没有原始点云数据，无法导出完整点云文件")
                return ""
            
            # 获取完整点云的颜色信息
            colors = self._get_full_pointcloud_colors(raw_vggt_result, len(original_points))
            
            # 为每个点添加分割信息（通过颜色标记）
            # 物体点使用特殊颜色，背景点保持原色或使用灰色
            marked_colors = colors.copy() if colors is not None else np.ones((len(original_points), 3), dtype=np.uint8) * 128
            
            # 为分割出的物体点着色
            if target_object_id > 0:
                # 特定物体用特殊颜色标记
                object_color = self._get_object_color(target_object_id)
                marked_colors[segment_mask] = object_color
            else:
                # 所有非背景物体用红色标记
                marked_colors[segment_mask] = [255, 0, 0]
            
            # 生成文件名
            if target_object_id > 0:
                filename = f"vggt_full_pointcloud_with_mask_obj{target_object_id}_{timestamp}.ply"
            else:
                filename = f"vggt_full_pointcloud_with_mask_all_{timestamp}.ply"
            
            ply_path = os.path.join(output_dir, filename)
            
            # 写入PLY文件，包含额外的mask信息
            self._write_ply_file_with_mask_info(ply_path, original_points, marked_colors, segment_mask, original_shape)
            
            object_count = np.sum(segment_mask)
            total_count = len(original_points)
            object_ratio = object_count / total_count * 100 if total_count > 0 else 0
            
            logger.info(f"完整点云（带mask）已保存: {ply_path}")
            logger.info(f"  总点数: {total_count}")
            logger.info(f"  物体点数: {object_count} ({object_ratio:.1f}%)")
            logger.info(f"  背景点数: {total_count - object_count} ({100-object_ratio:.1f}%)")
            logger.info(f"  原始形状: {original_shape}")
            
            return ply_path
            
        except Exception as e:
            logger.error(f"导出完整点云（带mask）失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _get_colors_for_segmented_points(self, raw_vggt_result: Dict, segment_mask: np.ndarray, target_object_id: int) -> np.ndarray:
        """获取分割点的颜色信息"""
        try:
            if 'images' in raw_vggt_result:
                images = raw_vggt_result['images']
                if isinstance(images, torch.Tensor):
                    images_np = images.cpu().numpy()
                else:
                    images_np = images
                
                if images_np.ndim == 5 and images_np.shape[0] == 1:
                    images_np = np.squeeze(images_np, axis=0)
                
                if images_np.ndim == 4:
                    if images_np.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                        images_np = np.transpose(images_np, (0, 2, 3, 1))
                    
                    # 展平颜色数据
                    colors_flat = images_np.reshape(-1, 3)
                    if colors_flat.max() <= 1.0:
                        colors_flat = (colors_flat * 255).astype(np.uint8)
                    else:
                        colors_flat = colors_flat.astype(np.uint8)
                    
                    # 应用分割mask
                    if len(colors_flat) >= len(segment_mask):
                        return colors_flat[segment_mask]
        except Exception as e:
            logger.warning(f"获取分割点颜色失败: {e}")
        
        # 使用默认颜色
        segmented_count = np.sum(segment_mask)
        if target_object_id > 0:
            color = self._get_object_color(target_object_id)
            return np.tile(color, (segmented_count, 1)).astype(np.uint8)
        else:
            return np.ones((segmented_count, 3), dtype=np.uint8) * 128
    
    def _get_full_pointcloud_colors(self, raw_vggt_result: Dict, point_count: int) -> np.ndarray:
        """获取完整点云的颜色信息"""
        try:
            if 'images' in raw_vggt_result:
                images = raw_vggt_result['images']
                if isinstance(images, torch.Tensor):
                    images_np = images.cpu().numpy()
                else:
                    images_np = images
                
                if images_np.ndim == 5 and images_np.shape[0] == 1:
                    images_np = np.squeeze(images_np, axis=0)
                
                if images_np.ndim == 4:
                    if images_np.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
                        images_np = np.transpose(images_np, (0, 2, 3, 1))
                    
                    # 展平颜色数据
                    colors_flat = images_np.reshape(-1, 3)
                    if colors_flat.max() <= 1.0:
                        colors_flat = (colors_flat * 255).astype(np.uint8)
                    else:
                        colors_flat = colors_flat.astype(np.uint8)
                    
                    # 确保长度匹配
                    if len(colors_flat) >= point_count:
                        return colors_flat[:point_count]
        except Exception as e:
            logger.warning(f"获取完整点云颜色失败: {e}")
        
        # 使用默认灰色
        return np.ones((point_count, 3), dtype=np.uint8) * 128
    
    def _get_object_color(self, object_id: int) -> np.ndarray:
        """为特定物体ID获取颜色"""
        colors = [
            [255, 0, 0],    # 红色
            [0, 255, 0],    # 绿色  
            [0, 0, 255],    # 蓝色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 洋红
            [0, 255, 255],  # 青色
            [255, 128, 0],  # 橙色
            [128, 0, 255],  # 紫色
        ]
        return np.array(colors[object_id % len(colors)], dtype=np.uint8)
    
    def _write_ply_file_with_mask_info(self, filepath: str, vertices: np.ndarray, colors: np.ndarray, 
                                     segment_mask: np.ndarray, original_shape: tuple):
        """写入包含mask信息的PLY文件"""
        try:
            with open(filepath, 'w') as f:
                # PLY头部
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"comment VGGT segmented pointcloud with spatial preservation\n")
                f.write(f"comment Original shape: {original_shape}\n")
                f.write(f"comment Object points: {np.sum(segment_mask)}\n")
                f.write(f"comment Background points: {len(segment_mask) - np.sum(segment_mask)}\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("property uchar is_object\n")  # 额外属性：是否为物体点
                f.write("end_header\n")
                
                # 写入顶点数据
                for i in range(len(vertices)):
                    x, y, z = vertices[i]
                    r, g, b = colors[i]
                    is_obj = 255 if segment_mask[i] else 0  # 物体点为255，背景点为0
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {is_obj}\n")
                        
        except Exception as e:
            logger.error(f"写入PLY文件（带mask信息）失败: {e}")
            raise

    def _export_segmented_ply_glb_quality(self, raw_vggt_result: Dict, mask_sequence: List[np.ndarray], 
                                         target_object_id: int, output_dir: str, timestamp: int) -> str:
        """导出GLB质量的分割PLY文件"""
        try:
            # 使用GLB质量分割方法
            vertices, colors = create_segmented_pointcloud_glb_quality(
                raw_vggt_result, mask_sequence, target_object_id if target_object_id > 0 else None
            )
            
            if len(vertices) == 0:
                logger.warning("GLB质量分割后没有点，回退到标准方法")
                return self._export_segmented_ply_with_spatial_preservation(
                    raw_vggt_result, mask_sequence, target_object_id, output_dir, timestamp
                )
            
            # 生成文件名
            if target_object_id > 0:
                filename = f"vggt_segmented_glb_quality_obj{target_object_id}_{timestamp}.ply"
            else:
                filename = f"vggt_segmented_glb_quality_all_{timestamp}.ply"
            
            ply_path = os.path.join(output_dir, filename)
            self._write_ply_file(ply_path, vertices, colors)
            
            logger.info(f"GLB质量分割PLY已保存: {ply_path}")
            logger.info(f"  点数: {len(vertices)}")
            logger.info(f"  质量: 与GLB模型输出一致")
            logger.info(f"  过滤: 使用了VGGT官方过滤算法")
            
            return ply_path
            
        except Exception as e:
            logger.error(f"导出GLB质量分割PLY失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到标准方法
            logger.info("回退到标准空间保持方法")
            return self._export_segmented_ply_with_spatial_preservation(
                raw_vggt_result, mask_sequence, target_object_id, output_dir, timestamp
            )

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTMaskProcessorNode": VGGTMaskProcessorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTMaskProcessorNode": "🎭 VGGT Mask Processor",
}
