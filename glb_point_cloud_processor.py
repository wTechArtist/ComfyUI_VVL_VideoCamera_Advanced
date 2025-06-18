import os
import json
import tempfile
import logging
import struct
from typing import List, Any, Dict, Tuple

import numpy as np
import torch

# 导入ComfyUI的路径管理
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
    FOLDER_PATHS_AVAILABLE = False

# 导入trimesh用于GLB文件处理
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

# 配置日志
logger = logging.getLogger('glb_point_cloud_processor')

class GLBPointCloudProcessor:
    """GLB点云文件处理器 - 专门用于删除黑色点和暗色点优化"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要处理的GLB格式点云文件路径。支持绝对路径(如C:/path/to/file.glb)或相对于ComfyUI output目录的相对路径(如pointcloud.glb)。GLB文件应包含点云数据，支持带颜色信息的点云处理"
                }),
            },
            "optional": {
                "black_threshold": ("INT", {
                    "default": 30, "min": 0, "max": 255, "step": 1,
                    "tooltip": "黑色阈值控制：RGB三色总和小于此值的点将被删除。0=仅删除纯黑点(0,0,0)；30=删除深黑色点；60=删除较暗点；100=删除中等暗度点；255=删除所有点。调大=删除更多暗色点，点云更亮净；调小=保留更多暗色细节，但可能有噪点"
                }),
                "output_filename": ("STRING", {
                    "default": "clean_pointcloud.glb",
                    "tooltip": "输出GLB文件名：处理后点云的保存文件名，自动添加.glb扩展名。文件将保存到ComfyUI的output目录中。建议使用描述性名称如'clean_pointcloud.glb'、'filtered_scan.glb'等便于识别"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
        "STRING",    # 处理统计信息（JSON格式）
        "STRING",    # 处理日志信息
    )
    RETURN_NAMES = (
        "processed_glb_path",
        "processing_stats",
        "processing_log",
    )
    OUTPUT_TOOLTIPS = [
        "处理后的GLB文件完整路径",
        "处理统计信息（JSON格式）- 包含删除的点数、剩余点数等",
        "详细的处理日志信息"
    ]
    OUTPUT_NODE = True
    FUNCTION = "process_point_cloud"
    CATEGORY = "💃VVL/Point Cloud Cleaning"

    def process_point_cloud(self,
                          glb_file_path: str,
                          black_threshold: int = 30,
                          output_filename: str = "clean_pointcloud.glb"):
        """
        处理GLB点云文件，删除黑色点和暗色点
        """
        
        processing_log = []
        processing_log.append("开始GLB点云黑色点清理...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
        
        try:
            # 加载GLB文件
            processing_log.append("正在加载GLB文件...")
            scene = trimesh.load(input_path)
            
            # 提取点云数据
            point_clouds = []
            other_geometries = []
            
            if isinstance(scene, trimesh.Scene):
                for name, geometry in scene.geometry.items():
                    if isinstance(geometry, trimesh.PointCloud):
                        point_clouds.append((name, geometry))
                        processing_log.append(f"发现点云: {name}, 点数: {len(geometry.vertices)}")
                    else:
                        other_geometries.append((name, geometry))
                        processing_log.append(f"发现其他几何体: {name}, 类型: {type(geometry).__name__}")
            elif isinstance(scene, trimesh.PointCloud):
                point_clouds.append(("main_pointcloud", scene))
                processing_log.append(f"发现点云: main_pointcloud, 点数: {len(scene.vertices)}")
            else:
                # 尝试转换为点云
                if hasattr(scene, 'vertices'):
                    pc = trimesh.PointCloud(vertices=scene.vertices, colors=getattr(scene.visual, 'vertex_colors', None))
                    point_clouds.append(("converted_pointcloud", pc))
                    processing_log.append(f"转换为点云: converted_pointcloud, 点数: {len(pc.vertices)}")
                else:
                    error_msg = "GLB文件中未找到点云数据"
                    processing_log.append(f"错误: {error_msg}")
                    return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
            
            if not point_clouds:
                error_msg = "GLB文件中没有点云数据"
                processing_log.append(f"错误: {error_msg}")
                return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
            
            # 处理每个点云
            processed_point_clouds = []
            total_original_points = 0
            total_removed_points = 0
            
            for name, point_cloud in point_clouds:
                processing_log.append(f"\n处理点云: {name}")
                original_count = len(point_cloud.vertices)
                total_original_points += original_count
                
                # 获取顶点和颜色
                vertices = point_cloud.vertices.copy()
                colors = None
                
                if hasattr(point_cloud.visual, 'vertex_colors') and point_cloud.visual.vertex_colors is not None:
                    colors = point_cloud.visual.vertex_colors.copy()
                elif hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                    colors = point_cloud.colors.copy()
                
                # 初始化掩码（所有点都保留）
                keep_mask = np.ones(len(vertices), dtype=bool)
                
                # 删除暗色点（默认启用）
                if colors is not None:
                    removed_count = self._remove_dark_points(vertices, colors, keep_mask, black_threshold, processing_log)
                    processing_log.append(f"删除黑色点: {removed_count} 个")
                else:
                    processing_log.append("跳过黑色点过滤: 点云无颜色信息")
                
                # 应用掩码
                filtered_vertices = vertices[keep_mask]
                filtered_colors = colors[keep_mask] if colors is not None else None
                
                remaining_count = len(filtered_vertices)
                removed_count = original_count - remaining_count
                total_removed_points += removed_count
                
                processing_log.append(f"点云 {name}: 原始 {original_count} -> 剩余 {remaining_count} (删除 {removed_count})")
                
                # 创建处理后的点云
                if remaining_count > 0:
                    processed_pc = trimesh.PointCloud(vertices=filtered_vertices, colors=filtered_colors)
                    processed_point_clouds.append((name, processed_pc))
                else:
                    processing_log.append(f"警告: 点云 {name} 处理后没有剩余点")
            
            # 创建新的场景
            if processed_point_clouds:
                new_scene = trimesh.Scene()
                
                # 添加处理后的点云
                for name, pc in processed_point_clouds:
                    new_scene.add_geometry(pc, node_name=name)
                
                # 添加其他几何体（如相机模型）
                for name, geometry in other_geometries:
                    new_scene.add_geometry(geometry, node_name=name)
                    processing_log.append(f"保留几何体: {name}")
                
                # 生成输出路径
                output_path = self._generate_output_path(output_filename)
                processing_log.append(f"输出文件路径: {output_path}")
                
                # 保存处理后的GLB文件
                processing_log.append("正在保存处理后的GLB文件...")
                new_scene.export(output_path)
                
                # 验证文件是否保存成功
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    processing_log.append(f"GLB文件保存成功，文件大小: {file_size} bytes")
                    
                    # 生成统计信息
                    stats = {
                        "original_points": total_original_points,
                        "remaining_points": total_original_points - total_removed_points,
                        "removed_points": total_removed_points,
                        "removal_percentage": (total_removed_points / total_original_points * 100) if total_original_points > 0 else 0,
                        "point_clouds_processed": len(point_clouds),
                        "other_geometries_preserved": len(other_geometries),
                        "output_file_size": file_size,
                        "settings": {
                            "black_threshold": black_threshold,
                            "remove_dark_points": True
                        }
                    }
                    
                    processing_log.append("GLB点云黑色点清理完成!")
                    return output_path, json.dumps(stats, indent=2), "\n".join(processing_log)
                else:
                    error_msg = "GLB文件保存失败"
                    processing_log.append(f"错误: {error_msg}")
                    return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
            else:
                error_msg = "处理后没有剩余的点云数据"
                processing_log.append(f"错误: {error_msg}")
                return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
                
        except Exception as e:
            error_msg = f"处理GLB文件时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return "", json.dumps({"error": error_msg}), "\n".join(processing_log)
    
    def _resolve_file_path(self, file_path: str) -> str:
        """解析文件路径，支持绝对路径和相对路径"""
        if os.path.isabs(file_path):
            return file_path
        
        # 尝试相对于ComfyUI输出目录
        if FOLDER_PATHS_AVAILABLE:
            output_dir = folder_paths.get_output_directory()
            candidate_path = os.path.join(output_dir, file_path)
            if os.path.exists(candidate_path):
                return candidate_path
        
        # 尝试相对于当前工作目录
        if os.path.exists(file_path):
            return os.path.abspath(file_path)
        
        # 返回原始路径（让后续检查处理错误）
        return file_path
    
    def _generate_output_path(self, filename: str) -> str:
        """生成输出文件路径"""
        if FOLDER_PATHS_AVAILABLE:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = "output"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理文件名，确保是.glb扩展名
        if not filename.lower().endswith('.glb'):
            filename = f"{filename}.glb"
        
        return os.path.join(output_dir, filename)
    
    def _remove_dark_points(self, vertices: np.ndarray, colors: np.ndarray, keep_mask: np.ndarray, 
                           black_threshold: int, processing_log: List[str]) -> int:
        """删除暗色点"""
        removed_count = 0
        
        # 确保颜色数据格式正确
        if colors.shape[1] == 4:  # RGBA
            rgb_colors = colors[:, :3]
        else:  # RGB
            rgb_colors = colors
        
        # 如果颜色值在0-1范围内，转换为0-255
        if rgb_colors.max() <= 1.0:
            rgb_colors = (rgb_colors * 255).astype(np.uint8)
        else:
            rgb_colors = rgb_colors.astype(np.uint8)
        
        # 计算RGB总和
        rgb_sum = rgb_colors.sum(axis=1)
        

        
        # 应用黑色阈值过滤
        if black_threshold > 0:
            dark_mask = rgb_sum >= black_threshold
            before_count = keep_mask.sum()
            keep_mask &= dark_mask
            after_count = keep_mask.sum()
            removed_by_black = before_count - after_count
            removed_count += removed_by_black
            processing_log.append(f"  按RGB总和阈值({black_threshold})删除: {removed_by_black} 个点")
        
        return removed_count
    



# -----------------------------------------------------------------------------
# 节点注册 - 独立注册，ComfyUI会自动发现此文件
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GLBPointCloudProcessor": GLBPointCloudProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLBPointCloudProcessor": "VVL GLB Point Cloud Processor",
}

# # 添加节点信息，帮助ComfyUI更好地识别
# WEB_DIRECTORY = "./web"
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 