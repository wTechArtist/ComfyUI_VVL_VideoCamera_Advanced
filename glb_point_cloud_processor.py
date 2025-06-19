import os
import json
import tempfile
import logging
import struct
import time
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
                    "default": "clean_pointcloud",
                    "tooltip": "输出GLB文件名：处理后点云的保存文件名，系统会自动添加.glb扩展名。文件将保存到ComfyUI的output目录中。建议使用描述性名称如'clean_pointcloud'、'filtered_scan'等便于识别"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
    )
    RETURN_NAMES = (
        "processed_glb_path",
    )
    OUTPUT_TOOLTIPS = [
        "处理后的GLB文件完整路径",
    ]
    OUTPUT_NODE = True
    FUNCTION = "process_point_cloud"
    CATEGORY = "💃VVL/Point Cloud Cleaning"

    def process_point_cloud(self,
                          glb_file_path: str,
                          black_threshold: int = 30,
                          output_filename: str = "clean_pointcloud"):
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
            return ("",)
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("",)
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("",)
        
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
                    return ("",)
            
            if not point_clouds:
                error_msg = "GLB文件中没有点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("",)
            
            # 处理每个点云
            processed_point_clouds = []
            total_original_points = 0
            total_removed_points = 0
            
            for name, point_cloud in point_clouds:
                processing_log.append(f"\n处理点云: {name}")
                original_count = len(point_cloud.vertices)
                total_original_points += original_count
                
                # 获取原始点云的所有属性和状态
                original_vertices = point_cloud.vertices.copy()
                colors = None
                
                # 获取颜色信息
                if hasattr(point_cloud.visual, 'vertex_colors') and point_cloud.visual.vertex_colors is not None:
                    colors = point_cloud.visual.vertex_colors.copy()
                elif hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                    colors = point_cloud.colors.copy()
                
                # 初始化掩码（所有点都保留）
                keep_mask = np.ones(len(original_vertices), dtype=bool)
                
                # 删除暗色点（仅基于颜色，不改变顶点坐标）
                if colors is not None:
                    removed_count = self._remove_dark_points(original_vertices, colors, keep_mask, black_threshold, processing_log)
                    processing_log.append(f"删除黑色点: {removed_count} 个")
                else:
                    processing_log.append("跳过黑色点过滤: 点云无颜色信息")
                
                # 应用掩码，保留原始的顶点坐标
                filtered_vertices = original_vertices[keep_mask]
                filtered_colors = colors[keep_mask] if colors is not None else None
                
                remaining_count = len(filtered_vertices)
                removed_count = original_count - remaining_count
                total_removed_points += removed_count
                
                processing_log.append(f"点云 {name}: 原始 {original_count} -> 剩余 {remaining_count} (删除 {removed_count})")
                
                # 创建处理后的点云，保留所有原始属性
                if remaining_count > 0:
                    # 使用原始点云作为模板，只更新顶点和颜色
                    processed_pc = point_cloud.copy()
                    processed_pc.vertices = filtered_vertices
                    if filtered_colors is not None:
                        processed_pc.visual.vertex_colors = filtered_colors
                    
                    processed_point_clouds.append((name, processed_pc))
                    processing_log.append(f"保留点云 {name} 的所有原始属性和状态")
                else:
                    processing_log.append(f"警告: 点云 {name} 处理后没有剩余点")
            
            # 直接修改原始场景，保持所有几何信息不变
            if processed_point_clouds:
                # 使用原始场景作为基础，直接替换点云数据
                new_scene = scene
                
                # 直接修改原始场景中的点云几何体，保持变换不变
                for name, processed_pc in processed_point_clouds:
                    if isinstance(scene, trimesh.Scene):
                        # 获取原始几何体
                        if name in scene.geometry:
                            original_geometry = scene.geometry[name]
                            
                            # 如果是点云，直接修改顶点和颜色
                            if isinstance(original_geometry, trimesh.PointCloud):
                                original_geometry.vertices = processed_pc.vertices
                                if hasattr(processed_pc.visual, 'vertex_colors') and processed_pc.visual.vertex_colors is not None:
                                    original_geometry.visual.vertex_colors = processed_pc.visual.vertex_colors
                                
                                processing_log.append(f"直接修改原始点云 {name}，保持所有变换和属性不变")
                            else:
                                # 如果不是点云类型，替换整个几何体但保持变换
                                scene.delete_geometry(name)
                                scene.add_geometry(processed_pc, node_name=name)
                                processing_log.append(f"替换几何体 {name}，尝试保持变换")
                        else:
                            # 新增几何体
                            scene.add_geometry(processed_pc, node_name=name)
                            processing_log.append(f"添加新点云 {name}")
                    else:
                        # 单个几何体的情况
                        new_scene = processed_pc
                        processing_log.append("单个几何体，直接使用处理后的点云")
                
                processing_log.append("直接修改原始场景，所有变换矩阵和几何属性完全保持不变")
                
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
                    
                    # 验证输出文件的一致性
                    try:
                        # 加载输出文件进行验证
                        verification_scene = trimesh.load(output_path)
                        processing_log.append("文件验证: 输出文件可正常加载")
                        
                        # 检查是否保留了场景结构
                        if isinstance(scene, trimesh.Scene) and isinstance(verification_scene, trimesh.Scene):
                            processing_log.append(f"场景结构验证: 原始 {len(scene.geometry)} 个对象 -> 输出 {len(verification_scene.geometry)} 个对象")
                        
                    except Exception as e:
                        processing_log.append(f"文件验证警告: {str(e)}")
                    
                    processing_log.append("GLB点云黑色点清理完成! 已保留原始模型的大小、朝向和几何属性")
                    return (output_path,)
                else:
                    error_msg = "GLB文件保存失败"
                    processing_log.append(f"错误: {error_msg}")
                    return ("",)
            else:
                error_msg = "处理后没有剩余的点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("",)
                
        except Exception as e:
            error_msg = f"处理GLB文件时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("",)
    
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
    

class GLBPointCloudBounds:
    """GLB点云包围盒计算器 - 计算包围盒并生成带可视化的预览文件（原模型数据保持不变）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要计算包围盒的GLB格式点云文件。支持绝对路径或相对于ComfyUI output目录的相对路径。注意：此节点在原模型基础上添加预览信息，原模型数据保持不变"
                }),
            },
            "optional": {
                "bounding_box_type": (["axis_aligned", "oriented"], {
                    "default": "axis_aligned",
                    "tooltip": "包围盒类型：axis_aligned=轴对齐包围盒(AABB)，计算快速；oriented=有向包围盒(OBB)，体积最小"
                }),
                "units": (["meters", "centimeters", "millimeters"], {
                    "default": "meters", 
                    "tooltip": "输出单位：指定包围盒尺寸的输出单位"
                }),
                "add_bounding_box_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "添加包围盒可视化：是否在输出的点云文件中添加红色包围盒线框点云，便于直接预览验证"
                }),
                "add_coordinate_axes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "添加坐标轴：是否添加RGB坐标轴(X=红色，Y=绿色，Z=蓝色)到输出点云中"
                }),
                "wireframe_density": ("INT", {
                    "default": 150, "min": 20, "max": 200, "step": 10,
                    "tooltip": "线框密度：每条包围盒边的点数，越高线框越清晰但点数越多。推荐50-100获得清晰效果"
                }),
                "enhance_visibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "增强可见性：添加顶点高亮和面中心标记，让包围盒更容易识别"
                }),
                "output_filename": ("STRING", {
                    "default": "pointcloud_with_bounds",
                    "tooltip": "输出文件名：带包围盒可视化的点云文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 输出的GLB文件路径
        "STRING",    # JSON格式的scale数组
    )
    RETURN_NAMES = (
        "output_glb_path",
        "scale_json",
    )
    OUTPUT_TOOLTIPS = [
        "带预览的GLB文件路径 - 包含原始点云+包围盒可视化+坐标轴的完整点云文件",
        "纯净的scale数组JSON: {\"scale\": [长, 宽, 高]} - 仅包含scale信息，可直接用于3D引擎缩放",
    ]
    OUTPUT_NODE = True
    FUNCTION = "calculate_bounds_and_visualize"
    CATEGORY = "💃VVL/Point Cloud Analysis"

    def calculate_bounds_and_visualize(self,
                        glb_file_path: str,
                        bounding_box_type: str = "axis_aligned",
                        units: str = "meters",
                        add_bounding_box_visualization: bool = True,
                        add_coordinate_axes: bool = True,
                        wireframe_density: int = 50,
                        enhance_visibility: bool = True,
                        output_filename: str = "pointcloud_with_bounds"):
        """
        计算GLB点云文件的包围盒尺寸并生成带可视化的点云文件
        """
        
        processing_log = []
        processing_log.append("开始GLB点云包围盒计算和可视化生成...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        self._current_input_path = input_path  # 保存当前输入路径供后续使用
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        try:
            # 加载GLB文件
            processing_log.append("正在加载GLB文件...")
            scene = trimesh.load(input_path)
            
            # 收集所有点云数据，考虑变换矩阵
            all_vertices = []
            point_cloud_count = 0
            
            if isinstance(scene, trimesh.Scene):
                for node_name in scene.graph.nodes:
                    if node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                        transform_matrix = scene.graph[node_name][0]  # 获取变换矩阵
                        
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 应用变换矩阵获取真实世界坐标
                            if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
                                world_vertices = trimesh.transformations.transform_points(geometry.vertices, transform_matrix)
                                processing_log.append(f"发现几何体: {node_name}, 点数: {len(geometry.vertices)}, 已应用变换矩阵")
                            else:
                                world_vertices = geometry.vertices.copy()
                                processing_log.append(f"发现几何体: {node_name}, 点数: {len(geometry.vertices)}, 无变换")
                            
                            all_vertices.append(world_vertices)
                            point_cloud_count += 1
            elif isinstance(scene, trimesh.PointCloud):
                all_vertices.append(scene.vertices)
                point_cloud_count = 1
                processing_log.append(f"发现点云: main_pointcloud, 点数: {len(scene.vertices)}")
            elif hasattr(scene, 'vertices') and scene.vertices is not None:
                all_vertices.append(scene.vertices)
                point_cloud_count = 1
                processing_log.append(f"发现几何体(作为点云处理): main_geometry, 点数: {len(scene.vertices)}")
            else:
                error_msg = "GLB文件中未找到点云或几何体数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            if not all_vertices:
                error_msg = "GLB文件中没有可用的顶点数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 合并所有顶点（已经是世界坐标）
            combined_vertices = np.vstack(all_vertices)
            total_points = len(combined_vertices)
            processing_log.append(f"合并了 {point_cloud_count} 个几何体，总点数: {total_points} (已应用变换矩阵)")
            
            # 计算包围盒
            processing_log.append(f"计算包围盒类型: {bounding_box_type}")
            
            if bounding_box_type == "axis_aligned":
                # 轴对齐包围盒 (AABB)
                min_point = np.min(combined_vertices, axis=0)
                max_point = np.max(combined_vertices, axis=0)
                extents = max_point - min_point
                center = (min_point + max_point) / 2
                
                processing_log.append(f"AABB计算完成:")
                processing_log.append(f"  最小点: [{min_point[0]:.6f}, {min_point[1]:.6f}, {min_point[2]:.6f}]")
                processing_log.append(f"  最大点: [{max_point[0]:.6f}, {max_point[1]:.6f}, {max_point[2]:.6f}]")
                processing_log.append(f"  中心点: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
                processing_log.append(f"  尺寸: [{extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f}]")
                processing_log.append(f"  体积: {np.prod(extents):.6f}")
                
            else:  # oriented
                # 有向包围盒 (OBB)
                try:
                    to_origin, obb_extents = trimesh.bounds.oriented_bounds(combined_vertices)
                    
                    # 计算OBB的中心点
                    obb_center = -to_origin[:3, 3]  # 变换矩阵的平移部分的负值
                    center = obb_center  # 设置center变量
                    extents = obb_extents
                    
                    processing_log.append(f"OBB计算完成:")
                    processing_log.append(f"  中心点: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
                    processing_log.append(f"  尺寸: [{extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f}]")
                    processing_log.append(f"  体积: {np.prod(extents):.6f}")
                    
                except Exception as e:
                    processing_log.append(f"OBB计算失败，回退到AABB: {str(e)}")
                    # 回退到AABB
                    min_point = np.min(combined_vertices, axis=0)
                    max_point = np.max(combined_vertices, axis=0)
                    extents = max_point - min_point
                    center = (min_point + max_point) / 2
                    

            
            # 应用单位转换
            unit_scale = {"meters": 1.0, "centimeters": 100.0, "millimeters": 1000.0}
            scale_factor = unit_scale.get(units, 1.0)
            
            scaled_extents = extents * scale_factor
            
            # 生成scale JSON (按照 [长, 宽, 高] 的顺序，通常是 [X, Y, Z])
            scale_array = [float(scaled_extents[0]), float(scaled_extents[1]), float(scaled_extents[2])]
            scale_json = {
                "name": output_filename,
                "scale": scale_array
            }
            

            
            processing_log.append(f"")
            processing_log.append(f"输出结果 (单位: {units}):")
            processing_log.append(f"  Scale数组: [{scale_array[0]:.6f}, {scale_array[1]:.6f}, {scale_array[2]:.6f}]")
            processing_log.append(f"  长度(X): {scale_array[0]:.6f} {units}")
            processing_log.append(f"  宽度(Y): {scale_array[1]:.6f} {units}")
            processing_log.append(f"  高度(Z): {scale_array[2]:.6f} {units}")
            
            # 生成带可视化的点云文件（原模型数据不变，仅添加预览）
            output_glb_path = ""
            
            try:
                processing_log.append("正在生成带包围盒可视化的点云文件...")
                output_glb_path = self._generate_visualization_pointcloud(
                    combined_vertices, extents, center, bounding_box_type,
                    add_bounding_box_visualization, add_coordinate_axes, 
                    wireframe_density, enhance_visibility, output_filename, processing_log
                )
                processing_log.append("可视化点云文件生成完成!")
            except Exception as e:
                processing_log.append(f"可视化点云生成失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            processing_log.append("GLB点云包围盒计算和可视化完成!")
            
            return (
                output_glb_path,
                json.dumps(scale_json, indent=2),
            )
                
        except Exception as e:
            error_msg = f"计算包围盒时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
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

    def _get_current_input_path(self):
        """获取当前输入文件路径"""
        return getattr(self, '_current_input_path', None)

    def _normalize_colors(self, colors):
        """标准化颜色数组为RGBA格式，0-255范围"""
        if colors is None:
            return None
        
        colors = np.array(colors)
        
        # 如果是0-1范围，转换为0-255
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        # 确保是RGBA格式
        if colors.shape[1] == 3:  # RGB -> RGBA
            alpha_channel = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
            colors = np.hstack([colors, alpha_channel])
        elif colors.shape[1] != 4:
            # 如果不是3或4通道，创建白色RGBA
            colors = np.tile([255, 255, 255, 255], (colors.shape[0], 1)).astype(np.uint8)
        
        return colors

    def _generate_visualization_pointcloud(self, combined_vertices, extents, center, bounds_type,
                                          add_bounding_box_visualization, add_coordinate_axes,
                                          wireframe_density, enhance_visibility, output_filename, processing_log):
        """生成包含原始点云、包围盒线框和坐标轴的可视化点云文件"""
        try:
            # 直接使用已加载的原始场景数据，避免重新加载可能改变朝向
            input_path = self._get_current_input_path()
            
            # 提取原始点云数据（完全保持原始状态）
            all_vertices = []
            all_colors = []
            original_point_count = 0
            
            # 使用之前已经加载的combined_vertices作为基础，重新从原始场景提取完整数据
            original_scene = trimesh.load(input_path)
            
            # 创建一个新的场景，完全保持原始场景的结构和变换
            visualization_scene = trimesh.Scene()
            
            # 保持原始场景的完整结构
            if isinstance(original_scene, trimesh.Scene):
                # 完全复制原始场景的所有几何体和变换
                for node_name in original_scene.graph.nodes:
                    if node_name in original_scene.geometry:
                        geometry = original_scene.geometry[node_name]
                        transform_matrix = original_scene.graph[node_name][0]  # 获取变换矩阵
                        
                        # 完全复制几何体，保持所有属性
                        copied_geometry = geometry.copy()
                        
                        # 添加到新场景，保持原始的变换矩阵
                        visualization_scene.add_geometry(copied_geometry, node_name=node_name, transform=transform_matrix)
                        
                        # 提取顶点用于统计（注意：这里用于显示统计，真实包围盒计算已在前面完成）
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 这里不需要再次应用变换，因为包围盒计算已经在combined_vertices中处理过了
                            all_vertices.append(geometry.vertices)
                            
                            # 保持原始颜色
                            if isinstance(geometry, trimesh.PointCloud):
                                if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                                    colors = geometry.visual.vertex_colors.copy()
                                elif hasattr(geometry, 'colors') and geometry.colors is not None:
                                    colors = geometry.colors.copy()
                                else:
                                    colors = np.tile([255, 255, 255, 255], (len(geometry.vertices), 1))
                            else:
                                if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                                    colors = geometry.visual.vertex_colors.copy()
                                else:
                                    colors = np.tile([200, 200, 200, 255], (len(geometry.vertices), 1))
                            
                            # 确保颜色格式一致 (RGBA, 0-255)
                            colors = self._normalize_colors(colors)
                            all_colors.append(colors)
                            original_point_count += len(geometry.vertices)
                            
                        processing_log.append(f"完全保持原始几何体: {node_name}, 变换矩阵和所有属性已保留")
            
            elif isinstance(original_scene, trimesh.PointCloud):
                # 单个点云，直接复制
                visualization_scene = original_scene.copy()
                all_vertices.append(original_scene.vertices)
                
                if hasattr(original_scene.visual, 'vertex_colors') and original_scene.visual.vertex_colors is not None:
                    colors = original_scene.visual.vertex_colors.copy()
                elif hasattr(original_scene, 'colors') and original_scene.colors is not None:
                    colors = original_scene.colors.copy()
                else:
                    colors = np.tile([255, 255, 255, 255], (len(original_scene.vertices), 1))
                
                colors = self._normalize_colors(colors)
                all_colors.append(colors)
                original_point_count = len(original_scene.vertices)
                processing_log.append(f"完全保持原始点云: {original_point_count:,} 个点")
            
            else:
                # 单个几何体，转换为场景保持结构
                visualization_scene.add_geometry(original_scene.copy(), node_name="main_geometry")
                if hasattr(original_scene, 'vertices') and original_scene.vertices is not None:
                    all_vertices.append(original_scene.vertices)
                    
                    if hasattr(original_scene.visual, 'vertex_colors') and original_scene.visual.vertex_colors is not None:
                        colors = original_scene.visual.vertex_colors.copy()
                    else:
                        colors = np.tile([200, 200, 200, 255], (len(original_scene.vertices), 1))
                    
                    colors = self._normalize_colors(colors)
                    all_colors.append(colors)
                    original_point_count = len(original_scene.vertices)
                    processing_log.append(f"完全保持原始几何体: {original_point_count:,} 个点")
            
            processing_log.append(f"原始模型结构完全保留: {original_point_count:,} 个点，所有变换矩阵和朝向不变")
            
            # 添加包围盒线框点云到场景
            if add_bounding_box_visualization:
                box_vertices, box_colors = self._create_bounding_box_pointcloud(
                    extents, center, bounds_type, wireframe_density, enhance_visibility, processing_log
                )
                if len(box_vertices) > 0:
                    # 创建包围盒点云并添加到场景
                    bounding_box_pointcloud = trimesh.PointCloud(vertices=box_vertices, colors=box_colors)
                    visualization_scene.add_geometry(bounding_box_pointcloud, node_name="bounding_box_visualization")
                    processing_log.append(f"包围盒线框: {len(box_vertices):,} 个点")
            
            # 添加坐标轴点云到场景
            if add_coordinate_axes:
                axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud(
                    extents, center, wireframe_density, processing_log
                )
                if len(axes_vertices) > 0:
                    # 创建坐标轴点云并添加到场景
                    coordinate_axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                    visualization_scene.add_geometry(coordinate_axes_pointcloud, node_name="coordinate_axes")
                    processing_log.append(f"坐标轴: {len(axes_vertices):,} 个点")
            
            # 统计总点数
            total_visualization_points = original_point_count
            box_point_count = 0
            axes_point_count = 0
            
            if add_bounding_box_visualization:
                box_point_count = len(box_vertices) if 'box_vertices' in locals() and len(box_vertices) > 0 else 0
                total_visualization_points += box_point_count
            
            if add_coordinate_axes:
                axes_point_count = len(axes_vertices) if 'axes_vertices' in locals() and len(axes_vertices) > 0 else 0
                total_visualization_points += axes_point_count
            
            processing_log.append(f"总点数: {total_visualization_points:,} 个点")
            
            # 生成输出路径
            output_path = self._generate_output_path(output_filename)
            
            # 保存场景，完全保持原始模型的结构和朝向
            visualization_scene.export(output_path)
            
            processing_log.append(f"可视化点云已保存: {output_path}")
            return output_path
            
        except Exception as e:
            processing_log.append(f"可视化点云生成失败: {str(e)}")
            raise e

    def _create_bounding_box_pointcloud(self, extents, center, bounds_type, wireframe_density, enhance_visibility, processing_log):
        """创建包围盒线框的点云表示"""
        try:
            # 计算包围盒的8个顶点
            min_point = center - extents/2
            max_point = center + extents/2
            
            box_vertices = np.array([
                [min_point[0], min_point[1], min_point[2]],  # 0: min corner
                [max_point[0], min_point[1], min_point[2]],  # 1
                [max_point[0], max_point[1], min_point[2]],  # 2
                [min_point[0], max_point[1], min_point[2]],  # 3
                [min_point[0], min_point[1], max_point[2]],  # 4
                [max_point[0], min_point[1], max_point[2]],  # 5
                [max_point[0], max_point[1], max_point[2]],  # 6: max corner
                [min_point[0], max_point[1], max_point[2]],  # 7
            ])
            
            # 定义包围盒的12条边
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
                (0, 4), (1, 5), (2, 6), (3, 7),  # 垂直边
            ]
            
            wireframe_points = []
            
            # 1. 生成线框边上的点（更密集）
            for start_idx, end_idx in edges:
                start_point = box_vertices[start_idx]
                end_point = box_vertices[end_idx]
                
                # 在每条边上生成密集的点
                for i in range(wireframe_density):
                    t = i / (wireframe_density - 1) if wireframe_density > 1 else 0
                    point = start_point + t * (end_point - start_point)
                    wireframe_points.append(point)
            
            # 2. 添加顶点高亮和面中心标记（可选增强可见性）
            if enhance_visibility:
                # 顶点高亮（每个顶点周围添加小点云）
                vertex_highlight_radius = np.min(extents) * 0.01  # 顶点高亮半径
                highlight_density = max(5, wireframe_density // 10)  # 顶点周围的点数
                
                for vertex in box_vertices:
                    # 在每个顶点周围创建小的点云球
                    for i in range(highlight_density):
                        # 生成随机方向
                        theta = np.random.uniform(0, 2 * np.pi)
                        phi = np.random.uniform(0, np.pi)
                        r = np.random.uniform(0, vertex_highlight_radius)
                        
                        # 球坐标转换为笛卡尔坐标
                        x = vertex[0] + r * np.sin(phi) * np.cos(theta)
                        y = vertex[1] + r * np.sin(phi) * np.sin(theta)
                        z = vertex[2] + r * np.cos(phi)
                        
                        wireframe_points.append([x, y, z])
                
                # 面中心点标记
                face_centers = [
                    # 底面中心
                    (box_vertices[0] + box_vertices[1] + box_vertices[2] + box_vertices[3]) / 4,
                    # 顶面中心
                    (box_vertices[4] + box_vertices[5] + box_vertices[6] + box_vertices[7]) / 4,
                    # 前面中心
                    (box_vertices[0] + box_vertices[1] + box_vertices[4] + box_vertices[5]) / 4,
                    # 后面中心
                    (box_vertices[2] + box_vertices[3] + box_vertices[6] + box_vertices[7]) / 4,
                    # 左面中心
                    (box_vertices[0] + box_vertices[3] + box_vertices[4] + box_vertices[7]) / 4,
                    # 右面中心
                    (box_vertices[1] + box_vertices[2] + box_vertices[5] + box_vertices[6]) / 4,
                ]
                
                # 在每个面中心添加标记点
                face_mark_density = max(3, wireframe_density // 20)
                face_mark_radius = np.min(extents) * 0.005
                
                for face_center in face_centers:
                    for i in range(face_mark_density):
                        # 在面中心周围添加小的点云
                        theta = np.random.uniform(0, 2 * np.pi)
                        phi = np.random.uniform(0, np.pi)
                        r = np.random.uniform(0, face_mark_radius)
                        
                        x = face_center[0] + r * np.sin(phi) * np.cos(theta)
                        y = face_center[1] + r * np.sin(phi) * np.sin(theta)
                        z = face_center[2] + r * np.cos(phi)
                        
                        wireframe_points.append([x, y, z])
            
            wireframe_vertices = np.array(wireframe_points)
            
            # 为包围盒线框设置红色
            wireframe_colors = np.tile([255, 0, 0, 255], (len(wireframe_vertices), 1))
            
            processing_log.append(f"包围盒线框生成: {len(wireframe_vertices):,} 个点 (密度={wireframe_density})")
            
            return wireframe_vertices, wireframe_colors
            
        except Exception as e:
            processing_log.append(f"创建包围盒点云失败: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_coordinate_axes_pointcloud(self, extents, origin_center, wireframe_density, processing_log):
        """创建坐标轴的点云表示"""
        try:
            axis_length = np.max(extents) * 0.4
            axis_thickness = np.min(extents) * 0.005  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density
            axis_thickness_points = max(3, wireframe_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（粗线条）- 从点云中心开始
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0] + t * axis_length, origin_center[1], origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([255, 0, 0, 255])
                
                # 增加厚度（在YZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_y = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0], base_point[1] + offset_y, base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([255, 0, 0, 255])
            
            # X轴箭头头部
            arrow_length = axis_length * 0.1
            arrow_base = axis_length * 0.9
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                # 箭头点
                arrow_x = origin_center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, origin_center[1] + arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1] - arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] + arrow_offset])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（粗线条）- 从点云中心开始
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1] + t * axis_length, origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 255, 0, 255])
                
                # 增加厚度（在XZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1], base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 255, 0, 255])
            
            # Y轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_y = origin_center[1] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0] - arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] + arrow_offset])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] - arrow_offset])
                axes_colors.extend([[0, 255, 0, 255]] * 4)
            
            # Z轴 - 蓝色（粗线条）- 从点云中心开始
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1], origin_center[2] + t * axis_length]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 0, 255, 255])
                
                # 增加厚度（在XY平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_y = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1] + offset_y, base_point[2]]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 0, 255, 255])
            
            # Z轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_z = origin_center[2] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0] - arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0], origin_center[1] + arrow_offset, arrow_z])
                axes_points.append([origin_center[0], origin_center[1] - arrow_offset, arrow_z])
                axes_colors.extend([[0, 0, 255, 255]] * 4)
            
            axes_vertices = np.array(axes_points)
            axes_colors_array = np.array(axes_colors)
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 原点=[{origin_center[0]:.3f}, {origin_center[1]:.3f}, {origin_center[2]:.3f}])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_output_path(self, output_filename):
        """生成输出文件路径"""
        # 确保文件名有正确的扩展名
        if not output_filename.lower().endswith('.glb'):
            output_filename += '.glb'
        
        # 生成输出路径
        output_dir = folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加时间戳避免文件名冲突
        import time
        timestamp = str(int(time.time()))
        name_parts = output_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            timestamped_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            timestamped_filename = f"{output_filename}_{timestamp}"
        
        return os.path.join(output_dir, timestamped_filename)


class GLBPointCloudOriginAdjuster:
    """GLB点云原点调整计算器 - 计算原点调整信息并生成带坐标轴的预览文件（原模型数据保持不变）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要调整原点的GLB格式点云文件。注意：此节点在原模型基础上添加坐标轴预览，原模型数据保持不变"
                }),
            },
            "optional": {
                "origin_mode": (["center", "bottom_center"], {
                    "default": "bottom_center",
                    "tooltip": "原点模式：center=点云几何中心；bottom_center=点云底部中心(脚底)"
                }),
                "output_units": (["meters", "centimeters", "millimeters"], {
                    "default": "centimeters",
                    "tooltip": "输出单位：变换信息的输出单位"
                }),
                "add_coordinate_axes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "添加坐标轴：是否添加RGB坐标轴(X=红色，Y=绿色，Z=蓝色)到输出点云中，显示调整后的原点位置"
                }),
                "wireframe_density": ("INT", {
                    "default": 100, "min": 20, "max": 200, "step": 10,
                    "tooltip": "坐标轴密度：每条坐标轴的点数，越高坐标轴越清晰但点数越多。推荐80-120获得清晰效果"
                }),
                "output_filename": ("STRING", {
                    "default": "adjusted_pointcloud",
                    "tooltip": "输出文件名：调整后的点云文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
        "STRING",    # 变换信息JSON (position)
    )
    RETURN_NAMES = (
        "adjusted_glb_path",
        "transform_info",
    )
    OUTPUT_TOOLTIPS = [
        "带预览的GLB文件路径 - 包含原始点云+坐标轴预览的完整点云文件",
        "纯净的position信息JSON: {\"position\": [x, y, z]} - 仅包含位置信息，可直接用于UE等引擎",
    ]
    OUTPUT_NODE = True
    FUNCTION = "adjust_origin"
    CATEGORY = "💃VVL/Point Cloud Transform"

    def adjust_origin(self,
                     glb_file_path: str,
                     origin_mode: str = "bottom_center",
                     output_units: str = "centimeters",
                     add_coordinate_axes: bool = False,
                     wireframe_density: int = 100,
                     output_filename: str = "adjusted_pointcloud"):
        """
        调整GLB点云的原点位置和旋转
        """
        
        processing_log = []
        processing_log.append("开始GLB点云原点调整...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        try:
            # 加载GLB文件
            processing_log.append("正在加载GLB文件...")
            scene = trimesh.load(input_path)
            
            # 收集所有几何体和点云，考虑变换矩阵
            geometries_to_transform = []
            all_vertices = []
            total_points = 0
            
            if isinstance(scene, trimesh.Scene):
                for node_name in scene.graph.nodes:
                    if node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                        transform_matrix = scene.graph[node_name][0]  # 获取变换矩阵
                        geometries_to_transform.append((node_name, geometry))
                        
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 应用变换矩阵获取真实世界坐标
                            if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
                                world_vertices = trimesh.transformations.transform_points(geometry.vertices, transform_matrix)
                                processing_log.append(f"发现几何体: {node_name}, 顶点数: {len(geometry.vertices)}, 已应用变换矩阵")
                            else:
                                world_vertices = geometry.vertices.copy()
                                processing_log.append(f"发现几何体: {node_name}, 顶点数: {len(geometry.vertices)}, 无变换")
                            
                            all_vertices.append(world_vertices)
                            total_points += len(geometry.vertices)
            else:
                # 单个几何体
                geometries_to_transform.append(("main_geometry", scene))
                if hasattr(scene, 'vertices') and scene.vertices is not None:
                    all_vertices.append(scene.vertices)
                    total_points = len(scene.vertices)
                    processing_log.append(f"发现几何体: main_geometry, 顶点数: {total_points}")
            
            if not all_vertices:
                error_msg = "GLB文件中没有可用的顶点数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 合并所有顶点计算包围盒（已经是世界坐标）
            combined_vertices = np.vstack(all_vertices)
            processing_log.append(f"总顶点数: {total_points:,} (已应用变换矩阵)")
            
            # 计算原始包围盒
            original_min = np.min(combined_vertices, axis=0)
            original_max = np.max(combined_vertices, axis=0)
            original_center = (original_min + original_max) / 2
            original_size = original_max - original_min
            
            processing_log.append(f"原始包围盒:")
            processing_log.append(f"  最小点: [{original_min[0]:.6f}, {original_min[1]:.6f}, {original_min[2]:.6f}]")
            processing_log.append(f"  最大点: [{original_max[0]:.6f}, {original_max[1]:.6f}, {original_max[2]:.6f}]")
            processing_log.append(f"  中心点: [{original_center[0]:.6f}, {original_center[1]:.6f}, {original_center[2]:.6f}]")
            processing_log.append(f"  尺寸: [{original_size[0]:.6f}, {original_size[1]:.6f}, {original_size[2]:.6f}]")
            
            # 确定新的原点位置
            if origin_mode == "center":
                new_origin = original_center.copy()
                processing_log.append("原点模式: 几何中心")
            else:  # bottom_center
                new_origin = np.array([original_center[0], original_center[1], original_min[2]])
                processing_log.append("原点模式: 底部中心(脚底)")
            
            # 计算平移向量
            translation = -new_origin
            processing_log.append(f"平移向量: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
            
            # 创建平移变换矩阵（仅平移，无旋转）
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = translation
            
            # 应用变换到所有几何体
            new_scene = trimesh.Scene()
            transformed_vertices_list = []
            
            for name, geometry in geometries_to_transform:
                # 复制几何体
                new_geometry = geometry.copy()
                
                # 应用变换
                new_geometry.apply_transform(transform_matrix)
                
                # 添加到新场景
                new_scene.add_geometry(new_geometry, node_name=name)
                
                # 收集变换后的顶点用于统计
                if hasattr(new_geometry, 'vertices') and new_geometry.vertices is not None:
                    transformed_vertices_list.append(new_geometry.vertices)
                
                processing_log.append(f"已变换几何体: {name}")
            
            # 添加坐标轴预览（在变换后的原点位置，即(0,0,0)）
            if add_coordinate_axes:
                try:
                    # 计算变换后的包围盒来确定坐标轴长度
                    if transformed_vertices_list:
                        all_transformed_vertices = np.vstack(transformed_vertices_list)
                        transformed_extents = np.max(all_transformed_vertices, axis=0) - np.min(all_transformed_vertices, axis=0)
                        axis_length = np.max(transformed_extents) * 0.4
                    else:
                        axis_length = 1.0  # 默认长度
                    
                    # 在新的原点(0,0,0)处生成坐标轴
                    axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud_at_origin(
                        axis_length, wireframe_density, processing_log
                    )
                    
                    if len(axes_vertices) > 0:
                        # 创建坐标轴点云
                        axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                        new_scene.add_geometry(axes_pointcloud, node_name="coordinate_axes")
                        processing_log.append(f"坐标轴预览: {len(axes_vertices):,} 个点 (显示新的原点位置)")
                    
                except Exception as e:
                    processing_log.append(f"坐标轴生成失败: {str(e)}")
            
            # 计算变换后的包围盒
            if transformed_vertices_list:
                transformed_vertices = np.vstack(transformed_vertices_list)
                new_min = np.min(transformed_vertices, axis=0)
                new_max = np.max(transformed_vertices, axis=0)
                new_center = (new_min + new_max) / 2
                new_size = new_max - new_min
                
                processing_log.append(f"变换后包围盒:")
                processing_log.append(f"  最小点: [{new_min[0]:.6f}, {new_min[1]:.6f}, {new_min[2]:.6f}]")
                processing_log.append(f"  最大点: [{new_max[0]:.6f}, {new_max[1]:.6f}, {new_max[2]:.6f}]")
                processing_log.append(f"  中心点: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
                processing_log.append(f"  尺寸: [{new_size[0]:.6f}, {new_size[1]:.6f}, {new_size[2]:.6f}]")
            
            # 生成带预览的输出文件（原模型数据不变，仅添加坐标轴预览）
            output_path = self._generate_output_path(output_filename)
            processing_log.append(f"输出文件路径: {output_path}")
            
            # 创建包含原始点云和坐标轴的预览场景，完全保持原始结构
            original_scene = trimesh.load(input_path)  # 重新加载原始场景
            
            # 创建预览场景，完全复制原始场景的结构
            if isinstance(original_scene, trimesh.Scene):
                preview_scene = trimesh.Scene()
                # 完全复制原始场景的所有几何体和变换矩阵
                for node_name in original_scene.graph.nodes:
                    if node_name in original_scene.geometry:
                        geometry = original_scene.geometry[node_name]
                        transform_matrix = original_scene.graph[node_name][0]  # 获取变换矩阵
                        
                        # 完全复制几何体，保持所有属性
                        copied_geometry = geometry.copy()
                        
                        # 添加到预览场景，保持原始的变换矩阵
                        preview_scene.add_geometry(copied_geometry, node_name=node_name, transform=transform_matrix)
                        processing_log.append(f"完全保持原始几何体: {node_name}, 变换矩阵和朝向已保留")
            else:
                # 单个几何体的情况
                preview_scene = trimesh.Scene()
                preview_scene.add_geometry(original_scene.copy(), node_name="main_geometry")
                processing_log.append("完全保持原始几何体: main_geometry, 所有属性已保留")
            
            # 添加坐标轴预览（如果需要）
            if add_coordinate_axes:
                try:
                    # 使用原始包围盒来确定坐标轴长度
                    axis_length = np.max(original_size) * 0.4
                    
                    # 在计算出的新原点位置生成坐标轴
                    axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud_at_position(
                        new_origin, axis_length, wireframe_density, processing_log
                    )
                    
                    if len(axes_vertices) > 0:
                        # 创建坐标轴点云并添加到场景
                        axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                        preview_scene.add_geometry(axes_pointcloud, node_name="coordinate_axes_preview")
                        processing_log.append(f"坐标轴预览: {len(axes_vertices):,} 个点 (显示建议的新原点位置)")
                    
                except Exception as e:
                    processing_log.append(f"坐标轴生成失败: {str(e)}")
            
            # 保存预览文件
            processing_log.append("正在保存带预览的GLB文件...")
            preview_scene.export(output_path)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                processing_log.append(f"预览GLB文件保存成功，文件大小: {file_size} bytes")
            else:
                error_msg = "预览GLB文件保存失败"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 应用单位转换
            unit_scale = {"meters": 1.0, "centimeters": 100.0, "millimeters": 1000.0}
            scale_factor = unit_scale.get(output_units, 1.0)
            
            # 生成变换信息（UE格式）
            final_position = new_origin * scale_factor  # 相对于原始坐标系的位置
            transform_info = {
                "name": output_filename,
                "position": [float(final_position[0]), float(final_position[1]), float(final_position[2])]
            }
            
            processing_log.append("")
            processing_log.append(f"变换信息 (单位: {output_units}):")
            processing_log.append(f"  Position: [{transform_info['position'][0]:.2f}, {transform_info['position'][1]:.2f}, {transform_info['position'][2]:.2f}]")
            processing_log.append("原点调整预览生成完成!")
            
            return (
                output_path,
                json.dumps(transform_info, indent=2),
            )
                
        except Exception as e:
            error_msg = f"调整原点时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
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
        # 确保文件名有正确的扩展名
        if not filename.lower().endswith('.glb'):
            filename += '.glb'
        
        # 生成输出路径
        output_dir = folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加时间戳避免文件名冲突
        timestamp = str(int(time.time()))
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            timestamped_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            timestamped_filename = f"{filename}_{timestamp}"
        
        return os.path.join(output_dir, timestamped_filename)
    

    
    def _create_coordinate_axes_pointcloud_at_position(self, position, axis_length, wireframe_density, processing_log):
        """在指定位置创建坐标轴的点云表示"""
        try:
            center = np.array(position)  # 指定的位置
            axis_thickness = axis_length * 0.01  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density  # 使用参数控制密度
            axis_thickness_points = max(3, axis_line_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0] + t * axis_length, center[1], center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([255, 0, 0, 255])
                
                # 增加厚度（在YZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_y = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0], base_point[1] + offset_y, base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([255, 0, 0, 255])
            
            # X轴箭头头部
            arrow_length = axis_length * 0.1
            arrow_base = axis_length * 0.9
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_x = center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, center[1] + arrow_offset, center[2]])
                axes_points.append([arrow_x, center[1] - arrow_offset, center[2]])
                axes_points.append([arrow_x, center[1], center[2] + arrow_offset])
                axes_points.append([arrow_x, center[1], center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0], center[1] + t * axis_length, center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 255, 0, 255])
                
                # 增加厚度（在XZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1], base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 255, 0, 255])
            
            # Y轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_y = center[1] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([center[0] + arrow_offset, arrow_y, center[2]])
                axes_points.append([center[0] - arrow_offset, arrow_y, center[2]])
                axes_points.append([center[0], arrow_y, center[2] + arrow_offset])
                axes_points.append([center[0], arrow_y, center[2] - arrow_offset])
                axes_colors.extend([[0, 255, 0, 255]] * 4)
            
            # Z轴 - 蓝色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0], center[1], center[2] + t * axis_length]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 0, 255, 255])
                
                # 增加厚度（在XY平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_y = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1] + offset_y, base_point[2]]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 0, 255, 255])
            
            # Z轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_z = center[2] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([center[0] + arrow_offset, center[1], arrow_z])
                axes_points.append([center[0] - arrow_offset, center[1], arrow_z])
                axes_points.append([center[0], center[1] + arrow_offset, arrow_z])
                axes_points.append([center[0], center[1] - arrow_offset, arrow_z])
                axes_colors.extend([[0, 0, 255, 255]] * 4)
            
            axes_vertices = np.array(axes_points)
            axes_colors_array = np.array(axes_colors)
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 密度={wireframe_density}, 位置=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])

    def _create_coordinate_axes_pointcloud_at_origin(self, axis_length, wireframe_density, processing_log):
        """在原点(0,0,0)创建坐标轴的点云表示"""
        try:
            origin_center = np.array([0.0, 0.0, 0.0])  # 新的原点位置
            axis_thickness = axis_length * 0.01  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density  # 使用参数控制密度
            axis_thickness_points = max(3, axis_line_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0] + t * axis_length, origin_center[1], origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([255, 0, 0, 255])
                
                # 增加厚度（在YZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_y = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0], base_point[1] + offset_y, base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([255, 0, 0, 255])
            
            # X轴箭头头部
            arrow_length = axis_length * 0.1
            arrow_base = axis_length * 0.9
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_x = origin_center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, origin_center[1] + arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1] - arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] + arrow_offset])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1] + t * axis_length, origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 255, 0, 255])
                
                # 增加厚度（在XZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1], base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 255, 0, 255])
            
            # Y轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_y = origin_center[1] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0] - arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] + arrow_offset])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] - arrow_offset])
                axes_colors.extend([[0, 255, 0, 255]] * 4)
            
            # Z轴 - 蓝色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1], origin_center[2] + t * axis_length]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 0, 255, 255])
                
                # 增加厚度（在XY平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_y = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1] + offset_y, base_point[2]]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 0, 255, 255])
            
            # Z轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_z = origin_center[2] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0] - arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0], origin_center[1] + arrow_offset, arrow_z])
                axes_points.append([origin_center[0], origin_center[1] - arrow_offset, arrow_z])
                axes_colors.extend([[0, 0, 255, 255]] * 4)
            
            axes_vertices = np.array(axes_points)
            axes_colors_array = np.array(axes_colors)
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 密度={wireframe_density}, 原点=[0.000, 0.000, 0.000])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])


class GLBPointCloudDensityFilter:
    """GLB点云密度过滤器 - 根据局部密度删除稀疏区域，保留密度最高的核心区域"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要进行密度过滤的GLB格式点云文件。支持绝对路径或相对于ComfyUI output目录的相对路径"
                }),
            },
            "optional": {
                "density_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "密度阈值：保留密度高于此比例的点。0.3=保留密度前70%的点；0.5=保留密度前50%的点。值越大删除越多稀疏点"
                }),
                "neighborhood_radius": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "邻域半径：计算密度时的搜索半径。较小值=精细密度检测；较大值=平滑密度检测。建议0.05-0.2"
                }),
                "min_neighbors": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "最小邻居数：一个点在邻域内至少需要的邻居点数量才不被视为稀疏点。值越大要求密度越高"
                }),
                "preserve_core_percentage": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "核心区域保留比例：无论密度如何，都会保留密度最高的这部分点。0.8=保证保留最密集的80%区域"
                }),
                "use_adaptive_radius": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自适应半径：根据点云整体尺寸自动调整邻域半径，确保在不同尺寸的点云上都有良好效果"
                }),
                "output_filename": ("STRING", {
                    "default": "density_filtered_pointcloud",
                    "tooltip": "输出GLB文件名：密度过滤后点云的保存文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
        "STRING",    # 过滤统计信息JSON
    )
    RETURN_NAMES = (
        "filtered_glb_path",
        "filter_stats",
    )
    OUTPUT_TOOLTIPS = [
        "密度过滤后的GLB文件完整路径",
        "过滤统计信息JSON：包含原始点数、保留点数、删除点数等统计数据",
    ]
    OUTPUT_NODE = True
    FUNCTION = "filter_by_density"
    CATEGORY = "💃VVL/Point Cloud Filtering"

    def filter_by_density(self,
                         glb_file_path: str,
                         density_threshold: float = 0.3,
                         neighborhood_radius: float = 0.1,
                         min_neighbors: int = 5,
                         preserve_core_percentage: float = 0.8,
                         use_adaptive_radius: bool = True,
                         output_filename: str = "density_filtered_pointcloud"):
        """
        根据局部密度过滤GLB点云文件，删除稀疏区域
        """
        
        processing_log = []
        processing_log.append("开始GLB点云密度过滤...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
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
                    return ("", "")
            
            if not point_clouds:
                error_msg = "GLB文件中没有点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 处理每个点云
            filtered_point_clouds = []
            total_original_points = 0
            total_filtered_points = 0
            
            for name, point_cloud in point_clouds:
                processing_log.append(f"\n处理点云: {name}")
                original_count = len(point_cloud.vertices)
                total_original_points += original_count
                
                # 获取原始点云的所有属性和状态
                original_vertices = point_cloud.vertices.copy()
                colors = None
                
                # 获取颜色信息
                if hasattr(point_cloud.visual, 'vertex_colors') and point_cloud.visual.vertex_colors is not None:
                    colors = point_cloud.visual.vertex_colors.copy()
                elif hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                    colors = point_cloud.colors.copy()
                
                # 应用密度过滤（仅基于密度，不改变顶点坐标）
                keep_mask, filter_info = self._apply_density_filter(
                    original_vertices, colors, density_threshold, neighborhood_radius, min_neighbors,
                    preserve_core_percentage, use_adaptive_radius, processing_log
                )
                
                # 应用掩码，保留原始的顶点坐标
                filtered_vertices = original_vertices[keep_mask]
                filtered_colors = colors[keep_mask] if colors is not None else None
                
                filtered_count = len(filtered_vertices)
                total_filtered_points += filtered_count
                
                processing_log.append(f"点云 {name}: 原始 {original_count} -> 保留 {filtered_count} (删除 {original_count - filtered_count})")
                
                # 创建过滤后的点云，保留所有原始属性
                if filtered_count > 0:
                    # 使用原始点云作为模板，只更新顶点和颜色
                    filtered_pc = point_cloud.copy()
                    filtered_pc.vertices = filtered_vertices
                    if filtered_colors is not None:
                        filtered_pc.visual.vertex_colors = filtered_colors
                    
                    filtered_point_clouds.append((name, filtered_pc))
                    processing_log.append(f"保留点云 {name} 的所有原始属性和状态")
                else:
                    processing_log.append(f"警告: 点云 {name} 过滤后没有剩余点")
            
            # 直接修改原始场景，保持所有几何信息不变
            if filtered_point_clouds:
                # 使用原始场景作为基础，直接替换点云数据
                new_scene = scene
                
                # 直接修改原始场景中的点云几何体，保持变换不变
                for name, filtered_pc in filtered_point_clouds:
                    if isinstance(scene, trimesh.Scene):
                        # 获取原始几何体
                        if name in scene.geometry:
                            original_geometry = scene.geometry[name]
                            
                            # 如果是点云，直接修改顶点和颜色
                            if isinstance(original_geometry, trimesh.PointCloud):
                                original_geometry.vertices = filtered_pc.vertices
                                if hasattr(filtered_pc.visual, 'vertex_colors') and filtered_pc.visual.vertex_colors is not None:
                                    original_geometry.visual.vertex_colors = filtered_pc.visual.vertex_colors
                                
                                processing_log.append(f"直接修改原始点云 {name}，保持所有变换和属性不变")
                            else:
                                # 如果不是点云类型，替换整个几何体但保持变换
                                scene.delete_geometry(name)
                                scene.add_geometry(filtered_pc, node_name=name)
                                processing_log.append(f"替换几何体 {name}，尝试保持变换")
                        else:
                            # 新增几何体
                            scene.add_geometry(filtered_pc, node_name=name)
                            processing_log.append(f"添加新点云 {name}")
                    else:
                        # 单个几何体的情况
                        new_scene = filtered_pc
                        processing_log.append("单个几何体，直接使用过滤后的点云")
                
                processing_log.append("直接修改原始场景，所有变换矩阵和几何属性完全保持不变")
                
                # 生成输出路径
                output_path = self._generate_output_path(output_filename)
                processing_log.append(f"输出文件路径: {output_path}")
                
                # 保存过滤后的GLB文件
                processing_log.append("正在保存过滤后的GLB文件...")
                new_scene.export(output_path)
                
                # 验证文件是否保存成功
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    processing_log.append(f"GLB文件保存成功，文件大小: {file_size} bytes")
                    
                    # 生成统计信息
                    filter_stats = {
                        "original_points": total_original_points,
                        "filtered_points": total_filtered_points,
                        "removed_points": total_original_points - total_filtered_points,
                        "retention_rate": total_filtered_points / total_original_points if total_original_points > 0 else 0,
                        "parameters": {
                            "density_threshold": density_threshold,
                            "neighborhood_radius": neighborhood_radius,
                            "min_neighbors": min_neighbors,
                            "preserve_core_percentage": preserve_core_percentage,
                            "use_adaptive_radius": use_adaptive_radius
                        }
                    }
                    
                    stats_json = json.dumps(filter_stats, indent=2)
                    
                    processing_log.append("GLB点云密度过滤完成! 已保留原始模型的大小、朝向和几何属性")
                    processing_log.append(f"统计: 原始{total_original_points}点 -> 保留{total_filtered_points}点 (保留率: {filter_stats['retention_rate']:.1%})")
                    
                    return (output_path, stats_json)
                else:
                    error_msg = "GLB文件保存失败"
                    processing_log.append(f"错误: {error_msg}")
                    return ("", "")
            else:
                error_msg = "过滤后没有剩余的点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
                
        except Exception as e:
            error_msg = f"处理GLB文件时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
    def _apply_density_filter(self, vertices, colors, density_threshold, neighborhood_radius, 
                             min_neighbors, preserve_core_percentage, use_adaptive_radius, processing_log):
        """
        应用密度过滤算法，返回保留点的掩码
        """
        
        n_points = len(vertices)
        processing_log.append(f"开始密度分析，总点数: {n_points}")
        
        # 自适应半径调整
        if use_adaptive_radius:
            # 计算点云的包围盒对角线长度
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
            
            # 根据包围盒大小调整半径
            adaptive_radius = neighborhood_radius * bbox_diagonal / 10.0  # 可调整的比例因子
            processing_log.append(f"自适应半径: {neighborhood_radius} -> {adaptive_radius:.4f} (基于包围盒对角线 {bbox_diagonal:.4f})")
            neighborhood_radius = adaptive_radius
        
        # 使用KDTree进行高效的邻域搜索
        try:
            from scipy.spatial import cKDTree
            kdtree = cKDTree(vertices)
            processing_log.append("使用scipy.spatial.cKDTree进行邻域搜索")
        except ImportError:
            processing_log.append("scipy不可用，使用简化的距离计算")
            kdtree = None
        
        # 计算每个点的局部密度
        densities = np.zeros(n_points)
        
        if kdtree is not None:
            # 高效的KDTree搜索
            for i in range(n_points):
                # 查找邻域内的点
                neighbor_indices = kdtree.query_ball_point(vertices[i], neighborhood_radius)
                densities[i] = len(neighbor_indices) - 1  # 减去自己
        else:
            # 简化的距离计算（适用于小点云）
            for i in range(n_points):
                distances = np.linalg.norm(vertices - vertices[i], axis=1)
                neighbor_count = np.sum(distances <= neighborhood_radius) - 1  # 减去自己
                densities[i] = neighbor_count
        
        processing_log.append(f"密度统计: 最小={densities.min():.1f}, 最大={densities.max():.1f}, 平均={densities.mean():.1f}")
        
        # 应用多层过滤策略
        keep_mask = np.ones(n_points, dtype=bool)
        
        # 1. 基于最小邻居数的过滤
        min_neighbor_mask = densities >= min_neighbors
        removed_by_min_neighbors = np.sum(~min_neighbor_mask)
        keep_mask &= min_neighbor_mask
        processing_log.append(f"最小邻居数过滤: 删除 {removed_by_min_neighbors} 个点")
        
        # 2. 基于密度阈值的过滤
        if density_threshold > 0 and np.sum(keep_mask) > 0:
            valid_densities = densities[keep_mask]
            density_percentile = np.percentile(valid_densities, (1 - density_threshold) * 100)
            density_mask = densities >= density_percentile
            
            # 保留当前mask中的点，再应用密度过滤
            temp_mask = keep_mask.copy()
            keep_mask[temp_mask] &= density_mask[temp_mask]
            
            removed_by_density = np.sum(temp_mask) - np.sum(keep_mask)
            processing_log.append(f"密度阈值过滤: 删除 {removed_by_density} 个点 (阈值密度: {density_percentile:.1f})")
        
        # 3. 核心区域保护
        if preserve_core_percentage < 1.0 and np.sum(keep_mask) > 0:
            # 计算需要保留的核心点数
            target_core_points = int(n_points * preserve_core_percentage)
            current_points = np.sum(keep_mask)
            
            if current_points < target_core_points:
                # 当前保留的点数少于核心要求，需要添加高密度点
                need_more_points = target_core_points - current_points
                
                # 找到密度最高的点
                excluded_indices = np.where(~keep_mask)[0]
                if len(excluded_indices) > 0:
                    excluded_densities = densities[excluded_indices]
                    # 按密度排序，选择密度最高的点
                    sorted_indices = excluded_indices[np.argsort(-excluded_densities)]
                    points_to_restore = min(need_more_points, len(sorted_indices))
                    
                    keep_mask[sorted_indices[:points_to_restore]] = True
                    processing_log.append(f"核心区域保护: 恢复 {points_to_restore} 个高密度点")
        
        # 统计信息
        original_count = n_points
        filtered_count = np.sum(keep_mask)
        removed_count = original_count - filtered_count
        
        filter_info = {
            "original_points": original_count,
            "filtered_points": filtered_count,
            "removed_points": removed_count,
            "retention_rate": filtered_count / original_count if original_count > 0 else 0,
            "density_stats": {
                "min": float(densities.min()),
                "max": float(densities.max()),
                "mean": float(densities.mean()),
                "std": float(densities.std())
            }
        }
        
        processing_log.append(f"密度过滤完成: 保留 {filtered_count}/{original_count} 点 (保留率: {filter_info['retention_rate']:.1%})")
        
        return keep_mask, filter_info
    
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


# -----------------------------------------------------------------------------
# 节点注册 - 更新注册信息
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GLBPointCloudProcessor": GLBPointCloudProcessor,
    "GLBPointCloudBounds": GLBPointCloudBounds,
    "GLBPointCloudOriginAdjuster": GLBPointCloudOriginAdjuster,
    "GLBPointCloudDensityFilter": GLBPointCloudDensityFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLBPointCloudProcessor": "VVL GLB Point Cloud Processor",
    "GLBPointCloudBounds": "VVL GLB Point Cloud Bounds Visualizer",
    "GLBPointCloudOriginAdjuster": "VVL GLB Point Cloud Origin Adjuster",
    "GLBPointCloudDensityFilter": "VVL GLB Point Cloud Density Filter",
}

# # 添加节点信息，帮助ComfyUI更好地识别
# WEB_DIRECTORY = "./web"
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]