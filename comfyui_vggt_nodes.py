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

# 导入 VGGT 相关函数
try:
    # 确保可以找到vggt模块
    import sys
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    VGGT_UTILS_AVAILABLE = True
except Exception as e:
    load_and_preprocess_images = None
    pose_encoding_to_extri_intri = None
    unproject_depth_map_to_point_map = None
    VGGT_UTILS_AVAILABLE = False
    _VGGT_UTILS_IMPORT_ERROR = e

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

# 原版VGGT的dependencies
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    _TRIMESH_IMPORT_ERROR = "trimesh not available"

try:
    import matplotlib
    import matplotlib.colormaps
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    _MATPLOTLIB_IMPORT_ERROR = "matplotlib not available"

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    _SCIPY_IMPORT_ERROR = "scipy not available"

try:
    import copy
    import requests
    EXTRA_DEPS_AVAILABLE = True
except ImportError:
    EXTRA_DEPS_AVAILABLE = False

# 配置日志
logger = logging.getLogger('vvl_vggt_nodes')

# -----------------------------------------------------------------------------
# 原生VGGT推理函数
# -----------------------------------------------------------------------------

def run_vggt_native_inference(images: torch.Tensor, model_instance, device) -> Dict:
    """使用原生VGGT接口进行推理"""
    if not VGGT_NATIVE_AVAILABLE:
        raise ImportError("VGGT Native Interface not available")
    
    try:
        # 创建原生接口
        vggt_interface = VGGTNativeInterface(model_instance, device)
        
        # 执行完整推理流程
        results = vggt_interface.full_inference_pipeline(images)
        
        logger.info(f"VGGT native inference completed. Results keys: {list(results.keys())}")
        return results
        
    except Exception as e:
        logger.error(f"VGGT native inference failed: {e}")
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

def format_results_native(results: Dict) -> Dict:
    """简化的结果格式化器，直接基于VGGT标准输出"""
    try:
        formatted_results = {}
        
        # 相机参数格式化
        if 'cameras' in results:
            extrinsic = results['cameras']['extrinsic']
            intrinsic = results['cameras']['intrinsic']
            
            # 确保是numpy格式用于JSON序列化
            if hasattr(extrinsic, 'cpu'):
                extrinsic_np = extrinsic.cpu().numpy()
            else:
                extrinsic_np = extrinsic
            
            if hasattr(intrinsic, 'cpu'):
                intrinsic_np = intrinsic.cpu().numpy()
            else:
                intrinsic_np = intrinsic
            
            formatted_results['cameras'] = {
                "raw": {
                    "extrinsic": extrinsic,
                    "intrinsic": intrinsic,
                },
                "json": {
                    "extrinsic_matrices": extrinsic_np.tolist(),
                    "intrinsic_matrices": intrinsic_np.tolist(),
                    "format": "opencv_convention",
                }
            }
        
        # 深度结果格式化
        if 'depth' in results and 'depth_conf' in results:
            depth_map = results['depth']
            confidence = results['depth_conf']
            
            formatted_results['depth'] = {
                "raw": {
                    "depth_map": depth_map,
                    "confidence": confidence,
                },
                "json": {
                    "has_depth": True,
                    "has_confidence": True,
                }
            }
        
        # 点云格式化
        if 'world_points' in results:
            points = results['world_points']
            
            formatted_results['points'] = {
                "raw": {
                    "point_map": points,
                    "confidence": results.get('world_points_conf', None),
                },
                "json": {
                    "has_points": True,
                    "point_format": "world_coordinates",
                }
            }
        
        # 深度点云格式化
        if 'points_from_depth' in results:
            formatted_results['points_from_depth'] = {
                "raw": {
                    "points": results['points_from_depth'],
                },
                "json": {
                    "has_points": True,
                    "point_format": "unprojected_from_depth",
                }
            }
        
        # 跟踪结果格式化
        if 'track' in results:
            formatted_results['tracks'] = {
                "raw": {
                    "track_list": results['track'],
                    "visibility_score": results.get('vis', None),
                    "confidence_score": results.get('conf', None),
                },
                "json": {
                    "has_tracks": True,
                }
            }
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Native result formatting failed: {e}")
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# GLB文件生成函数（基于ComfyUI内置功能）
# -----------------------------------------------------------------------------

def save_glb(vertices, faces, filepath, colors=None, metadata=None):
    """
    将顶点和面保存为GLB文件（基于ComfyUI内置功能，增强版支持颜色）
    
    Parameters:
    vertices: numpy.ndarray of shape (N, 3) - 顶点坐标
    faces: numpy.ndarray of shape (M, 3) - 面索引（三角形面）
    filepath: str - 输出文件路径（应该以.glb结尾）
    colors: numpy.ndarray of shape (N, 3) - 顶点颜色（可选）
    metadata: dict - 可选的元数据
    """
    
    try:
        logger.info(f"save_glb: 开始保存GLB文件到 {filepath}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(filepath)
        if not os.path.exists(output_dir):
            logger.info(f"save_glb: 创建输出目录 {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # 确保是numpy数组
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if colors is not None and isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        
        vertices_np = vertices.astype(np.float32)
        faces_np = faces.astype(np.uint32)
        
        logger.info(f"save_glb: 顶点形状 {vertices_np.shape}, 面形状 {faces_np.shape}")
        if colors is not None:
            colors_np = colors.astype(np.float32)
            logger.info(f"save_glb: 颜色形状 {colors_np.shape}")
    
        vertices_buffer = vertices_np.tobytes()
        indices_buffer = faces_np.tobytes()
        
        # 处理颜色数据
        colors_buffer = b''
        if colors is not None:
            colors_buffer = colors_np.tobytes()

        def pad_to_4_bytes(buffer):
            padding_length = (4 - (len(buffer) % 4)) % 4
            return buffer + b'\x00' * padding_length

        vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
        indices_buffer_padded = pad_to_4_bytes(indices_buffer)
        colors_buffer_padded = pad_to_4_bytes(colors_buffer) if colors is not None else b''

        buffer_data = vertices_buffer_padded + colors_buffer_padded + indices_buffer_padded

        vertices_byte_length = len(vertices_buffer)
        vertices_byte_offset = 0
        colors_byte_length = len(colors_buffer)
        colors_byte_offset = len(vertices_buffer_padded)
        indices_byte_length = len(indices_buffer)
        indices_byte_offset = len(vertices_buffer_padded) + len(colors_buffer_padded)

        # 构建buffer views
        buffer_views = [
            {
                "buffer": 0,
                "byteOffset": vertices_byte_offset,
                "byteLength": vertices_byte_length,
                "target": 34962  # ARRAY_BUFFER
            }
        ]
        
        # 添加颜色buffer view
        if colors is not None:
            buffer_views.append({
                "buffer": 0,
                "byteOffset": colors_byte_offset,
                "byteLength": colors_byte_length,
                "target": 34962  # ARRAY_BUFFER
            })
        
        # 添加索引buffer view
        buffer_views.append({
            "buffer": 0,
            "byteOffset": indices_byte_offset,
            "byteLength": indices_byte_length,
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })

        # 构建accessors
        accessors = [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices_np),
                "type": "VEC3",
                "max": vertices_np.max(axis=0).tolist(),
                "min": vertices_np.min(axis=0).tolist()
            }
        ]
        
        # 添加颜色accessor
        if colors is not None:
            accessors.append({
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(colors_np),
                "type": "VEC3",
                "max": colors_np.max(axis=0).tolist(),
                "min": colors_np.min(axis=0).tolist()
            })
        
        # 添加索引accessor
        indices_accessor_index = 2 if colors is not None else 1
        accessors.append({
            "bufferView": indices_accessor_index,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": faces_np.size,
            "type": "SCALAR"
        })

        # 构建mesh attributes
        attributes = {"POSITION": 0}
        if colors is not None:
            attributes["COLOR_0"] = 1

        gltf = {
            "asset": {"version": "2.0", "generator": "ComfyUI-VGGT"},
            "buffers": [
                {
                    "byteLength": len(buffer_data)
                }
            ],
            "bufferViews": buffer_views,
            "accessors": accessors,
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": attributes,
                            "indices": indices_accessor_index,
                            "mode": 4  # TRIANGLES
                        }
                    ]
                }
            ],
            "nodes": [
                {
                    "mesh": 0
                }
            ],
            "scenes": [
                {
                    "nodes": [0]
                }
            ],
            "scene": 0
        }

        if metadata is not None:
            gltf["asset"]["extras"] = metadata

        # Convert the JSON to bytes
        gltf_json = json.dumps(gltf).encode('utf8')

        def pad_json_to_4_bytes(buffer):
            padding_length = (4 - (len(buffer) % 4)) % 4
            return buffer + b' ' * padding_length

        gltf_json_padded = pad_json_to_4_bytes(gltf_json)

        # Create the GLB header
        # Magic glTF
        glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json_padded) + 8 + len(buffer_data))

        # Create JSON chunk header (chunk type 0)
        json_chunk_header = struct.pack('<II', len(gltf_json_padded), 0x4E4F534A)  # "JSON" in little endian

        # Create BIN chunk header (chunk type 1)
        bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # "BIN\0" in little endian

        # Write the GLB file
        logger.info(f"save_glb: 开始写入GLB文件")
        with open(filepath, 'wb') as f:
            f.write(glb_header)
            f.write(json_chunk_header)
            f.write(gltf_json_padded)
            f.write(bin_chunk_header)
            f.write(buffer_data)

        logger.info(f"save_glb: GLB文件写入完成")
        return filepath
    
    except Exception as e:
        logger.error(f"save_glb: 初始化失败 {e}")
        raise e

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def _matrices_to_json(intrinsic, extrinsic, source_type="video") -> (str, str):
    """将相机矩阵转换为 JSON 字符串。"""
    num_views = extrinsic.shape[0]
    intrinsics_list = []
    poses_list = []
    for i in range(num_views):
        K = intrinsic[i].tolist()
        Rt = extrinsic[i].tolist()
        # 相机位置 world 坐标 ( -R^T * t )
        R = np.array(Rt)[:3, :3]
        t = np.array(Rt)[:3, 3]
        position = (-R.T @ t).tolist()
        intrinsics_list.append({
            "view_id": i,  # view_id对应输入序列中的索引
            "intrinsic_matrix": K
        })
        poses_list.append({
            "view_id": i,  # view_id对应输入序列中的索引
            "extrinsic_matrix": Rt,
            "position": position
        })
    
    # 根据源类型生成不同的元数据说明
    if source_type == "images":
        description = "view_id对应输入图片序列中的索引"
        note = "view_id=0对应第一张输入图片，view_id=1对应第二张输入图片，以此类推"
    else:  # video
        description = "view_id对应提取的帧序号，不是原视频的帧索引"
        note = "原版Gradio模式：固定每秒1帧提取，无帧数限制，处理整个视频"
    
    # 相机内参矩阵格式说明
    intrinsic_format_info = {
        "matrix_format": "3x3相机内参矩阵，OpenCV标准格式",
        "matrix_structure": [
            ["fx",  "0",  "cx"],
            ["0",   "fy", "cy"],
            ["0",   "0",  "1"]
        ],
        "parameters": {
            "fx": "X轴焦距（像素单位）- 图像宽度方向的焦距",
            "fy": "Y轴焦距（像素单位）- 图像高度方向的焦距",
            "cx": "主点X坐标（像素单位）- 光轴与图像平面交点的X坐标",
            "cy": "主点Y坐标（像素单位）- 光轴与图像平面交点的Y坐标"
        },
        "coordinate_system": "OpenCV图像坐标系：原点在左上角，X轴向右，Y轴向下",
        "units": "所有参数单位为像素(pixels)"
    }
    
    # 相机外参矩阵格式说明
    extrinsic_format_info = {
        "matrix_format": "3x4相机外参矩阵 [R|t]，世界坐标系到相机坐标系的变换",
        "matrix_structure": [
            ["r11", "r12", "r13", "tx"],
            ["r21", "r22", "r23", "ty"],
            ["r31", "r32", "r33", "tz"]
        ],
        "components": {
            "R": "3x3旋转矩阵 - 世界坐标系到相机坐标系的旋转变换",
            "t": "3x1平移向量 - 相机在世界坐标系中的位置（经过旋转变换）"
        },
        "coordinate_system": {
            "world_frame": "世界坐标系：Z轴向上，X轴向前，Y轴向左（右手坐标系）",
            "camera_frame": "相机坐标系：Z轴向前（光轴方向），X轴向右，Y轴向下"
        },
        "position_calculation": "相机在世界坐标系中的实际位置 = -R^T * t",
        "units": "平移向量单位为米(meters)，旋转矩阵无量纲"
    }
    
    # 添加元数据说明
    intrinsics_data = {
        "cameras": intrinsics_list,
        "metadata": {
            "source_type": source_type,
            "description": description,
            "note": note,
            "total_views": num_views,
            "format_specification": intrinsic_format_info
        }
    }
    
    poses_data = {
        "poses": poses_list,
        "metadata": {
            "source_type": source_type,
            "description": description,
            "note": note,
            "total_views": num_views,
            "format_specification": extrinsic_format_info
        }
    }
    
    return (
        json.dumps(intrinsics_data, ensure_ascii=False, indent=2),
        json.dumps(poses_data, ensure_ascii=False, indent=2),
    )

def _create_traj_preview(extrinsic: torch.Tensor) -> torch.Tensor:
    """根据相机外参创建3D轨迹可视化（使用matplotlib 3D绘图）。"""
    ext = extrinsic.cpu().numpy()  # (N,3,4)
    positions = []
    orientations = []
    
    for mat in ext:
        R = mat[:3, :3]
        t = mat[:3, 3]
        pos = -R.T @ t  # 相机在世界坐标系中的位置
        positions.append(pos)
        # 提取相机朝向（Z轴方向）
        forward = -R[:, 2]  # 相机朝向（Z轴负方向）
        orientations.append(forward)
    
    positions = np.array(positions)
    orientations = np.array(orientations)
    
    # 检查数据有效性
    if len(positions) == 0:
        print("VGGT: 没有有效的相机位姿数据")
        return _create_insufficient_data_image()
    
    print(f"VGGT: 处理 {len(positions)} 个相机位姿")
    
    # 即使只有一个位姿也可以显示
    if len(positions) == 1:
        print("VGGT: 单个相机位姿，将创建简化可视化")

    try:
        # 使用matplotlib创建3D立体可视化
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹线
        if len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=3, alpha=0.8, label='Camera Path')
        
        # 用颜色渐变表示时间进程
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c=colors, s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
        
        # 标记起点和终点
        if len(positions) > 0:
            ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                      c='green', s=150, marker='^', label='Start', edgecolors='darkgreen', linewidth=2)
        if len(positions) > 1:
            ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                      c='red', s=150, marker='o', label='End', edgecolors='darkred', linewidth=2)
        
        # 添加相机方向指示器（每几个位姿显示一个）
        if len(positions) > 0 and len(orientations) > 0:
            # 安全地计算显示步长，最多显示10个箭头
            step = max(1, len(positions) // 10)
            
            # 计算合适的箭头长度
            position_range = positions.max(axis=0) - positions.min(axis=0)
            scene_scale = np.linalg.norm(position_range)
            
            # 更明显的箭头长度计算，确保箭头清晰可见
            if scene_scale < 1e-6:
                direction_length = 0.5  # 极小场景使用更大的默认长度
            else:
                # 使用更明显的比例：8%~20%
                direction_length = scene_scale * 0.15  # 基准 15%
                min_len = scene_scale * 0.08  # 最小8%
                max_len = scene_scale * 0.20   # 最大20%
                direction_length = max(min_len, min(direction_length, max_len))
                
                # 进一步限制最大长度，避免箭头过长
                absolute_max_length = scene_scale * 0.4  # 绝对不超过场景尺度的40%
                direction_length = min(direction_length, absolute_max_length)
            
            # 绘制箭头，确保索引不越界
            arrow_count = 0
            for i in range(0, len(positions), step):
                if i < len(orientations) and arrow_count < 12:  # 减少到最多12个箭头
                    pos = positions[i]
                    direction = orientations[i]
                    
                    # 归一化方向向量，避免异常长度
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        direction_scaled = direction * direction_length
                        
                        # 检查箭头终点是否会超出场景边界
                        arrow_end = pos + direction_scaled
                        scene_min = positions.min(axis=0)
                        scene_max = positions.max(axis=0)
                        
                        # 如果箭头会超出边界，进一步缩短
                        for axis in range(3):
                            if arrow_end[axis] < scene_min[axis] or arrow_end[axis] > scene_max[axis]:
                                direction_scaled *= 0.7  # 缩短30%
                                break
                        
                        ax.quiver(pos[0], pos[1], pos[2], 
                                 direction_scaled[0], direction_scaled[1], direction_scaled[2], 
                                 color='orange', alpha=0.8, arrow_length_ratio=0.2, linewidth=1.5)
                        arrow_count += 1
        
        # 设置坐标轴
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_zlabel('Z (meters)', fontsize=12)
        ax.set_title('VGGT Camera Trajectory (3D View)', fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Time Progress', fontsize=10)
        
        # 设置相等的坐标轴比例
        if len(positions) > 1:
            max_range = np.array([positions.max(axis=0) - positions.min(axis=0)]).max() / 2.0
            mid_x = (positions.max(axis=0)[0] + positions.min(axis=0)[0]) * 0.5
            mid_y = (positions.max(axis=0)[1] + positions.min(axis=0)[1]) * 0.5
            mid_z = (positions.max(axis=0)[2] + positions.min(axis=0)[2]) * 0.5
        else:
            # 单个位姿的情况，设置一个合理的显示范围
            max_range = 2.0  # 默认2米的显示范围
            mid_x, mid_y, mid_z = positions[0]
        
        # 确保范围不为零
        if max_range < 0.1:
            max_range = 1.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        # 保存为图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 读取图像
            img = cv2.imread(tmp.name)
            os.unlink(tmp.name)
            
            if img is not None:
                # BGR转RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为torch tensor
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                print(f"VGGT: 成功创建3D轨迹可视化，图像尺寸: {img.shape}")
                return img_tensor.unsqueeze(0)
            else:
                print("VGGT: 读取生成的可视化图像失败")
                return _create_insufficient_data_image()
    
    except Exception as e:
        print(f"VGGT: 创建3D可视化失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果3D可视化失败，创建备用的2D可视化
        return _create_fallback_2d_visualization_vggt(positions, orientations)

def _create_fallback_2d_visualization_vggt(positions: np.ndarray, orientations: np.ndarray) -> torch.Tensor:
    """创建VGGT备用2D可视化（当3D可视化失败时使用）"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('VGGT Camera Trajectory (2D Views)', fontsize=16, fontweight='bold')
        
        # XY视图（俯视图）
        if len(positions) > 1:
            ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        ax1.scatter(positions[:, 0], positions[:, 1], c=colors, s=50, alpha=0.8, edgecolors='black')
        if len(positions) > 0:
            ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='^', label='Start')
        if len(positions) > 1:
            ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='o', label='End')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Top View (XY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XZ视图（侧视图）
        if len(positions) > 1:
            ax2.plot(positions[:, 0], positions[:, 2], 'g-', linewidth=2, alpha=0.7)
        ax2.scatter(positions[:, 0], positions[:, 2], c=colors, s=50, alpha=0.8, edgecolors='black')
        if len(positions) > 0:
            ax2.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='^')
        if len(positions) > 1:
            ax2.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='o')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Z (meters)')
        ax2.set_title('Side View (XZ)')
        ax2.grid(True, alpha=0.3)
        
        # YZ视图（正视图）
        if len(positions) > 1:
            ax3.plot(positions[:, 1], positions[:, 2], 'r-', linewidth=2, alpha=0.7)
        ax3.scatter(positions[:, 1], positions[:, 2], c=colors, s=50, alpha=0.8, edgecolors='black')
        if len(positions) > 0:
            ax3.scatter(positions[0, 1], positions[0, 2], c='green', s=100, marker='^')
        if len(positions) > 1:
            ax3.scatter(positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o')
        ax3.set_xlabel('Y (meters)')
        ax3.set_ylabel('Z (meters)')
        ax3.set_title('Front View (YZ)')
        ax3.grid(True, alpha=0.3)
        
        # 统计信息面板
        ax4.axis('off')
        stats_text = f"""VGGT Trajectory Statistics
        
Total Poses: {len(positions)}
Position Range:
  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]
  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]
  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]

Path Length: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.3f}m"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存为图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 读取图像
            img = cv2.imread(tmp.name)
            os.unlink(tmp.name)
            
            if img is not None:
                # BGR转RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为torch tensor
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                return img_tensor.unsqueeze(0)
            else:
                return _create_insufficient_data_image()
    
    except Exception as e:
        print(f"VGGT: 创建备用2D可视化也失败: {e}")
        return _create_insufficient_data_image()

def _create_insufficient_data_image():
    """创建数据不足的提示图像"""
    canvas = np.ones((600, 800, 3), dtype=np.float32) * 0.9
    cv2.putText(canvas, "Insufficient Camera Data", (200, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0.3, 0.3, 0.3), 2)
    cv2.putText(canvas, "Need at least 2 frames", (250, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0.5, 0.5, 0.5), 1)
    return torch.from_numpy(canvas).unsqueeze(0)

def _run_vggt_model_inference(image_paths: List[str], model_instance, device) -> Dict:
    """
    运行VGGT模型推理，参考Gradio代码中的run_model函数
    """
    try:
        # 加载并预处理图像
        images = load_and_preprocess_images(image_paths).to(device)
        logger.info(f"预处理图像形状: {images.shape}")
        
        # 确定数据类型
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 模型推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model_instance(images)
        
        # 转换pose encoding为外参和内参矩阵
        logger.info("转换pose encoding为外参和内参矩阵...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # 转换tensors为numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # 移除batch维度
        
        # 从深度图生成世界坐标点
        logger.info("从深度图计算世界坐标点...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        return predictions
        
    except Exception as e:
        logger.error(f"VGGT模型推理失败: {e}")
        raise e

def _generate_3d_model_from_predictions(predictions: Dict, filename_prefix: str = "3d/vggt_model", 
                                      conf_thres: float = 50.0, 
                                      show_cam: bool = True,
                                      mask_black_bg: bool = False,
                                      mask_white_bg: bool = False,
                                      mask_sky: bool = False) -> tuple:
    """
    根据VGGT预测结果生成3D模型文件（GLB格式）
    使用原版predictions_to_glb函数确保原汁原味的输出质量，不降低精度
    
    Returns:
        tuple: (glb_path, ui_result) - 文件路径和UI结果字典
    """
    try:
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            logger.error(f"trimesh不可用: {_TRIMESH_IMPORT_ERROR}")
            return _generate_3d_model_fallback(predictions, filename_prefix, conf_thres, show_cam, mask_black_bg, mask_white_bg, mask_sky)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error(f"matplotlib不可用: {_MATPLOTLIB_IMPORT_ERROR}")
            return _generate_3d_model_fallback(predictions, filename_prefix, conf_thres, show_cam, mask_black_bg, mask_white_bg, mask_sky)
        
        if not SCIPY_AVAILABLE:
            logger.error(f"scipy不可用: {_SCIPY_IMPORT_ERROR}")
            return _generate_3d_model_fallback(predictions, filename_prefix, conf_thres, show_cam, mask_black_bg, mask_white_bg, mask_sky)
        
        # 使用ComfyUI标准路径处理方式
        if FOLDER_PATHS_AVAILABLE:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        else:
            # 备用方案
            full_output_folder = os.path.join("output", "3d")
            os.makedirs(full_output_folder, exist_ok=True)
            filename = "vggt_model"
            counter = 0
            subfolder = "3d"
        
        # 生成文件名（包含参数信息但更简洁）
        param_suffix = f"_conf{conf_thres:.0f}"
        if show_cam:
            param_suffix += "_cam"
        if mask_black_bg:
            param_suffix += "_mbg"
        if mask_white_bg:
            param_suffix += "_mwg"
        if mask_sky:
            param_suffix += "_msky"
            
        glb_filename = f"{filename}_{counter:05}{param_suffix}.glb"
        glb_path = os.path.join(full_output_folder, glb_filename)
        
        logger.info(f"使用原版predictions_to_glb生成3D模型（原汁原味，不降低精度）: {glb_path}")
        
        # 调用原版predictions_to_glb函数 - 使用原汁原味的参数
        scene_3d = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames="all",  # 处理所有帧
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=None,  # 不需要中间文件目录，因为我们在ComfyUI环境中
            prediction_mode="Predicted Pointmap"  # 优先使用Pointmap分支
        )
        
        # 保存trimesh.Scene为GLB文件 - 原滋原味，不降低精度
        logger.info(f"保存原版trimesh.Scene到GLB文件: {glb_path}")
        scene_3d.export(glb_path)
        
        # 验证文件是否真的被创建
        if os.path.exists(glb_path):
            file_size = os.path.getsize(glb_path)
            logger.info(f"成功生成原版质量3D模型: {glb_path}, 文件大小: {file_size} bytes")
            
            # 返回ComfyUI标准格式的结果
            ui_result = {
                "filename": glb_filename,
                "subfolder": subfolder,
                "type": "output"
            }
            
            return glb_path, ui_result
        else:
            logger.error(f"GLB文件保存失败，文件不存在: {glb_path}")
            return "", {}
            
    except Exception as e:
        logger.error(f"使用原版predictions_to_glb生成3D模型失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果原版方法失败，尝试备用方法
        logger.info("尝试使用备用GLB生成方法...")
        return _generate_3d_model_fallback(predictions, filename_prefix, conf_thres, show_cam, mask_black_bg, mask_white_bg, mask_sky)

def _generate_3d_model_fallback(predictions: Dict, filename_prefix: str = "3d/vggt_model", 
                              conf_thres: float = 50.0, 
                              show_cam: bool = True,
                              mask_black_bg: bool = False,
                              mask_white_bg: bool = False,
                              mask_sky: bool = False) -> tuple:
    """
    备用3D模型生成方法（当原版方法失败时使用）
    保持原汁原味的数据，不降低精度
    """
    try:
        # 使用ComfyUI标准路径处理方式
        if FOLDER_PATHS_AVAILABLE:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        else:
            # 备用方案
            full_output_folder = os.path.join("output", "3d")
            os.makedirs(full_output_folder, exist_ok=True)
            filename = "vggt_model"
            counter = 0
            subfolder = "3d"
        
        # 生成文件名
        param_suffix = f"_fallback_conf{conf_thres}"
        glb_filename = f"{filename}_{counter:05}_{param_suffix}.glb"
        glb_path = os.path.join(full_output_folder, glb_filename)
        
        # 获取数据
        if "world_points_from_depth" in predictions:
            world_points = predictions["world_points_from_depth"]  # (S, H, W, 3)
        else:
            logger.warning("world_points_from_depth not found, skipping 3D model generation")
            return "", {}
        
        # 获取图像颜色信息
        images = predictions.get("images", None)
        
        # 简单的点云处理 - 不降低精度，保持原汁原味
        S, H, W = world_points.shape[:3]
        
        # 按照原版逻辑处理
        vertices_3d = world_points.reshape(-1, 3)
        
        # 处理颜色
        if images is not None:
            # 处理图像格式
            if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
                colors_rgb = np.transpose(images, (0, 2, 3, 1))
            else:  # Assume already in NHWC format
                colors_rgb = images
            colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)
        else:
            # 没有颜色信息，生成默认颜色
            colors_rgb = np.ones((len(vertices_3d), 3), dtype=np.uint8) * 128
        
        # 简单的置信度过滤（按照原版逻辑）
        if conf_thres > 0:
            # 使用距离作为置信度的简单替代
            center = np.mean(vertices_3d, axis=0)
            distances = np.linalg.norm(vertices_3d - center, axis=1)
            threshold = np.percentile(distances, conf_thres)
            conf_mask = distances <= threshold
        else:
            conf_mask = np.ones(len(vertices_3d), dtype=bool)
        
        # 应用背景过滤
        if mask_black_bg:
            black_bg_mask = colors_rgb.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask
        
        if mask_white_bg:
            white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
            conf_mask = conf_mask & white_bg_mask
        
        # 应用过滤
        vertices_3d = vertices_3d[conf_mask]
        colors_rgb = colors_rgb[conf_mask]
        
        if len(vertices_3d) == 0:
            vertices_3d = np.array([[1, 0, 0]])
            colors_rgb = np.array([[255, 255, 255]])
        
        # 生成简单的面（每3个点组成一个三角形）
        n_points = len(vertices_3d)
        n_triangles = max(1, n_points // 3)
        faces = []
        for i in range(n_triangles):
            if i*3+2 < n_points:
                faces.append([i*3, i*3+1, i*3+2])
        
        if not faces:
            faces = [[0, 0, 0]]  # 至少一个面
        
        faces = np.array(faces, dtype=np.uint32)
        
        logger.info(f"备用方法生成了 {len(vertices_3d)} 个顶点和 {len(faces)} 个面（保持原始精度）")
        
        # 生成元数据
        metadata = {
            "source": "VGGT-Fallback",
            "confidence_threshold": conf_thres,
            "show_cameras": show_cam,
            "num_vertices": len(vertices_3d),
            "num_faces": len(faces),
            "method": "fallback_original_quality"
        }
        
        # 使用简化的GLB保存方法
        save_glb_simple(vertices_3d, faces, glb_path, metadata)
        
        # 验证文件是否真的被创建
        if os.path.exists(glb_path):
            file_size = os.path.getsize(glb_path)
            logger.info(f"备用方法成功生成3D模型: {glb_path}, 文件大小: {file_size} bytes")
            
            # 返回ComfyUI标准格式的结果
            ui_result = {
                "filename": glb_filename,
                "subfolder": subfolder,
                "type": "output"
            }
            
            return glb_path, ui_result
        else:
            logger.error(f"备用方法GLB文件保存失败")
            return "", {}
            
    except Exception as e:
        logger.error(f"备用3D模型生成方法也失败: {e}")
        return "", {}

# -----------------------------------------------------------------------------
# 主要节点实现
# -----------------------------------------------------------------------------

class VGGTMultiInputNode:
    """VGGT 多输入重建节点 - 支持图片序列输入，实现原生VGGT算法"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vggt_model": ("VVL_VGGT_MODEL", {
                    "tooltip": "来自VVLVGGTLoader的VGGT模型实例，包含已加载的模型和设备信息"
                }),
                "images": ("IMAGE", {
                    "tooltip": "图片序列输入（必需）- 支持1-200帧图像"
                }),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1,
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
    OUTPUT_TOOLTIPS = [
        "原生VGGT推理结果（包含所有Tensor数据）",
        "相机内参数据 (JSON格式)",
        "相机轨迹2D预览图像",
        "相机位姿数据 (JSON格式)",
        "3D模型文件路径 (可连接到Preview3D或其他节点)"
    ]
    OUTPUT_NODE = True
    FUNCTION = "estimate_multi_input"
    CATEGORY = "💃VVL/VGGT"

    def _resolve_video_path(self, video: Any) -> str:
        """解析视频路径"""
        if video is None:
            return ""
        # 如果 video 是字符串
        if isinstance(video, str):
            return video
        # 常见属性
        attrs = ["_VideoFromFile__file", "path", "video_path", "_path", "file_path"]
        for attr in attrs:
            if hasattr(video, attr):
                val = getattr(video, attr)
                if isinstance(val, str):
                    return val
        return ""

    def _extract_video_frames_original(self, video_path: str) -> List[np.ndarray]:
        """按照原版Gradio方式提取视频帧：固定每秒1帧，无帧数限制"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 每秒1帧，与原版Gradio一致
        
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                frames.append(frame.copy())
        
        cap.release()
        logger.info(f"VGGTMultiInputNode: 按原版方式提取了 {len(frames)} 帧 (每秒1帧)")
        return frames

    def estimate_multi_input(self, vggt_model: Dict, 
                           images,
                           confidence_threshold: float = 50.0,
                           show_cameras: bool = True,
                           mask_black_bg: bool = False,
                           mask_white_bg: bool = False,
                           mask_sky: bool = False):
        """
        VGGT多输入重建 - 使用原生VGGT算法
        支持图片序列输入，输出原生结果
        """
        logger.info("开始VGGT原生多输入重建")
        
        # 获取模型实例和设备
        model_instance = vggt_model.get("model")
        device = vggt_model.get("device")
        
        if model_instance is None:
            raise ValueError("无效的VGGT模型实例")
        
        logger.info(f"使用设备: {device}")
        
        # 处理图片序列输入
        image_list = []
        if isinstance(images, torch.Tensor):
            # ComfyUI的IMAGE格式通常是 (batch, height, width, channels)
            images_np = images.cpu().numpy()
            for i in range(images_np.shape[0]):
                img = images_np[i]
                # 确保像素值在0-255范围内
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
        
        # 限制图像数量（避免内存溢出）
        max_frames = 200
        if len(image_list) > max_frames:
            logger.warning(f"图像数量 ({len(image_list)}) 超过最大限制 ({max_frames})，将进行采样")
            # 均匀采样
            indices = np.linspace(0, len(image_list) - 1, max_frames, dtype=int)
            image_list = [image_list[i] for i in indices]
            logger.info(f"采样后图像数量: {len(image_list)}")
        
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
            
            # 运行原生VGGT推理
            raw_results = run_vggt_native_inference(processed_images, model_instance, device)
            
            # 格式化结果
            formatted_results = format_results_native(raw_results)
            
            # 生成JSON输出
            intrinsics_json = ""
            poses_json = ""
            if 'cameras' in formatted_results:
                intrinsics_json = json.dumps(formatted_results['cameras']['json']['intrinsic_matrices'], indent=2)
                poses_json = json.dumps(formatted_results['cameras']['json']['extrinsic_matrices'], indent=2)
            
            # 生成轨迹预览
            trajectory_preview = _create_insufficient_data_image()  # 默认图像
            if 'cameras' in raw_results:
                try:
                    trajectory_preview = _create_traj_preview(raw_results['cameras']['extrinsic'])
                except Exception as e:
                    logger.warning(f"轨迹预览生成失败: {e}")
            
            # 生成3D模型文件
            model_3d_path = ""
            try:
                # 使用官方VGGT的predictions_to_glb函数生成高质量3D模型
                model_3d_path = self._generate_3d_model_official(
                    raw_results, 
                    confidence_threshold, 
                    show_cameras,
                    mask_black_bg,
                    mask_white_bg,
                    mask_sky
                )
            except Exception as e:
                logger.warning(f"3D模型生成失败: {e}")
            
            logger.info("VGGT原生多输入重建完成")
            return (raw_results, intrinsics_json, trajectory_preview, poses_json, model_3d_path)
            
        except Exception as e:
            logger.error(f"VGGT原生推理失败: {e}")
            # 返回错误信息
            error_msg = f"VGGT原生推理失败: {str(e)}"
            error_image = _create_insufficient_data_image()
            empty_result = {}
            return (empty_result, error_msg, error_image, error_msg, "")
    
    def _generate_3d_model_official(self, raw_results: Dict, confidence_threshold: float, show_cameras: bool, 
                                   mask_black_bg: bool = False, mask_white_bg: bool = False, mask_sky: bool = False) -> str:
        """使用官方VGGT的predictions_to_glb函数生成高质量3D模型"""
        try:
            import trimesh
            
            # 创建输出目录
            output_dir = os.path.join(folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output", "3d")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import time
            timestamp = int(time.time())
            model_path = os.path.join(output_dir, f"vggt_official_{timestamp}.glb")
            
            logger.info(f"save_glb: 开始保存GLB文件到 {model_path}")
            
            # 准备官方格式的预测数据 - 关键：使用depth-based points（官方推荐）
            predictions_formatted = {
                'world_points_from_depth': raw_results.get('points_from_depth'),
                'depth_conf': raw_results.get('depth', {}).get('confidence') if 'depth' in raw_results else None,
                'images': raw_results.get('images'),
                'extrinsic': raw_results.get('cameras', {}).get('extrinsic') if 'cameras' in raw_results else None,
            }
            
            # 检查数据是否完整
            missing_fields = [k for k, v in predictions_formatted.items() if v is None]
            if missing_fields:
                logger.warning(f"Missing fields for predictions_to_glb: {missing_fields}")
            
            # 如果没有depth-based points，尝试使用原始点云
            if predictions_formatted['world_points_from_depth'] is None:
                if 'points' in raw_results:
                    logger.info("Using point_map as fallback for world_points")
                    predictions_formatted['world_points'] = raw_results['points']['point_map']
                    predictions_formatted['world_points_conf'] = raw_results['points']['confidence']
            
            # 确保数据格式正确（转换为numpy）
            for key, value in predictions_formatted.items():
                if value is not None and isinstance(value, torch.Tensor):
                    predictions_formatted[key] = value.cpu().numpy()
            
            # 使用官方VGGT的predictions_to_glb函数（优先使用depth-based points，质量更高）
            scene_3d = predictions_to_glb(
                predictions_formatted,
                conf_thres=confidence_threshold,
                filter_by_frames="all",
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cameras,
                mask_sky=mask_sky,
                target_dir=None,
                prediction_mode="Depthmap and Camera Branch"  # 官方推荐模式
            )
            
            # 导出为GLB文件
            scene_3d.export(model_path)
            
            # 获取点云统计信息用于日志
            point_clouds = [geom for geom in scene_3d.geometry.values() if isinstance(geom, trimesh.PointCloud)]
            if point_clouds:
                total_vertices = sum(len(pc.vertices) for pc in point_clouds)
                logger.info(f"save_glb: 顶点形状 ({total_vertices}, 3), 使用官方VGGT高质量处理")
            
            logger.info(f"save_glb: GLB文件写入完成")
            logger.info(f"3D模型已保存到: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"生成3D模型失败: {e}")
            # 回退到简单方法
            return self._generate_3d_model_simple_fallback(raw_results, confidence_threshold)
    
    def _generate_3d_model_simple_fallback(self, raw_results: Dict, confidence_threshold: float) -> str:
        """简单的3D模型生成回退方案"""
        try:
            # 创建输出目录
            output_dir = os.path.join(folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output", "3d")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import time
            timestamp = int(time.time())
            model_path = os.path.join(output_dir, f"vggt_fallback_{timestamp}.glb")
            
            # 从原生结果提取点云数据
            if 'points_from_depth' in raw_results:
                points = raw_results['points_from_depth']
            elif 'points' in raw_results:
                points = raw_results['points']['point_map']
            else:
                logger.warning("No point cloud data found in raw results")
                return ""
            
            # 转换为numpy数组
            if isinstance(points, torch.Tensor):
                points_np = points.cpu().numpy()
            else:
                points_np = points
            
            # 重塑点云数据
            if points_np.ndim == 4:  # (batch, height, width, 3)
                points_np = points_np.reshape(-1, 3)
            elif points_np.ndim == 3:  # (height, width, 3)
                points_np = points_np.reshape(-1, 3)
            
            # 过滤无效点
            valid_mask = ~np.isnan(points_np).any(axis=1)
            points_np = points_np[valid_mask]
            
            if len(points_np) == 0:
                logger.warning("No valid points found")
                return ""
            
            # 简单的点云保存（只保存点，不生成面）
            import trimesh
            point_cloud = trimesh.PointCloud(vertices=points_np)
            point_cloud.export(model_path)
            
            logger.info(f"3D模型已保存到: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"生成3D模型失败: {e}")
            return ""

# -----------------------------------------------------------------------------
# 原版VGGT的predictions_to_glb函数和所有辅助函数（完整移植自visual_util.py）
# -----------------------------------------------------------------------------

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> "trimesh.Scene":
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # Get the shape of pred_world_points_conf to match
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                # Check if mask already exists
                if os.path.exists(mask_filepath):
                    # Load existing mask
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # Generate new mask
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # Resize mask to match H×W if needed
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # Convert list to numpy array with shape S×H×W
            sky_mask_array = np.array(sky_mask_list)

            # Apply sky mask to confidence scores
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: "trimesh.Scene", transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: "trimesh.Scene", extrinsics_matrices: np.ndarray) -> "trimesh.Scene":
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: "trimesh.Trimesh") -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

# -----------------------------------------------------------------------------
# 简化的GLB生成函数（仅作为备用）
# -----------------------------------------------------------------------------

def save_glb_simple(vertices, faces, filepath, metadata=None):
    """
    简化的GLB保存函数（备用）
    """
    try:
        logger.info(f"save_glb_simple: 开始保存GLB文件到 {filepath}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 确保是numpy数组
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        
        vertices_np = vertices.astype(np.float32)
        faces_np = faces.astype(np.uint32)
    
        vertices_buffer = vertices_np.tobytes()
        indices_buffer = faces_np.tobytes()

        def pad_to_4_bytes(buffer):
            padding_length = (4 - (len(buffer) % 4)) % 4
            return buffer + b'\x00' * padding_length

        vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
        indices_buffer_padded = pad_to_4_bytes(indices_buffer)

        buffer_data = vertices_buffer_padded + indices_buffer_padded

        vertices_byte_length = len(vertices_buffer)
        vertices_byte_offset = 0
        indices_byte_length = len(indices_buffer)
        indices_byte_offset = len(vertices_buffer_padded)

        gltf = {
            "asset": {"version": "2.0", "generator": "ComfyUI-VGGT"},
            "buffers": [
                {
                    "byteLength": len(buffer_data)
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": vertices_byte_offset,
                    "byteLength": vertices_byte_length,
                    "target": 34962  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": indices_byte_offset,
                    "byteLength": indices_byte_length,
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(vertices_np),
                    "type": "VEC3",
                    "max": vertices_np.max(axis=0).tolist(),
                    "min": vertices_np.min(axis=0).tolist()
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": faces_np.size,
                    "type": "SCALAR"
                }
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0
                            },
                            "indices": 1,
                            "mode": 4  # TRIANGLES
                        }
                    ]
                }
            ],
            "nodes": [
                {
                    "mesh": 0
                }
            ],
            "scenes": [
                {
                    "nodes": [0]
                }
            ],
            "scene": 0
        }

        if metadata is not None:
            gltf["asset"]["extras"] = metadata

        # Convert the JSON to bytes
        gltf_json = json.dumps(gltf).encode('utf8')

        def pad_json_to_4_bytes(buffer):
            padding_length = (4 - (len(buffer) % 4)) % 4
            return buffer + b' ' * padding_length

        gltf_json_padded = pad_json_to_4_bytes(gltf_json)

        # Create the GLB header
        # Magic glTF
        glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json_padded) + 8 + len(buffer_data))

        # Create JSON chunk header (chunk type 0)
        json_chunk_header = struct.pack('<II', len(gltf_json_padded), 0x4E4F534A)  # "JSON" in little endian

        # Create BIN chunk header (chunk type 1)
        bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # "BIN\0" in little endian

        # Write the GLB file
        with open(filepath, 'wb') as f:
            f.write(glb_header)
            f.write(json_chunk_header)
            f.write(gltf_json_padded)
            f.write(bin_chunk_header)
            f.write(buffer_data)

        logger.info(f"save_glb_simple: GLB文件写入完成")
        return filepath
    
    except Exception as e:
        logger.error(f"save_glb_simple: 保存失败 {e}")
        raise e

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTMultiInputNode": VGGTMultiInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTMultiInputNode": "VVL VGGT Multi Input Camera Estimator",
}

# 如果模型加载器可用，添加到映射中
if MODEL_LOADER_AVAILABLE:
    NODE_CLASS_MAPPINGS["VVLVGGTLoader"] = VVLVGGTLoader
    NODE_DISPLAY_NAME_MAPPINGS["VVLVGGTLoader"] = "VVL VGGT Model Loader" 