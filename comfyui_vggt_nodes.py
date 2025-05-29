import os
import json
import tempfile
from typing import List, Any, Dict
import logging

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

# 导入 VGGT 相关函数
try:
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    VGGT_UTILS_AVAILABLE = True
except Exception as e:
    load_and_preprocess_images = None
    pose_encoding_to_extri_intri = None
    VGGT_UTILS_AVAILABLE = False
    _VGGT_UTILS_IMPORT_ERROR = e

# 导入模型加载器
try:
    from .vggt_model_loader import VVLVGGTLoader
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    VVLVGGTLoader = None
    MODEL_LOADER_AVAILABLE = False

# 配置日志
logger = logging.getLogger('vvl_vggt_nodes')

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def _extract_video_frames(video_path: str, interval: int, max_frames: int) -> List[np.ndarray]:
    """按照给定间隔提取视频帧 (BGR)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame.copy())
        idx += 1
    cap.release()
    return frames

def _matrices_to_json(intrinsic, extrinsic) -> (str, str):
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
            "view_id": i,
            "intrinsic_matrix": K
        })
        poses_list.append({
            "view_id": i,
            "extrinsic_matrix": Rt,
            "position": position
        })
    return (
        json.dumps({"cameras": intrinsics_list}, ensure_ascii=False, indent=2),
        json.dumps({"poses": poses_list}, ensure_ascii=False, indent=2),
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

# -----------------------------------------------------------------------------
# 主要节点实现
# -----------------------------------------------------------------------------

class VGGTVideoCameraNode:
    """VGGT 视频相机参数估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vggt_model": ("VVL_VGGT_MODEL", {
                    "tooltip": "来自VVLVGGTLoader的VGGT模型实例，包含已加载的模型和设备信息"
                }),
                "video": (IO.VIDEO, {
                    "tooltip": "来自 LoadVideo 的视频对象，或直接输入视频文件路径"
                }),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "备用视频路径，当video输入为空时使用"
                }),
                "frame_interval": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "帧提取间隔，数值越小提取的帧越密集，但计算量更大"
                }),
                "max_frames": ("INT", {
                    "default": 60, "min": 5, "max": 200, "step": 5,
                    "tooltip": "最大提取帧数，用于控制计算量和内存使用"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("intrinsics_json", "trajectory_preview", "poses_json")
    FUNCTION = "estimate"
    CATEGORY = "💃VVL/VGGT"

    # ---------------------------------------------------------
    def _resolve_video_path(self, video: Any, fallback: str) -> str:
        """解析视频路径"""
        if video is None:
            return fallback
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
        return fallback

    # ---------------------------------------------------------
    def estimate(self, vggt_model: Dict, video=None, video_path: str = "", 
                frame_interval: int = 5, max_frames: int = 60):
        """执行相机参数估计"""
        try:
            # 检查VGGT工具函数是否可用
            if not VGGT_UTILS_AVAILABLE:
                raise RuntimeError(f"VGGT utils not available: {_VGGT_UTILS_IMPORT_ERROR}")
            
            # 从模型字典中获取信息
            model_instance = vggt_model['model']
            device = vggt_model['device']
            model_name = vggt_model['model_name']
            
            logger.info(f"VGGTVideoCameraNode: Using {model_name} on {device}")
            
            # 确定数据类型
            if device.type == "cuda":
                try:
                    # 尝试使用BFloat16，如果不支持则fallback到Float16
                    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                except:
                    dtype = torch.float16
            else:
                dtype = torch.float32

            # 解析视频路径
            vid_path = self._resolve_video_path(video, video_path)
            if not vid_path or not os.path.exists(vid_path):
                raise FileNotFoundError(f"找不到视频文件: {vid_path}")

            logger.info(f"VGGTVideoCameraNode: Processing video: {vid_path}")

            # 提取视频帧
            frames = _extract_video_frames(vid_path, frame_interval, max_frames)
            if not frames:
                raise RuntimeError("无法从视频中提取帧")

            logger.info(f"VGGTVideoCameraNode: Extracted {len(frames)} frames")

            # 将帧保存为 PNG 以复用官方预处理
            with tempfile.TemporaryDirectory() as tmpdir:
                img_paths = []
                for i, frm in enumerate(frames):
                    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    p = os.path.join(tmpdir, f"frame_{i:04d}.png")
                    # 使用PIL保存RGB图片，避免cv2的BGR问题
                    Image.fromarray(rgb).save(p)
                    img_paths.append(p)

                # 加载并预处理图像
                imgs = load_and_preprocess_images(img_paths).to(device)
                logger.info(f"VGGTVideoCameraNode: Preprocessed images shape: {imgs.shape}")

                # 模型推理
                with torch.no_grad():
                    try:
                        with torch.amp.autocast(device_type=device.type, dtype=dtype):
                            predictions = model_instance(imgs)
                    except:
                        # Fallback方案
                        try:
                            if device.type == "cuda":
                                with torch.cuda.amp.autocast(dtype=dtype):
                                    predictions = model_instance(imgs)
                            else:
                                predictions = model_instance(imgs)
                        except:
                            # 最后的fallback
                            predictions = model_instance(imgs)
                    
                    # 从predictions中提取pose_enc
                    pose_enc = predictions["pose_enc"]
                    logger.info(f"VGGTVideoCameraNode: pose_enc shape: {pose_enc.shape}")
                            
                # 转换为内外参矩阵
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
                
                # 去除批次维度
                if len(extrinsic.shape) == 4:  # (1,N,3,4)
                    extrinsic = extrinsic[0]   # (N,3,4)
                if len(intrinsic.shape) == 4:  # (1,N,3,3)
                    intrinsic = intrinsic[0]   # (N,3,3)
                    
                extrinsic = extrinsic.cpu()
                intrinsic = intrinsic.cpu()
                
                logger.info(f"VGGTVideoCameraNode: Final matrix shapes - "
                          f"extrinsic: {extrinsic.shape}, intrinsic: {intrinsic.shape}")

            # 生成JSON输出
            intrinsics_json, poses_json = _matrices_to_json(intrinsic.numpy(), extrinsic.numpy())

            # 生成轨迹预览图
            traj_tensor = _create_traj_preview(extrinsic)

            logger.info("VGGTVideoCameraNode: Camera estimation completed successfully")
            return (intrinsics_json, traj_tensor, poses_json)

        except Exception as e:
            error_msg = f"VGGT估计错误: {str(e)}"
            logger.error(error_msg)
            
            # 返回错误结果
            empty_img = torch.ones((1, 400, 400, 3), dtype=torch.float32) * 0.1
            error_json = json.dumps({"success": False, "error": error_msg}, ensure_ascii=False, indent=2)
            return (error_json, empty_img, error_json)

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTVideoCameraNode": VGGTVideoCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTVideoCameraNode": "VVL VGGT Video Camera Estimator",
}

# 如果模型加载器可用，添加到映射中
if MODEL_LOADER_AVAILABLE:
    NODE_CLASS_MAPPINGS["VVLVGGTLoader"] = VVLVGGTLoader
    NODE_DISPLAY_NAME_MAPPINGS["VVLVGGTLoader"] = "VVL VGGT Model Loader" 