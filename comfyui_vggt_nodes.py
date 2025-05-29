import os
import json
import tempfile
from typing import List, Any

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
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except Exception as e:
    VGGT = None
    _VGGT_IMPORT_ERROR = e
else:
    _VGGT_IMPORT_ERROR = None

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
    """根据相机外参创建3D轨迹可视化 (返回 1xHxWxC)。"""
    ext = extrinsic.cpu().numpy()  # (N,3,4)
    positions = []
    orientations = []
    
    for mat in ext:
        R = mat[:3, :3]
        t = mat[:3, 3]
        pos = -R.T @ t
        positions.append(pos)
        # 提取相机朝向（Z轴方向）
        forward = R[:, 2]  # 相机朝向
        orientations.append(forward)
    
    positions = np.array(positions)
    orientations = np.array(orientations)
    
    if positions.shape[0] < 2:
        # 少于两帧给提示图
        return _create_insufficient_data_image()

    # 创建组合视图：顶视图 + 侧视图 + 3D投影
    canvas_width, canvas_height = 800, 600
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.float32) * 0.95
    
    # 绘制标题
    cv2.putText(canvas, "VGGT Camera Trajectory Analysis", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0.1, 0.1, 0.1), 2)
    
    # 1. 顶视图 (Top View) - 左上角
    top_view = _create_top_view(positions, orientations, size=(350, 250))
    canvas[60:310, 20:370] = top_view
    cv2.putText(canvas, "Top View (X-Z)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.2, 0.2, 0.2), 1)
    
    # 2. 侧视图 (Side View) - 右上角  
    side_view = _create_side_view(positions, orientations, size=(350, 250))
    canvas[60:310, 420:770] = side_view
    cv2.putText(canvas, "Side View (X-Y)", (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.2, 0.2, 0.2), 1)
    
    # 3. 伪3D视图 - 下方
    pseudo_3d_view = _create_pseudo_3d_view(positions, orientations, size=(720, 250))
    canvas[330:580, 40:760] = pseudo_3d_view
    cv2.putText(canvas, "3D Perspective View", (40, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.2, 0.2, 0.2), 1)
    
    # 添加统计信息
    _add_trajectory_stats(canvas, positions, orientations)
    
    return torch.from_numpy(canvas).unsqueeze(0)

def _create_insufficient_data_image():
    """创建数据不足的提示图像"""
    canvas = np.ones((600, 800, 3), dtype=np.float32) * 0.9
    cv2.putText(canvas, "Insufficient Camera Data", (200, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0.3, 0.3, 0.3), 2)
    cv2.putText(canvas, "Need at least 2 frames", (250, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0.5, 0.5, 0.5), 1)
    return torch.from_numpy(canvas).unsqueeze(0)

def _create_top_view(positions, orientations, size=(350, 250)):
    """创建顶视图 (X-Z平面)"""
    canvas = np.ones((size[1], size[0], 3), dtype=np.float32) * 0.98
    
    # 计算投影范围
    x, z = positions[:, 0], positions[:, 2]
    x_range, z_range = _get_padded_range(x), _get_padded_range(z)
    
    # 绘制网格
    _draw_grid(canvas, size, color=(0.9, 0.9, 0.9))
    
    # 转换坐标
    pts_2d = _normalize_coords(np.column_stack([x, z]), x_range, z_range, size, margin=20)
    
    # 绘制轨迹线
    _draw_trajectory_line(canvas, pts_2d, color=(0.2, 0.4, 0.8), thickness=2)
    
    # 绘制相机位置和朝向
    for i, (pt, orient) in enumerate(zip(pts_2d, orientations)):
        # 相机位置
        color = _get_camera_color(i, len(pts_2d))
        cv2.circle(canvas, tuple(pt.astype(int)), 4, color, -1)
        
        # 朝向指示器 (X-Z投影)
        orient_2d = np.array([orient[0], orient[2]])  # X-Z投影
        orient_2d = orient_2d / (np.linalg.norm(orient_2d) + 1e-8) * 15
        end_pt = pt + orient_2d
        cv2.arrowedLine(canvas, tuple(pt.astype(int)), tuple(end_pt.astype(int)), 
                       color, 1, tipLength=0.3)
        
        # 帧编号
        cv2.putText(canvas, str(i), (int(pt[0]+6), int(pt[1]+6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    
    return canvas

def _create_side_view(positions, orientations, size=(350, 250)):
    """创建侧视图 (X-Y平面)"""
    canvas = np.ones((size[1], size[0], 3), dtype=np.float32) * 0.98
    
    # 计算投影范围
    x, y = positions[:, 0], positions[:, 1]
    x_range, y_range = _get_padded_range(x), _get_padded_range(y)
    
    # 绘制网格
    _draw_grid(canvas, size, color=(0.9, 0.9, 0.9))
    
    # 转换坐标 (注意Y轴翻转)
    pts_2d = _normalize_coords(np.column_stack([x, -y]), x_range, (-y_range[1], -y_range[0]), size, margin=20)
    
    # 绘制轨迹线
    _draw_trajectory_line(canvas, pts_2d, color=(0.8, 0.4, 0.2), thickness=2)
    
    # 绘制相机位置
    for i, pt in enumerate(pts_2d):
        color = _get_camera_color(i, len(pts_2d))
        cv2.circle(canvas, tuple(pt.astype(int)), 4, color, -1)
        cv2.putText(canvas, str(i), (int(pt[0]+6), int(pt[1]+6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    
    return canvas

def _create_pseudo_3d_view(positions, orientations, size=(720, 250)):
    """创建伪3D视图"""
    canvas = np.ones((size[1], size[0], 3), dtype=np.float32) * 0.98
    
    # 应用3D->2D投影变换 (等距投影)
    # 旋转角度
    angle_x, angle_y = np.pi/6, np.pi/4  # 30度和45度
    
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    
    # 3D到2D投影矩阵
    proj_matrix = np.array([
        [cos_y, sin_x*sin_y, 0],
        [0, cos_x, sin_x]
    ])
    
    # 投影3D点
    pts_3d = positions.T  # (3, N)
    pts_2d = proj_matrix @ pts_3d  # (2, N)
    pts_2d = pts_2d.T  # (N, 2)
    
    # 归一化到画布
    x_range = _get_padded_range(pts_2d[:, 0])
    y_range = _get_padded_range(pts_2d[:, 1])
    pts_canvas = _normalize_coords(pts_2d, x_range, y_range, size, margin=30)
    
    # 绘制3D网格效果
    _draw_3d_grid(canvas, size)
    
    # 根据深度排序绘制
    depths = positions[:, 2]  # Z坐标作为深度
    depth_order = np.argsort(depths)
    
    # 绘制轨迹线（带深度渐变）
    for i in range(len(depth_order)-1):
        idx1, idx2 = depth_order[i], depth_order[i+1]
        if abs(idx1 - idx2) == 1:  # 连续帧
            pt1, pt2 = pts_canvas[idx1], pts_canvas[idx2]
            # 深度渐变色
            depth_ratio = (depths[idx1] - depths.min()) / (depths.max() - depths.min() + 1e-8)
            color = (0.2 + 0.6*depth_ratio, 0.4, 0.8 - 0.4*depth_ratio)
            cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, 2)
    
    # 绘制相机（带深度大小变化）
    for i in depth_order:
        pt = pts_canvas[i]
        depth_ratio = (depths[i] - depths.min()) / (depths.max() - depths.min() + 1e-8)
        radius = int(3 + 4 * depth_ratio)  # 近大远小
        color = _get_camera_color(i, len(positions))
        
        # 相机主体
        cv2.circle(canvas, tuple(pt.astype(int)), radius, color, -1)
        cv2.circle(canvas, tuple(pt.astype(int)), radius+1, (0,0,0), 1)
        
        # 朝向指示（3D投影）
        orient_3d = orientations[i]
        orient_2d = proj_matrix @ orient_3d
        orient_2d = orient_2d / (np.linalg.norm(orient_2d) + 1e-8) * (10 + 5*depth_ratio)
        end_pt = pt + orient_2d
        cv2.arrowedLine(canvas, tuple(pt.astype(int)), tuple(end_pt.astype(int)), 
                       color, max(1, int(depth_ratio*2+1)), tipLength=0.3)
        
        # 帧编号
        cv2.putText(canvas, str(i), (int(pt[0]+8), int(pt[1]+8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    return canvas

def _get_padded_range(coords, padding=0.1):
    """获取带边距的坐标范围"""
    min_val, max_val = coords.min(), coords.max()
    span = max_val - min_val
    if span < 1e-8:
        span = 1.0
    margin = span * padding
    return (min_val - margin, max_val + margin)

def _normalize_coords(coords, x_range, y_range, size, margin=20):
    """归一化坐标到画布"""
    x_norm = (coords[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (coords[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    
    canvas_coords = np.column_stack([
        x_norm * (size[0] - 2*margin) + margin,
        (1 - y_norm) * (size[1] - 2*margin) + margin  # 翻转Y轴
    ])
    return canvas_coords

def _draw_grid(canvas, size, color=(0.9, 0.9, 0.9)):
    """绘制网格"""
    h, w = size[1], size[0]
    # 垂直线
    for x in range(0, w, w//8):
        cv2.line(canvas, (x, 0), (x, h), color, 1)
    # 水平线
    for y in range(0, h, h//6):
        cv2.line(canvas, (0, y), (w, y), color, 1)

def _draw_3d_grid(canvas, size):
    """绘制3D网格效果"""
    h, w = size[1], size[0]
    color = (0.85, 0.85, 0.85)
    
    # 斜向网格线
    for i in range(0, w, w//12):
        cv2.line(canvas, (i, 0), (i + h//3, h), color, 1)
    for i in range(0, h, h//8):
        cv2.line(canvas, (0, i), (w, i), color, 1)

def _draw_trajectory_line(canvas, points, color, thickness=2):
    """绘制轨迹线"""
    for i in range(len(points)-1):
        cv2.line(canvas, tuple(points[i].astype(int)), 
                tuple(points[i+1].astype(int)), color, thickness)

def _get_camera_color(index, total):
    """获取相机颜色（彩虹渐变）"""
    if total <= 1:
        return (0.5, 0.5, 0.5)
    
    ratio = index / (total - 1)
    if ratio < 0.2:
        return (0.0, 0.8, 0.0)  # 绿色起点
    elif ratio > 0.8:
        return (0.8, 0.0, 0.0)  # 红色终点
    else:
        # 蓝色中间段
        blue_intensity = 0.3 + 0.5 * np.sin(ratio * np.pi)
        return (0.0, 0.2, blue_intensity)

def _add_trajectory_stats(canvas, positions, orientations):
    """添加轨迹统计信息"""
    # 计算统计数据
    total_distance = 0
    for i in range(1, len(positions)):
        total_distance += np.linalg.norm(positions[i] - positions[i-1])
    
    max_height = positions[:, 1].max()
    min_height = positions[:, 1].min()
    height_range = max_height - min_height
    
    # 显示统计信息
    stats_x, stats_y = 40, 590
    font_scale, thickness = 0.45, 1
    color = (0.1, 0.1, 0.1)
    
    stats_text = [
        f"Frames: {len(positions)}",
        f"Distance: {total_distance:.2f}m",
        f"Height Range: {height_range:.2f}m",
        f"Avg Height: {positions[:, 1].mean():.2f}m"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(canvas, text, (stats_x + i*150, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# -----------------------------------------------------------------------------
# 节点实现
# -----------------------------------------------------------------------------

_VGGT_MODEL = None  # 全局缓存模型

class VGGTVideoCameraNode:
    """VGGT 视频相机参数估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {
                    "tooltip": "来自 LoadVideo 的视频对象，或直接输入视频文件路径"
                }),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "备用视频路径"
                }),
                "frame_interval": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1
                }),
                "max_frames": ("INT", {
                    "default": 60, "min": 5, "max": 200, "step": 5
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("intrinsics_json", "trajectory_preview", "poses_json")
    FUNCTION = "estimate"
    CATEGORY = "VGGT"

    def __init__(self):
        global _VGGT_MODEL
        if VGGT is None:
            print(f"[VGGT node] 导入 VGGT 失败: {_VGGT_IMPORT_ERROR}")
            _VGGT_MODEL = None
            return
        if _VGGT_MODEL is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                _VGGT_MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
                _VGGT_MODEL.eval()
                print("[VGGT node] VGGT 模型加载完成")
            except Exception as e:
                print(f"[VGGT node] 加载模型失败: {e}")
                _VGGT_MODEL = None

    # ---------------------------------------------------------
    def _resolve_video_path(self, video: Any, fallback: str) -> str:
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
    def estimate(self, video=None, video_path: str = "", frame_interval: int = 5, max_frames: int = 60, device: str = "auto"):
        try:
            if _VGGT_MODEL is None:
                raise RuntimeError("VGGT 模型未加载，无法推理")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 更兼容的dtype选择，避免BFloat16不支持问题
            if device == "cuda":
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

            frames = _extract_video_frames(vid_path, frame_interval, max_frames)
            if not frames:
                raise RuntimeError("无法从视频中提取帧")

            # 将帧保存为 PNG 以复用官方预处理
            with tempfile.TemporaryDirectory() as tmpdir:
                img_paths = []
                for i, frm in enumerate(frames):
                    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    p = os.path.join(tmpdir, f"frame_{i:04d}.png")
                    # 使用PIL保存RGB图片，避免cv2的BGR问题
                    Image.fromarray(rgb).save(p)
                    img_paths.append(p)

                imgs = load_and_preprocess_images(img_paths).to(device)
                print(f"[VGGT node] 加载图片数量: {len(img_paths)}, 处理后形状: {imgs.shape}")

                with torch.no_grad():
                    # 使用与demo_gradio.py相同的调用方式：直接调用model()
                    try:
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            predictions = _VGGT_MODEL(imgs)  # 直接调用整个模型
                    except:
                        # Fallback to old API
                        try:
                            with torch.cuda.amp.autocast(dtype=dtype):
                                predictions = _VGGT_MODEL(imgs)  # 直接调用整个模型
                        except:
                            # 如果autocast有问题，直接运行
                            predictions = _VGGT_MODEL(imgs)  # 直接调用整个模型
                    
                    # 从predictions中提取pose_enc
                    pose_enc = predictions["pose_enc"]
                    print(f"[VGGT node] pose_enc形状: {pose_enc.shape}")
                            
                # extrinsic (1,N,3,4), intrinsic (1,N,3,3)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
                
                # 如果有批次维度则去除
                if len(extrinsic.shape) == 4:  # (1,N,3,4)
                    extrinsic = extrinsic[0]   # (N,3,4)
                if len(intrinsic.shape) == 4:  # (1,N,3,3)
                    intrinsic = intrinsic[0]   # (N,3,3)
                    
                extrinsic = extrinsic.cpu()
                intrinsic = intrinsic.cpu()
                print(f"[VGGT node] 最终矩阵形状 - extrinsic: {extrinsic.shape}, intrinsic: {intrinsic.shape}")

            # JSON 输出
            intrinsics_json, poses_json = _matrices_to_json(intrinsic.numpy(), extrinsic.numpy())

            # 轨迹图像
            traj_tensor = _create_traj_preview(extrinsic)

            return (intrinsics_json, traj_tensor, poses_json)

        except Exception as e:
            err = f"VGGT 估计错误: {e}"
            print(err)
            empty_img = torch.ones((1, 400, 400, 3), dtype=torch.float32) * 0.1
            err_json = json.dumps({"success": False, "error": err}, ensure_ascii=False, indent=2)
            return (err_json, empty_img, err_json)

# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VGGTVideoCameraNode": VGGTVideoCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTVideoCameraNode": "VGGT Video Camera Estimator",
} 