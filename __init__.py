"""
ComfyUI VVL Video Camera Advanced
专业的视频相机参数估计工具集
"""

from .comfyui_vggt_nodes import VGGTMultiInputNode
from .vggt_model_loader import VVLVGGTLoader
from .glb_point_cloud_processor import *
from .mask_to_coordinates import MaskToCoordinates
from .json_merger import JSONMerger, UESceneGenerator

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "VVLVGGTLoader": VVLVGGTLoader,
    "VGGTMultiInputNode": VGGTMultiInputNode,   
    "GLBPointCloudProcessor": GLBPointCloudProcessor,
    "GLBPointCloudBounds": GLBPointCloudBounds,
    "GLBPointCloudOriginAdjuster": GLBPointCloudOriginAdjuster,
    "GLBPointCloudDensityFilter": GLBPointCloudDensityFilter,
    "MaskToCoordinates": MaskToCoordinates,
    "JSONMerger": JSONMerger,
    "UESceneGenerator": UESceneGenerator,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "VVLVGGTLoader": "VVL VGGT 模型加载器",
    "VGGTMultiInputNode": "VVL VGGT 多输入重建",
    "GLBPointCloudProcessor": "VVL GLB 点云黑色点清理",
    "GLBPointCloudBounds": "VVL GLB 点云包围盒计算",
    "GLBPointCloudOriginAdjuster": "VVL GLB 点云原点调整",
    "GLBPointCloudDensityFilter": "VVL GLB 点云密度过滤",
    "MaskToCoordinates": "VVL Mask转坐标",
    "JSONMerger": "VVL JSON数据合并器",
    "UESceneGenerator": "VVL UE场景生成器",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "1.0.0"