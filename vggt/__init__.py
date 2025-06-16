"""
VGGT (Visual Geometry Grounded Transformer) Package
Facebook Research
"""

__version__ = "1.0.0"
__author__ = "Facebook Research"

# 导入核心模块，确保可以通过 from vggt.models.vggt import VGGT 访问
try:
    from .models.vggt import VGGT
    from .models.aggregator import Aggregator
    
    # 导入头部模块
    from .heads.camera_head import CameraHead
    from .heads.dpt_head import DPTHead  
    from .heads.track_head import TrackHead
    
    # 导入工具模块
    from . import utils
    from . import layers
    
    __all__ = [
        'VGGT', 
        'Aggregator',
        'CameraHead', 
        'DPTHead', 
        'TrackHead',
        'utils',
        'layers'
    ]

except ImportError as e:
    # 如果导入失败，仍然允许模块被导入，但会在使用时报错
    import warnings
    warnings.warn(f"VGGT模块导入部分失败: {e}")
    
    # 定义空的类以避免导入错误
    class VGGT:
        def __init__(self, *args, **kwargs):
            raise ImportError("VGGT模型导入失败，请检查依赖")
    
    __all__ = ['VGGT'] 