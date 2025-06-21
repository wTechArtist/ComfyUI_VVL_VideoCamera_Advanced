"""
VGGT Model Loader
独立的VGGT模型加载和管理模块
"""

import os
import logging
from typing import Dict, Optional
import torch
from huggingface_hub import hf_hub_download

# ComfyUI相关导入
try:
    import folder_paths
    import comfy.model_management
    COMFYUI_AVAILABLE = True
except ImportError:
    folder_paths = None
    comfy = None
    COMFYUI_AVAILABLE = False

# VGGT相关导入
try:
    # 确保可以找到vggt模块
    import sys
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from vggt.models.vggt import VGGT
    VGGT_AVAILABLE = True
except Exception as e:
    VGGT = None
    VGGT_AVAILABLE = False
    _VGGT_IMPORT_ERROR = e

# 配置日志
logger = logging.getLogger('vvl_vggt_loader')

# VGGT模型配置
VGGT_MODEL_DIR_NAME = "vggt"
VGGT_MODEL_CONFIG = {
    "VGGT-1B": {
        "model_name": "facebook/VGGT-1B", 
        "description": "VGGT 1B parameter model for camera pose estimation",
        "file_name": "vggt_1b.pt",
        "size_mb": 4700  # 约4.7GB
    },
    # 可以添加更多模型配置
    # "VGGT-Large": {
    #     "model_name": "facebook/VGGT-Large",
    #     "description": "VGGT Large model with better accuracy",
    #     "file_name": "vggt_large.pt"
    # }
}

# 全局模型缓存
_VGGT_MODEL_CACHE = {}

def get_vggt_model_dir() -> str:
    """获取VGGT模型目录路径"""
    if COMFYUI_AVAILABLE and folder_paths:
        # 使用ComfyUI的models目录
        model_dir = os.path.join(folder_paths.models_dir, VGGT_MODEL_DIR_NAME)
    else:
        # 备用路径
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "comfyui", "models", VGGT_MODEL_DIR_NAME)
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def get_vggt_model_path(model_config: dict) -> Optional[str]:
    """获取VGGT模型的本地路径"""
    model_dir = get_vggt_model_dir()
    model_path = os.path.join(model_dir, model_config["file_name"])
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        logger.info(f"Found local VGGT model: {model_path}")
        return model_path
    
    logger.warning(f"Local VGGT model not found: {model_path}")
    logger.info(f"Expected model directory: {model_dir}")
    return None

def list_available_models() -> Dict[str, bool]:
    """列出可用的模型及其本地可用性"""
    available_models = {}
    for model_name, config in VGGT_MODEL_CONFIG.items():
        local_path = get_vggt_model_path(config)
        available_models[model_name] = local_path is not None
    return available_models

def load_vggt_model(model_name: str, device: torch.device) -> Optional[torch.nn.Module]:
    """加载VGGT模型"""
    if not VGGT_AVAILABLE:
        logger.error(f"VGGT not available: {_VGGT_IMPORT_ERROR}")
        return None
    
    try:
        # 检查缓存
        cache_key = f"{model_name}_{device}"
        if cache_key in _VGGT_MODEL_CACHE:
            logger.info(f"Using cached VGGT model: {model_name}")
            return _VGGT_MODEL_CACHE[cache_key]
        
        if model_name not in VGGT_MODEL_CONFIG:
            raise ValueError(f"Unknown VGGT model: {model_name}. Available: {list(VGGT_MODEL_CONFIG.keys())}")
        
        config = VGGT_MODEL_CONFIG[model_name]
        local_path = get_vggt_model_path(config)
        
        # ----------------------------
        # 1) 若本地已有权重，则直接加载
        # 2) 若无，则从 HuggingFace Hub 下载到同目录，再加载
        # ----------------------------
        if local_path and os.path.exists(local_path):
            logger.info(f"Loading VGGT model from local path: {local_path}")
        else:
            logger.info(f"Local weight not found, downloading from HuggingFace Hub: {config['model_name']}")
            try:
                local_path = hf_hub_download(
                    repo_id=config["model_name"],
                    filename="model.pt",
                    local_dir=get_vggt_model_dir(),
                    cache_dir=get_vggt_model_dir(),
                    resume_download=True,
                )
                logger.info(f"Model downloaded to: {local_path}")
            except Exception as e:
                logger.error(f"Failed to download VGGT model: {e}")
                raise

        # 统一使用本地权重文件加载
        model = VGGT()
        state_dict = torch.load(local_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        # 移动到指定设备并设置为评估模式
        model = model.to(device)
        model.eval()
        
        # 缓存模型
        _VGGT_MODEL_CACHE[cache_key] = model
        
        logger.info(f"VGGT model {model_name} loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load VGGT model {model_name}: {e}")
        return None

def clear_model_cache():
    """清除模型缓存"""
    global _VGGT_MODEL_CACHE
    for key in list(_VGGT_MODEL_CACHE.keys()):
        del _VGGT_MODEL_CACHE[key]
    logger.info("VGGT model cache cleared")

def get_model_info() -> Dict:
    """获取模型信息"""
    return {
        "available": VGGT_AVAILABLE,
        "error": _VGGT_IMPORT_ERROR if not VGGT_AVAILABLE else None,
        "model_dir": get_vggt_model_dir(),
        "cached_models": list(_VGGT_MODEL_CACHE.keys()),
        "config": VGGT_MODEL_CONFIG,
        "local_models": list_available_models()
    }

class VVLVGGTLoader:
    """VGGT模型加载器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        vggt_models = list(VGGT_MODEL_CONFIG.keys())
        device_list = ["auto", "cuda", "cpu"]
        
        return {
            "required": {
                "device": (device_list, {
                    "default": "auto",
                    "tooltip": "选择运行设备。auto会自动选择CUDA（如果可用）或CPU"
                }),
                "vggt_model": (vggt_models, {
                    "default": vggt_models[0] if vggt_models else "VGGT-1B",
                    "tooltip": "选择VGGT模型版本。VGGT-1B是标准的10亿参数模型，约4.7GB"
                }),
            }
        }

    RETURN_TYPES = ("VVL_VGGT_MODEL",)
    RETURN_NAMES = ("vggt_model",)
    FUNCTION = "load_vggt_model"
    CATEGORY = "💃VVL/VGGT"

    def load_vggt_model(self, device: str, vggt_model: str):
        """加载VGGT模型并返回模型实例"""
        
        # 确定设备
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)
        
        logger.info(f"VVLVGGTLoader: Loading VGGT model: {vggt_model} on device: {torch_device}")
        
        # 检查VGGT是否可用
        if not VGGT_AVAILABLE:
            error_msg = f"VGGT not available: {_VGGT_IMPORT_ERROR}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 显示模型信息
        model_info = get_model_info()
        logger.info(f"Model directory: {model_info['model_dir']}")
        logger.info(f"Local models: {model_info['local_models']}")
        
        # 加载模型
        model_instance = load_vggt_model(vggt_model, torch_device)
        
        if model_instance is None:
            raise RuntimeError(f"Failed to load VGGT model: {vggt_model}")
        
        # 创建包含模型和相关信息的字典
        model_data = {
            'model': model_instance,
            'device': torch_device,
            'model_name': vggt_model,
            'config': VGGT_MODEL_CONFIG[vggt_model],
            'loader_info': model_info
        }
        
        logger.info("VVLVGGTLoader: VGGT model loaded successfully")
        return (model_data,) 