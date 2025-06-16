#!/usr/bin/env python3
"""
VGGT ComfyUI集成安装脚本
自动安装VGGT专用依赖并设置模型目录
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_vggt_dependencies():
    """安装VGGT专用依赖"""
    logger.info("开始安装VGGT专用依赖...")
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements_vggt.txt"
    
    if not requirements_file.exists():
        logger.error(f"依赖文件不存在: {requirements_file}")
        return False
    
    try:
        # 安装依赖
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("VGGT专用依赖安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"依赖安装失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def setup_model_directory():
    """设置模型目录"""
    logger.info("设置VGGT模型目录...")
    
    try:
        # 尝试导入ComfyUI的folder_paths
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
        except ImportError:
            # 备用方案：使用相对路径
            models_dir = Path(__file__).parent.parent.parent / "models"
        
        # 创建VGGT模型目录
        vggt_model_dir = Path(models_dir) / "vggt"
        vggt_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VGGT模型目录已创建: {vggt_model_dir}")
        
        # 创建README文件
        readme_content = """# VGGT Models Directory

This directory stores VGGT model weights.

## Supported Models:
- VGGT-1B: facebook/VGGT-1B (约4.7GB)

## Usage:
Models will be automatically downloaded from HuggingFace Hub when first used.
You can also manually place model files here:
- vggt_1b.pt (for VGGT-1B)

## Model Sources:
- Official Repository: https://github.com/facebookresearch/vggt
- HuggingFace Hub: https://huggingface.co/facebook/VGGT-1B
"""
        
        readme_file = vggt_model_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return True
        
    except Exception as e:
        logger.error(f"模型目录设置失败: {e}")
        return False

def verify_installation():
    """验证安装是否成功"""
    logger.info("验证VGGT安装...")
    
    try:
        # 检查核心依赖
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        # 检查VGGT相关依赖
        dependencies = [
            'trimesh',
            'scipy',
            'matplotlib',
            'numpy',
            'PIL',
            'cv2'
        ]
        
        missing_deps = []
        for dep in dependencies:
            try:
                if dep == 'PIL':
                    import PIL
                elif dep == 'cv2':
                    import cv2
                else:
                    __import__(dep)
                logger.info(f"✓ {dep} 可用")
            except ImportError:
                missing_deps.append(dep)
                logger.warning(f"✗ {dep} 不可用")
        
        # 检查VGGT原生模块
        try:
            from vggt.models.vggt import VGGT
            logger.info("✓ VGGT原生模块可用")
        except ImportError as e:
            logger.warning(f"✗ VGGT原生模块不可用: {e}")
            missing_deps.append('vggt')
        
        # 检查可选依赖
        optional_deps = ['flash_attn', 'gsplat', 'pytorch3d']
        for dep in optional_deps:
            try:
                __import__(dep)
                logger.info(f"✓ {dep} (可选) 可用")
            except ImportError:
                logger.info(f"○ {dep} (可选) 不可用")
        
        if missing_deps:
            logger.warning(f"缺少依赖: {missing_deps}")
            return False
        else:
            logger.info("✓ 所有核心依赖都可用")
            return True
            
    except Exception as e:
        logger.error(f"安装验证失败: {e}")
        return False

def main():
    """主安装流程"""
    logger.info("=" * 50)
    logger.info("VGGT ComfyUI集成安装程序")
    logger.info("=" * 50)
    
    success = True
    
    # 1. 安装依赖
    if not install_vggt_dependencies():
        success = False
    
    # 2. 设置模型目录
    if not setup_model_directory():
        success = False
    
    # 3. 验证安装
    if not verify_installation():
        success = False
    
    # 输出结果
    logger.info("=" * 50)
    if success:
        logger.info("✓ VGGT集成安装完成!")
        logger.info("现在可以在ComfyUI中使用VGGT节点了")
    else:
        logger.error("✗ VGGT集成安装失败")
        logger.error("请检查错误信息并重试")
    logger.info("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 