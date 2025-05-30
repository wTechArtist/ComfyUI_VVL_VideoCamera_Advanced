#!/usr/bin/env python3
"""
测试ComfyUI的带注释文件路径格式
"""

import os
import sys

# 添加ComfyUI路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

try:
    import folder_paths
    
    def test_annotated_path():
        """测试带注释的文件路径格式"""
        # 测试我们使用的路径格式
        test_path = "3d/vggt_3d_model_conf50.0_camTrue.glb [output]"
        
        print(f"测试路径: {test_path}")
        
        # 解析带注释的路径
        name, base_dir = folder_paths.annotated_filepath(test_path)
        print(f"解析结果: name={name}, base_dir={base_dir}")
        
        # 获取完整路径
        full_path = folder_paths.get_annotated_filepath(test_path)
        print(f"完整路径: {full_path}")
        
        # 验证这是否与我们预期的output/3d目录匹配
        expected_dir = os.path.join(folder_paths.get_output_directory(), "3d")
        print(f"期望目录: {expected_dir}")
        
        return True
    
    if __name__ == "__main__":
        test_annotated_path()
        
except ImportError as e:
    print(f"无法导入folder_paths: {e}")
    print("请确保在ComfyUI环境中运行此测试") 