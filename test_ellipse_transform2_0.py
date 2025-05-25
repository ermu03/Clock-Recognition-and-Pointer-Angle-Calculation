# -*- coding: utf-8 -*-
"""
# @FileName      : all_test
# @Time          : 2025-05-20 21:27:05
# @Author        : ermu
# @Email         : 168235638@qq.com
# @description   : 测试模块
"""

import os
from ellipse_transform2_0 import transform_elliptical_images_batch

def test_ellipse_transform():
    """测试椭圆变换模块"""
    print("开始测试椭圆变换模块...")
    
    # 设置参数
    image_folder = r"test_cropped_clocks"
    model_path = r"yolov8m-seg.pt"
    output_folder = r"test_transform"
    vis_folder = r"test_transform_vis"
    
    # 调用椭圆变换模块
    transform_elliptical_images_batch(image_folder, model_path, output_folder, vis_folder)
    print(f"处理完成! 结果保存在 {output_folder} 和 {vis_folder}")

if __name__ == "__main__":
    test_ellipse_transform()