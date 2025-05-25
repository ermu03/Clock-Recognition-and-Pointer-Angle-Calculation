# -*- coding: utf-8 -*-
"""
# @FileName      : test02
# @Time          : 2025-05-18 18:06:41
# @Author        : ermu
# @Email         : 168235638@qq.com
# @description   :  测试 elipse_transform 模块
"""

import cv2
import os
import numpy as np
from ellipse_transform1_0 import transform_ellipse_image

def test_ellipse_transform():
    """
    测试椭圆变换模块，处理文件夹中的所有图像并保存结果
    """
    # 设置输入和输出文件夹
    input_folder = "./result_cropped/"
    output_folder = "./temp_result8/"
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 遍历处理每张图像
    for i, image_file in enumerate(image_files):
        print(f"正在处理图像 {i+1}/{len(image_files)}: {image_file}")
        
        # 构建完整的图像路径
        image_path = os.path.join(input_folder, image_file)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法加载图像: {image_path}")
            continue
        
        # 使用椭圆变换模块处理图像，同时获取可视化结果
        result, visualization = transform_ellipse_image(image)
        
        if result is not None:
            # 构建输出路径
            base_name = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_folder, f"corrected_{image_file}")
            vis_output_path = os.path.join(output_folder, f"visualization_{base_name}.png")
            
            # 保存结果
            cv2.imwrite(output_path, result)
            print(f"已保存校正结果至: {output_path}")
            
            # 保存可视化结果
            if visualization is not None:
                # 转换RGB为BGR (OpenCV格式)
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_output_path, visualization_bgr)
                print(f"已保存可视化结果至: {vis_output_path}")
        else:
            print(f"跳过图像: {image_file}")
    
    print(f"处理完成! 共处理 {len(image_files)} 张图像.")

if __name__ == "__main__":
    test_ellipse_transform()


