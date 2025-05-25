# -*- coding: utf-8 -*-
"""
# @FileName      : transform_clock
# @Time          : 2025-05-24 16:35:15
# @Author        : ermu
# @Email         : 168235638@qq.com
# @description   :

调用ellipse_transform2_0模块处理裁剪的时钟图像
输入: cropped_clocks文件夹中的裁剪表盘图像
输出: transforms_clocks文件夹中的椭圆矫正后图像
"""
import os
import sys

# 添加父目录到Python路径，以便导入clock_detector模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ellipse_transform2_0 import transform_elliptical_images_batch

def main():
    # 设置参数
    model_path = 'yolov8m-seg.pt'  # YOLO模型路径
    input_folder = 'cropped_clocks'  # 裁剪后的表盘图像文件夹
    output_folder = 'transforms_clocks'  # 矫正后的表盘图像保存文件夹
    vis_folder = 'transforms_vis'  # 变换过程的可视化结果文件夹
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在!")
        return
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)
    
    print(f"开始处理 '{input_folder}' 文件夹中的表盘图像...")
    
    # 调用椭圆变换模块的函数进行表盘矫正
    transform_elliptical_images_batch(
        image_folder=input_folder,
        model_path=model_path,
        output_folder=output_folder,
        vis_folder=vis_folder
    )
    
    # 输出处理结果
    print(f"\n处理完成! '{input_folder}' 中的表盘图像已被矫正")
    print(f"矫正后的图像已保存到 '{output_folder}' 文件夹")
    print(f"可视化结果已保存到 '{vis_folder}' 文件夹")

if __name__ == "__main__":
    main()