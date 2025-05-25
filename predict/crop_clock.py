# -*- coding: utf-8 -*-
"""
# @FileName      : crop_clock
# @Time          : 2025-05-24 16:32:21
# @Author        : ermu
# @Email         : 168235638@qq.com
# @description   :  

调用clock_detector模块检测和裁剪表盘
输入: ori_clocks文件夹中的原始图像
输出: cropped_clocks文件夹中的裁剪后表盘图像
"""
import os
import sys
import shutil

# 添加父目录到Python路径，以便导入clock_detector模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clock_detector import detect_and_crop_clocks

def main():
    # 设置参数
    model_path = 'yolov8m-seg.pt'  # YOLO模型路径
    input_folder = 'ori_clocks'     # 原始图像文件夹
    output_folder = 'cropped_clocks'  # 裁剪后的表盘图像保存文件夹
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在!")
        return
    
    # 清空输出文件夹(如果存在)
    if os.path.exists(output_folder):
        print(f"清空 '{output_folder}' 文件夹中的旧文件...")
        shutil.rmtree(output_folder)
    
    print(f"开始处理 '{input_folder}' 文件夹中的图像...")
    
    # 调用clock_detector模块的函数进行表盘检测和裁剪
    cropped_paths = detect_and_crop_clocks(
        model_path=model_path,
        image_folder=input_folder,
        output_folder=output_folder
    )
    
    # 输出处理结果
    print(f"\n处理完成! 从 {input_folder} 中的图像检测并裁剪了 {len(cropped_paths)} 个表盘")
    print(f"裁剪后的图像已保存到 '{output_folder}' 文件夹")

if __name__ == "__main__":
    main()