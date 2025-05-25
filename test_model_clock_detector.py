# -*- coding: utf-8 -*-
"""
# @description   : 演示如何导入和使用表盘检测模块
"""

# 导入 clock_detector 模块
from clock_detector import detect_and_crop_clocks

# 如果需要使用其他子函数，也可以单独导入
# from clock_detector import load_model, detect_objects, select_best_clock, crop_and_save_image

def main():
    """
    主函数，用于演示如何使用表盘检测模块
    """
    # 设置参数
    model_path = 'yolov8m-seg.pt'  # YOLO模型路径
    image_folder = './temp'  # 输入图像文件夹
    output_folder = './test_cropped_clocks'  # 输出文件夹
    
    print("开始检测和裁剪表盘...")
    
    # 调用模块的主函数进行表盘检测和裁剪
    cropped_images = detect_and_crop_clocks(
        model_path=model_path,
        image_folder=image_folder,
        output_folder=output_folder
    )
    
    # 处理检测结果
    if cropped_images:
        print(f"成功检测到 {len(cropped_images)} 个表盘！")
        print("裁剪后的图像保存在：")
        for img_path in cropped_images:
            print(f"  - {img_path}")
    else:
        print("未检测到任何表盘。")

if __name__ == "__main__":
    main()
