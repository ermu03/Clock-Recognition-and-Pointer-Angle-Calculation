# -*- coding: utf-8 -*-
"""
# @description   : 表盘检测和裁剪模块
"""

from ultralytics import YOLO
import os
import cv2

def detect_objects(model, image_folder):
    """
    使用YOLO模型检测文件夹中图像的目标
    
    参数:
    model: YOLO模型实例
    image_folder: 图像文件夹路径
    
    返回:
    检测结果列表
    """
    results = model.predict(source=image_folder, task='segment', save=False, retina_masks=True)
    print(f"检测完成, 共处理 {len(results)} 张图像")
    return results

def select_best_clock(result):
    """
    从检测结果中选择最佳的时钟目标
    
    参数:
    result: 单个图像的检测结果
    
    返回:
    tuple: (最佳边界框, 图像名称, 图像基础名称, 原始图像)
    """
    # 获取原始输入图像
    orig_img = result.orig_img
    
    # 提取文件名信息
    full_path = result.path
    orig_img_name = os.path.basename(full_path)
    orig_img_base = os.path.splitext(orig_img_name)[0]
    
    # 提取边界框信息
    boxes = result.boxes.xyxy
    classes = result.boxes.cls
    class_names = [result.names[int(cls)] for cls in classes]
    
    # 筛选类别为 "clock" 的边界框
    clock_candidates = []
    for box, class_name, conf in zip(boxes, class_names, result.boxes.conf):
        if class_name == "clock":
            # 获取边界框坐标和面积
            x_min, y_min, x_max, y_max = map(float, box.tolist())
            area = (x_max - x_min) * (y_max - y_min)
            clock_candidates.append((box, conf, area))
    
    # 如果有检测到表盘，选择最佳的一个
    if clock_candidates:
        print(f"图像 {orig_img_name}: 检测到 {len(clock_candidates)} 个 'clock' 类别的候选目标")
        
        # 获取置信度范围
        max_conf = max(candidate[1] for candidate in clock_candidates)
        min_conf = min(candidate[1] for candidate in clock_candidates)
        conf_range = max_conf - min_conf if max_conf > min_conf else 1.0
        
        # 获取面积范围
        max_area = max(candidate[2] for candidate in clock_candidates)
        min_area = min(candidate[2] for candidate in clock_candidates)
        area_range = max_area - min_area if max_area > min_area else 1.0
        
        # 计算每个候选框的综合得分
        best_score = -1
        best_box = None
        for box, conf, area in clock_candidates:
            # 归一化指标
            norm_conf = (conf - min_conf) / conf_range if conf_range > 0 else conf
            norm_area = 1.0 - ((area - min_area) / area_range if area_range > 0 else 0)
            
            # 综合得分
            score = norm_conf * 0.7 + norm_area * 0.3
            
            # 更新最佳框
            if score > best_score:
                best_score = score
                best_box = box
        
        print(f"选择了置信度最高且面积适当的表盘，得分: {best_score:.4f}")
        return best_box, orig_img_name, orig_img_base, orig_img
    else:
        print(f"图像 {orig_img_name}: 未检测到 'clock' 类别的目标")
        return None, orig_img_name, orig_img_base, orig_img

def crop_and_save_image(box, orig_img, orig_img_base, output_folder, index=0):
    """
    裁剪并保存图像
    
    参数:
    box: 边界框坐标
    orig_img: 原始图像
    orig_img_base: 图像基础名称
    output_folder: 输出文件夹路径
    index: 索引编号
    
    返回:
    str: 保存的图像路径
    """
    if box is None:
        return None
    
    # 提取边界框坐标并转换为整数
    x_min, y_min, x_max, y_max = map(int, box.tolist())
    print(f"Box {index+1}: ({x_min}, {y_min}, {x_max}, {y_max})")
    
    # 从原始图像中截取区域
    cropped_img = orig_img[y_min:y_max, x_min:x_max]
    
    # 保存截取的图像
    cropped_save_path = os.path.join(output_folder, f"{orig_img_base}_cropped.jpg")
    cv2.imwrite(cropped_save_path, cropped_img)
    print(f"保存截取的图像到: {cropped_save_path}")
    
    return cropped_save_path

def detect_and_crop_clocks(model_path, image_folder, output_folder):
    """
    检测图像中的表盘并裁剪保存
    
    参数:
    model_path: YOLO模型的路径, 例如 'yolov8m-seg.pt'
    image_folder: 输入图像所在的文件夹路径
    output_folder: 裁剪结果保存的文件夹路径
    
    返回:
    list: 裁剪后图像的保存路径列表
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 检测目标
    results = detect_objects(model, image_folder)
    
    # 保存裁剪的图像路径列表
    cropped_image_paths = []
    
    # 处理每个检测结果
    for idx, result in enumerate(results):
        # 选择最佳的时钟目标
        best_box, img_name, img_base, orig_img = select_best_clock(result)
        
        # 如果找到有效的表盘，裁剪并保存
        if best_box is not None:
            cropped_path = crop_and_save_image(best_box, orig_img, img_base, output_folder)
            if cropped_path:
                cropped_image_paths.append(cropped_path)
    
    print(f"处理完成！共裁剪 {len(cropped_image_paths)} 个表盘图像")
    return cropped_image_paths

# 示例用法
# if __name__ == "__main__":
#     # 默认参数
#     model_path = 'yolov8m-seg.pt'
#     image_folder = './clockImgs'
#     output_folder = './result_cropped'
    
#     # 调用函数
#     detect_and_crop_clocks(model_path, image_folder, output_folder)
