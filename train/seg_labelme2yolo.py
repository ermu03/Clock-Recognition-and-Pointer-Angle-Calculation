#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :new 
@File    : labelme2yolo.py
@IDE     : PyCharm 
@Author  : 肆十二(付费咨询QQ: 3045834499) 粉丝可享受99元调试服务
@Description  : TODO 添加文件描述
@Date    : 2025/1/30 10:05 
'''
import os
import json
import shutil
from sklearn.model_selection import train_test_split
import glob
import yaml

################################### 可选配置项 ###################################################
# 定义原始数据集路径 E:\YOLO\ImgSegment\test_transform
original_dataset_dir = r"E:\YOLO\ImgSegment"                    # 原始数据集的目录
annotations_dir = os.path.join(original_dataset_dir, 'jsons')   # 标注文件annotations 目录
images_dir = os.path.join(original_dataset_dir, 'test_transform')       # 图片所在目录
coverted_dataset_dir = r'E:\YOLO\ImgSegment\train_data'  # 转换过后的数据集

# 类别到ID的映射，根据实际情况修改
class_map = {
    "long_pointer": 0,
    "short_pointer": 1,
}


################################### 可选配置项 ###################################################

def convert_labelme_to_yolo(json_file, output_label_file):
    # json_str = json_str.replace('\\', '\\\\')
    # db = json.loads(json_str)
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    with open(output_label_file, 'w') as label_f:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # 根据类别名获取类别ID
            class_id = class_map.get(label)
            if class_id is None:
                print(f"警告: 未知类别 {label} 在文件 {json_file}")
                continue

            # YOLO格式要求: class x_center y_center width height
            # 这里的值都是相对于图像宽度和高度的归一化值
            label_line = f"{class_id}"
            # 遍历分割的点，将分割的点加入数据集中
            for point in shape["points"]:
                x = point[0] / image_width  # mask轮廓中一点的X坐标
                y = point[1] / image_height
                label_line = label_line + f" {x:.6f} {y:.6f}"

            # 写入YOLO格式的标签文件 分割
            label_f.write(label_line + "\n")



def copy_and_convert(json_files, phase):
    supported_extensions = ['.jpg', '.png']  # 支持的图像格式

    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_copied = False

        for ext in supported_extensions:
            image_file = os.path.join(images_dir, base_name + ext)
            if os.path.isfile(image_file):
                target_image_dir = os.path.join(coverted_dataset_dir, 'images', phase)
                os.makedirs(target_image_dir, exist_ok=True)
                shutil.copy(image_file, target_image_dir)
                image_copied = True
                break

        if not image_copied:
            print(f"警告: 没有找到与 {json_file} 对应的图片文件")
            continue

        target_labels_dir = os.path.join(coverted_dataset_dir, 'labels', phase)
        os.makedirs(target_labels_dir, exist_ok=True)

        output_label_file = os.path.join(target_labels_dir, base_name + '.txt')
        convert_labelme_to_yolo(json_file, output_label_file)

# 将labelme文件转化为yolo对应的txt格式的文件
if __name__ == "__main__":
    # 创建转换数据集目录
    if not os.path.exists(coverted_dataset_dir):
        os.mkdir(coverted_dataset_dir)

    # 创建转换数据集目录结构
    for folder in ['train', 'val']:
        os.makedirs(os.path.join(coverted_dataset_dir, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(coverted_dataset_dir, 'labels', folder), exist_ok=True)

    # 获取所有的json文件
    json_files = glob.glob(os.path.join(annotations_dir, '*.json'))

    # 将数据划分为训练集和验证集   8 : 2
    train_files, val_files = train_test_split(json_files, test_size=0.2, random_state=42)

    # 执行数据集划分与转换
    copy_and_convert(train_files, 'train')
    copy_and_convert(val_files, 'val')

    # 数据集配置信息，相对于 YOLOv8 目录
    data_yaml = {
        'yaml_file_path': os.path.join(coverted_dataset_dir, 'data.yaml'),
        'train': os.path.join('../..', coverted_dataset_dir, 'images', 'train'),
        'val': os.path.join('../..', coverted_dataset_dir, 'images', 'val'),
        'nc': len(class_map),  # 类别数量
        'names': list(class_map.keys())  # 类别名称列表
    }

    # 写入 YAML 文件
    with open(data_yaml['yaml_file_path'], 'w') as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML 文件已创建: {data_yaml['yaml_file_path']}")
