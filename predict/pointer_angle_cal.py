import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def compute_pointer_angle_from_box(box_points, image_shape, fallback=True):
    """
    给定指针矩形的4个顶点和图像尺寸, 计算指针的角度。
    优先使用对角线比较法判断方向, 必要时回退到长边方向法。

    Args:
        box_points: 4个顶点坐标, 形如 (4,2) ndarray
        image_shape: 图像尺寸 (h, w)
        fallback: 是否启用长边方向作为回退策略

    Returns:
        angle (float): 指针方向角度(相对图像中心, 顺时针, 从12点为0度)
        dir_vec (ndarray): 指向向量(用于可视化)
    """
    img_h, img_w = image_shape[:2]
    center = np.array([img_w / 2, img_h / 2])
    rect_center = np.mean(box_points, axis=0)

    diag_pairs = [(box_points[0], box_points[2]), (box_points[1], box_points[3])]
    best_vec = None
    best_dist_diff = 0

    for pt1, pt2 in diag_pairs:
        d1 = np.linalg.norm(pt1 - center)
        d2 = np.linalg.norm(pt2 - center)
        if abs(d1 - d2) > best_dist_diff:
            far = pt1 if d1 > d2 else pt2
            best_vec = far - rect_center
            best_dist_diff = abs(d1 - d2)

    if best_vec is None or best_dist_diff < 10:
        if not fallback:
            return None, None
        # 使用长边方向作为回退
        rect = cv2.minAreaRect(box_points.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.array(box)

        max_len = 0
        for i in range(4):
            pt1, pt2 = box[i], box[(i + 1) % 4]
            length = np.linalg.norm(pt1 - pt2)
            if length > max_len:
                max_len = length
                long_edge = (pt1, pt2)

        d1 = np.linalg.norm(long_edge[0] - center)
        d2 = np.linalg.norm(long_edge[1] - center)
        far = long_edge[0] if d1 > d2 else long_edge[1]
        best_vec = far - rect_center

    dx, dy = best_vec
    # 修正角度计算，确保12点钟方向为0度，顺时针旋转
    angle = np.degrees(np.arctan2(dx, -dy))
    angle = (angle + 360) % 360
    return angle, best_vec

def draw_pointer_angles(image, center, pointer_info):
    """
    在图像上绘制指针方向箭头和角度文本标注。

    Args:
        image: 原图像
        center: 指针起点中心坐标
        pointer_info: 每根指针的方向信息, 包括角度、向量、颜色

    Returns:
        annotated: 带标注的图像
    """
    annotated = image.copy()
    center = tuple(int(v) for v in center)
    cv2.circle(annotated, center, 5, (0, 255, 255), -1)

    # 添加夹角信息
    if 'l' in pointer_info and 's' in pointer_info:
        long_angle = pointer_info['l']['angle']
        short_angle = pointer_info['s']['angle']
        diff = abs(long_angle - short_angle)
        diff = min(diff, 360 - diff)
        
        # 在图像中央顶部添加夹角信息，使用0.8字体大小
        text = f"diff: {diff:.1f}"
        cv2.putText(annotated, text, 
                   (int(center[0] - 80), 30), # 固定在顶部位置
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
        
        # 在图像左下角显示两个指针的角度值，添加deg后缀
        h, w = annotated.shape[:2]
        cv2.putText(annotated, f"{long_angle:.1f} deg", 
                  (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, pointer_info['l']['color'], 2)
        cv2.putText(annotated, f"{short_angle:.1f} deg", 
                  (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, pointer_info['s']['color'], 2)

    # 只绘制箭头，不在箭头旁添加文字
    for name, info in pointer_info.items():
        vec = info['vec']
        color = info['color']
        end = (int(center[0] + vec[0]), int(center[1] + vec[1]))
        cv2.arrowedLine(annotated, center, end, color, 2, tipLength=0.1)

    return annotated

def compute_angles_for_result(result, image):
    """
    对单张图像的YOLO预测结果, 提取两个指针的角度并计算夹角。

    Args:
        result: YOLOv8 推理结果
        image: 原图像(np.ndarray)

    Returns:
        dict 或 None: 包含角度、可视化图像的结果, 若缺失则返回 None
    """
    img_h, img_w = image.shape[:2]
    center = np.array([img_w / 2, img_h / 2])
    
    pointers = {'l': None, 's': None}

    for cls, name in zip([0, 1], ['l', 's']):
        mask_indices = (result.boxes.cls.cpu().numpy() == cls)
        boxes = result.boxes[mask_indices]

        if len(boxes) == 0:
            return None

        confs = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)
        box = boxes.xyxy[best_idx].cpu().numpy()

        x1, y1, x2, y2 = box
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        angle, vec = compute_pointer_angle_from_box(rect, image.shape)
        if angle is None:
            return None

        pointers[name] = {'angle': angle, 'vec': vec, 'color': (0, 0, 255) if name == 'l' else (0, 255, 0)}

    long_angle = pointers['l']['angle']
    short_angle = pointers['s']['angle']
    diff = abs(long_angle - short_angle)
    diff = min(diff, 360 - diff)

    annotated = draw_pointer_angles(image, center, pointers)

    return {
        'long_angle': long_angle,
        'short_angle': short_angle,
        'angle_between': diff,
        'visual': annotated
    }

def batch_process_folder(model_path, input_folder, output_folder):
    """
    对整个文件夹中的图像进行指针识别、角度计算与可视化标注。

    Args:
        model_path: YOLOv8 分割模型路径
        input_folder: 输入图像目录
        output_folder: 输出标注图目录

    Returns:
        result_list: 每张图像的角度计算结果列表
    """
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    results = model.predict(source=input_folder, task='segment', save=False, retina_masks=True)

    result_list = []

    for i, result in enumerate(results):
        image_path = result.path
        image = cv2.imread(image_path)
        name = os.path.basename(image_path)

        computed = compute_angles_for_result(result, image)
        if computed is None:
            print(f"[跳过] {name} 指针缺失或方向不明")
            continue

        angles = {
            'image': name,
            'long_angle': computed['long_angle'],
            'short_angle': computed['short_angle'],
            'angle_between': computed['angle_between']
        }
        result_list.append(angles)

        out_path = os.path.join(output_folder, f"{os.path.splitext(name)[0]}_annotated.jpg")
        cv2.imwrite(out_path, computed['visual'])
        print(f"[处理] {name} → 夹角: {angles['angle_between']:.2f}°")

    return result_list

if __name__ == "__main__":
    # 示例参数路径，可根据实际情况修改
    model_path = "E:/YOLO/ImgSegment/runs/segment/train-s/weights/best-s.pt"
    input_folder = "transforms_clocks"
    output_folder = "end_results"

    print("开始批量处理...")
    results = batch_process_folder(model_path, input_folder, output_folder)
    print(f"共处理 {len(results)} 张图像")
