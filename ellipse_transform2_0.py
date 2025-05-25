import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from tqdm import tqdm

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def extract_and_evaluate_ellipse(image, mask, save_visualization=True):
    """
    从掩码中提取轮廓，拟合椭圆并评估其有效性
    
    参数:
    image: OpenCV格式的图像 (numpy array)
    mask: YOLO生成的掩码 (numpy array)
    save_visualization: 是否需要生成可视化图像
    
    返回:
    tuple: (椭圆参数, 是否有效椭圆, 紧凑度评分, 轮廓图像, 椭圆图像)
    """
    if image is None or mask is None:
        return None, False, 0, None, None
    
    # 制作一份RGB格式图像用于可视化
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if save_visualization else None
    
    # 将掩码转换为二值图像
    binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    
    # 提取掩码的轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 选择最大的轮廓
    if not contours:
        return None, False, 0, None, None  # 未检测到轮廓
    
    # 选出面积最大的轮廓
    best_contour = max(contours, key=cv2.contourArea)
    
    # 如果需要可视化, 绘制最佳轮廓图像
    best_contour_img = None
    if save_visualization:
        best_contour_img = image.copy()
        cv2.drawContours(best_contour_img, [best_contour], 0, (255, 0, 0), 2)
    
    # 确保轮廓点数足够拟合椭圆
    if len(best_contour) < 5:
        return None, False, 0, best_contour_img, None  # 点数不足
    
    # 拟合椭圆
    best_ellipse = cv2.fitEllipse(best_contour)
    
    # 1. 计算轮廓周长与面积比例(紧凑度)
    contour_area = cv2.contourArea(best_contour)
    perimeter = cv2.arcLength(best_contour, True)
    circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 2. 椭圆参数
    (center, (width, height), angle) = best_ellipse
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
    
    # 简化判断条件
    is_valid = (aspect_ratio < 3.0 and 
                aspect_ratio > 1.09 and 
                circularity > 0.5)
    
    # 如果需要可视化, 绘制拟合椭圆的图像
    best_ellipse_img = None
    if save_visualization:
        best_ellipse_img = image.copy()
        cv2.ellipse(best_ellipse_img, best_ellipse, (0, 0, 255), 2)
    
    return best_ellipse, is_valid, circularity, best_contour_img, best_ellipse_img



def transform_ellipse_from_mask(image, mask, save_visualization=True):
    """
    使用YOLO提供的掩码拟合椭圆并进行仿射变换
    
    参数:
    image: OpenCV格式的图像 (numpy array)
    mask: YOLO生成的掩码 (numpy array)
    save_visualization: 是否生成可视化结果图像
    
    返回:
    numpy.ndarray: 校正后的图像, 如果未检测到有效椭圆则返回原图
    numpy.ndarray: 可视化过程的图像, 如果save_visualization为False则返回None
    """
    # 提取并评估椭圆
    ellipse_data = extract_and_evaluate_ellipse(image, mask, save_visualization)
    best_ellipse, is_valid_ellipse, circularity, best_contour_img, best_ellipse_img = ellipse_data
    
    # 如果没有有效椭圆
    if best_ellipse is None:
        return image, None
    
    # 获取椭圆参数
    (center, (width, height), angle) = best_ellipse
    cx, cy = center
    major_axis = max(width, height)
    minor_axis = min(width, height)
    aspect_ratio = major_axis / minor_axis
    
    # 计算缺失的变量
    # 计算椭圆评分 (使用紧凑度作为基础)
    ellipse_score = circularity
    
    # 创建校正后的图像
    corrected_img = None
    
    # 仅当轮廓确实是椭圆时才进行仿射变换
    if is_valid_ellipse:
        # 修正角度解释
        if height > width:
            angle += 90
        
        # 转换角度为弧度
        angle_rad = np.deg2rad(angle)
        
        # 创建更大的空白画布
        h, w = image.shape[:2]
        canvas_size = int(max(h, w) * max(2.0, aspect_ratio) * 1.2)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # 复制原图到画布中心
        x_offset = (canvas_size - w) // 2
        y_offset = (canvas_size - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        # 调整中心点坐标
        cx_new = cx + x_offset
        cy_new = cy + y_offset
        
        # 计算变换矩阵
        T1 = np.array([[1, 0, -cx_new], [0, 1, -cy_new], [0, 0, 1]])
        
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        R1 = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        
        scale_x = 1.0
        scale_y = major_axis / minor_axis
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        
        R2 = np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]])
        
        T2 = np.array([[1, 0, cx_new], [0, 1, cy_new], [0, 0, 1]])
        
        # 组合变换矩阵
        M = T2.dot(R2).dot(S).dot(R1).dot(T1)
        affine_matrix = M[:2, :]
        
        # 执行仿射变换
        corrected_canvas = cv2.warpAffine(canvas, affine_matrix, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
        
        # 裁剪黑边
        corrected_gray = cv2.cvtColor(corrected_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(corrected_gray, 5, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w_crop, h_crop = cv2.boundingRect(max_contour)
            
            # 减小黑边
            padding = 5
            margin = 15  # 略微增加向内收缩的像素数
            x = max(0, x + margin - padding)
            y = max(0, y + margin - padding)
            w_crop = min(canvas_size - x, w_crop - 2*margin + 2*padding)
            h_crop = min(canvas_size - y, h_crop - 2*margin + 2*padding)
            
            corrected_img = corrected_canvas[y:y+h_crop, x:x+w_crop]
        else:
            corrected_img = corrected_canvas
    else:
        # 如果不是椭圆, 则直接使用原图
        corrected_img = image.copy()
    
    # 创建可视化图像
    visualization = None
    if save_visualization:
        # 为可视化重新创建RGB图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 将所有图像显示在一个2x2布局中
        fig = plt.figure(figsize=(12, 10))
        
        # 原始图像
        plt.subplot(2, 2, 1)
        plt.imshow(image_rgb)
        plt.title('原始图像')
        plt.axis('off')
        
        # 最佳轮廓
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(best_contour_img, cv2.COLOR_BGR2RGB))
        plt.title('YOLO掩码轮廓')
        plt.axis('off')
        
        # 拟合椭圆
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(best_ellipse_img, cv2.COLOR_BGR2RGB))
        ellipse_title = f'拟合椭圆 (比例: {aspect_ratio:.2f}, 分数:{ellipse_score:.2f})'
        if is_valid_ellipse:
            ellipse_title += f' - 有效椭圆'
        else:
            ellipse_title += f' - 非椭圆'
        plt.title(ellipse_title)
        plt.axis('off')
        
        # 校正后图像
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        if is_valid_ellipse:
            plt.title('校正后图像(已裁剪黑边)')
        else:
            plt.title('原始图像(未检测到有效椭圆)')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 将图形转换为图像
        fig.canvas.draw()
        visualization = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
    
    return corrected_img, visualization

def transform_elliptical_images_batch(image_folder, model_path, output_folder, vis_folder):
    """
    批量处理文件夹中的图像，检测椭圆并将其转换为圆形
    
    参数:
    image_folder: 输入图像文件夹路径
    model_path: YOLO模型路径
    output_folder: 输出变换后图像的文件夹路径
    vis_folder: 输出可视化结果的文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)
    
    # 加载YOLO模型
    model = YOLO(model_path)
    
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # 处理每个图像
    for img_file in tqdm(image_files, desc="处理图像"):
        # 图像完整路径
        img_path = os.path.join(image_folder, img_file)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 使用YOLO进行分割
        results = model.predict(source=img_path, task='segment', save=False, retina_masks=True)
        
        # 检查是否有检测结果
        if len(results) == 0 or results[0].masks is None:
            print(f"未检测到物体: {img_path}")
            continue
        
        # 获取最大的掩码(假设最大掩码是目标椭圆对象)
        result = results[0]
        if result.masks.data.shape[0] > 0:
            # 找到面积最大的掩码
            mask_areas = [(mask.sum().item(), i) for i, mask in enumerate(result.masks.data)]
            _, max_mask_idx = max(mask_areas)
            
            # 获取最大掩码
            max_mask = result.masks.data[max_mask_idx]
            
            # 使用掩码拟合椭圆并变换
            corrected_img, visualization = transform_ellipse_from_mask(image, max_mask)
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            # 保存变换后的图像
            if corrected_img is not None:
                output_path = os.path.join(output_folder, f"{base_name}_transformed.jpg")
                cv2.imwrite(output_path, corrected_img)
            
            # 保存可视化结果
            if visualization is not None:
                vis_path = os.path.join(vis_folder, f"{base_name}_vis.jpg")
                # 转换RGB为BGR (OpenCV格式)
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_path, visualization_bgr)
        else:
            print(f"未检测到有效掩码: {img_path}")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用YOLO分割结果来检测椭圆并进行仿射变换')
    parser.add_argument('--image_folder', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--model_path', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--output_folder', type=str, default='transformed_images', help='输出变换后图像的文件夹路径')
    parser.add_argument('--vis_folder', type=str, default='visualization_results', help='输出可视化结果的文件夹路径')
    
    args = parser.parse_args()
    
    # 处理文件夹
    transform_elliptical_images_batch(args.image_folder, args.model_path, args.output_folder, args.vis_folder)
    
    print(f"处理完成！变换后的图像保存在 {args.output_folder} 文件夹中")
    print(f"可视化结果保存在 {args.vis_folder} 文件夹中")

"""
椭圆判断标准及原理

代码中判断一个轮廓是否为有效椭圆使用了多个几何和形态学指标，每个指标都有其特定的意义: 

1. 长宽比(aspect_ratio)
条件: 1.08 < aspect_ratio < 3.0
原理: 
下限(1.05)确保形状不是完美圆形，因为圆的长宽比=1
上限(3.0)防止过于扁平的椭圆，这类通常是由线条或边缘引起的，不太可能是表盘
2. 紧凑度(circularity)
条件: circularity > 0.5
原理: 
紧凑度是形状周长与面积关系的度量: 4π x 面积 / 周长²
完美圆形的紧凑度=1, 越复杂或不规则的形状紧凑度越低
阈值0.5确保轮廓具有椭圆的基本几何特性而不是复杂形状
3. 综合评分(ellipse_score)
条件: ellipse_score > 0.5

    
    """
