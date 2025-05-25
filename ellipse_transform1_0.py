import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def is_ellipse(contour, ellipse, image_shape):
    """
    判断给定轮廓是否为椭圆
    
    参数:
    contour: 轮廓点集
    ellipse: 拟合的椭圆参数 ((cx,cy), (width,height), angle)
    image_shape: 图像尺寸 (height, width)
    
    返回:
    bool: 是否为椭圆
    float: 椭圆拟合度分数 (0-1)
    float: 椭圆面积与图像面积的比率
    """
    # 1. 计算轮廓周长与面积比例（紧凑度）
    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 2. 计算轮廓与拟合椭圆的重叠度
    # 创建掩码图像
    mask_contour = np.zeros((1000, 1000), dtype=np.uint8)
    mask_ellipse = np.zeros((1000, 1000), dtype=np.uint8)
    
    # 绘制轮廓和椭圆
    cv2.drawContours(mask_contour, [contour], 0, 255, -1)
    cv2.ellipse(mask_ellipse, ellipse, 255, -1)
    
    # 计算重叠区域
    intersection = cv2.bitwise_and(mask_contour, mask_ellipse)
    union = cv2.bitwise_or(mask_contour, mask_ellipse)
    
    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)
    
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # 3. 椭圆拟合误差
    (center, (width, height), angle) = ellipse
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
    
    # 4. 计算拟合椭圆面积与图像面积的比率
    ellipse_area = np.pi * (width/2) * (height/2)  # 椭圆面积公式：π * a * b
    image_area = image_shape[0] * image_shape[1]  # 图像面积 = 高 * 宽
    image_ratio = ellipse_area / image_area if image_area > 0 else float('inf')
    
    # 5. 计算拟合椭圆面积与轮廓面积比率
    contour_ratio = ellipse_area / contour_area if contour_area > 0 else float('inf')
    
    # 椭圆拟合度分数
    area_score_image = max(0, 1 - abs(image_ratio - 0.5)) if 0.1 <= image_ratio <= 0.9 else 0
    area_score_contour = max(0, 1 - abs(contour_ratio - 1)) if 0.7 <= contour_ratio <= 1.3 else 0
    
    ellipse_score = (circularity * 0.2) + (iou * 0.4) + (area_score_contour * 0.2) + (area_score_image * 0.2)
    
    # 判断条件
    is_valid_ellipse = (iou > 0.5 and 
                        aspect_ratio < 3.0 and 
                        aspect_ratio > 1.05 and 
                        circularity > 0.5 and
                        0.2 <= image_ratio <= 0.9 and
                        0.6 <= contour_ratio <= 1.3 and
                        ellipse_score > 0.6)
    
    return is_valid_ellipse, ellipse_score, image_ratio

def transform_ellipse_image(image, save_visualization=True):
    """
    检测图像中的椭圆并进行仿射变换使其变为圆形
    
    参数:
    image: OpenCV格式的图像 (numpy array)
    save_visualization: 是否生成可视化结果图像
    
    返回:
    numpy.ndarray: 校正后的图像，如果未检测到有效椭圆则返回原图
    numpy.ndarray: 可视化过程的图像，如果save_visualization为False则返回None
    """
    if image is None:
        return None, None
    
    # 制作一份RGB格式图像用于可视化
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if save_visualization else None
    
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去除噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 筛选合适的轮廓
    valid_contours = []
    for contour in contours:
        # 至少5个点才能拟合椭圆
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            # 排除太小的轮廓
            if area > 100:  
                # 计算凸包
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                # 计算椭圆度和凸性
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # 椭圆应该有较高的凸性
                if solidity > 0.8:
                    valid_contours.append((contour, area))
    
    # 选择面积最大的轮廓
    if not valid_contours:
        return image, None  # 未检测到有效轮廓，返回原图
    
    # 选出面积最大的轮廓
    best_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    # 如果需要可视化，绘制最佳轮廓图像
    best_contour_img = None
    if save_visualization:
        best_contour_img = image.copy()
        cv2.drawContours(best_contour_img, [best_contour], 0, (255, 0, 0), 2)
    
    # 拟合椭圆
    best_ellipse = cv2.fitEllipse(best_contour)
    
    # 判断是否为椭圆，传入图像尺寸
    is_valid_ellipse, ellipse_score, area_ratio = is_ellipse(best_contour, best_ellipse, image.shape[:2])
    
    # 如果需要可视化，绘制拟合椭圆的图像
    best_ellipse_img = None
    if save_visualization:
        best_ellipse_img = image.copy()
        cv2.ellipse(best_ellipse_img, best_ellipse, (0, 0, 255), 2)
    
    # 获取椭圆参数
    (center, (width, height), angle) = best_ellipse
    cx, cy = center
    major_axis = max(width, height)
    minor_axis = min(width, height)
    aspect_ratio = major_axis / minor_axis
    
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
        # 如果不是椭圆，则直接使用原图
        corrected_img = image.copy()
    
    # 创建可视化图像
    visualization = None
    if save_visualization:
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
        plt.title('最佳轮廓')
        plt.axis('off')
        
        # 拟合椭圆
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(best_ellipse_img, cv2.COLOR_BGR2RGB))
        ellipse_title = f'拟合椭圆 (比例: {aspect_ratio:.2f})'
        if is_valid_ellipse:
            ellipse_title += f' - 有效椭圆(分数:{ellipse_score:.2f}, 占图:{area_ratio:.2f})'
        else:
            ellipse_title += f' - 非椭圆(分数:{ellipse_score:.2f}, 占图:{area_ratio:.2f})'
        plt.title(ellipse_title)
        plt.axis('off')
        
        # 校正后图像
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        if is_valid_ellipse:
            plt.title('校正后图像（已裁剪黑边）')
        else:
            plt.title('原始图像（未检测到有效椭圆）')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 将图形转换为图像
        fig.canvas.draw()
        visualization = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # 关闭图形，避免显示
    
    return corrected_img, visualization

# 使用示例
if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        # 从命令行参数获取图像路径
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            # 处理图像，同时生成可视化结果
            result, visualization = transform_ellipse_image(image)
            
            # 显示结果
            cv2.imshow("原始图像", image)
            cv2.imshow("处理后图像", result)
            
            # 如果有可视化结果，显示它
            if visualization is not None:
                # 转换RGB为BGR (OpenCV格式)
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imshow("处理过程可视化", visualization_bgr)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 保存结果
            cv2.imwrite("result_transformed.jpg", result)
            if visualization is not None:
                cv2.imwrite("result_visualization.jpg", visualization_bgr)
            print("结果已保存为 result_transformed.jpg 和 result_visualization.jpg")
        else:
            print(f"无法加载图像: {image_path}")
    else:
        print("使用方法: python ellipse_transform.py <图像路径>")
