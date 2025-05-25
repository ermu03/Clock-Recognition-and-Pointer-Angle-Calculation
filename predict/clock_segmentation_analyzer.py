from ultralytics import YOLO
import cv2
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class PointerDetector:
    """时钟指针检测器类"""
    
    def __init__(self, model_path, input_folder, output_folder):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            input_folder: 输入图像文件夹
            output_folder: 输出结果文件夹
        """
        self.model_path = model_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model = None
        self.long_idx = None
        self.short_idx = None
        self.result_file = os.path.join(output_folder, "detection_results.txt")
        
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
    
    def load_model(self):
        """加载并初始化YOLOv8分割模型"""
        self.model = YOLO(self.model_path)
        
        # 修改模型的类别名称为简短形式
        if 'long_pointer' in self.model.names.values():
            # 通过值查找键
            for idx, name in self.model.names.items():
                if name == 'long_pointer':
                    self.long_idx = idx
                    self.model.names[idx] = 'l'
                elif name == 'short_pointer':
                    self.short_idx = idx
                    self.model.names[idx] = 's'
        else:
            # 假设是按索引编号的情况
            self.long_idx = 0  
            self.short_idx = 1  
            self.model.names[self.long_idx] = 'l'  
            self.model.names[self.short_idx] = 's'
        
        return self.model
    
    def init_result_file(self):
        """初始化结果文件"""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            f.write("# 图像名称  long_pointer数量  short_pointer数量\n")
    
    def get_image_paths(self):
        """获取输入文件夹中的所有图像路径"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.input_folder, ext)))
        return image_paths
    
    def process_image(self, img_path):
        """
        处理单张图像
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            dict: 包含检测结果的字典
        """
        # 获取图像文件名（不含路径）
        img_filename = os.path.basename(img_path)
        base_name, ext = os.path.splitext(img_filename)
        
        # 执行预测
        results = self.model(img_path)
        result = results[0]
        
        # 可视化结果（包含边界框和分割掩码）
        annotated_img = result.plot()
        
        # 保存结果
        output_path = os.path.join(self.output_folder, f"{base_name}_seg{ext}")
        cv2.imwrite(output_path, annotated_img)
        
        # 统计long_pointer和short_pointer的数量
        long_count = 0
        short_count = 0
        
        # 统计检测到的类别和数量
        for c in result.boxes.cls.unique():
            class_idx = int(c)
            n = (result.boxes.cls == c).sum()
            
            # 记录长短指针的数量
            if class_idx == self.long_idx:
                long_count = int(n)
            elif class_idx == self.short_idx:
                short_count = int(n)
        
        # 将结果写入文本文件
        with open(self.result_file, 'a') as f:
            f.write(f"{img_filename}  {long_count}  {short_count}\n")
        
        return {
            'img_path': img_path,
            'output_path': output_path,
            'long_count': long_count,
            'short_count': short_count
        }
    
    def run(self):
        """执行批量检测流程"""
        self.load_model()
        self.init_result_file()
        
        image_paths = self.get_image_paths()
        total_images = len(image_paths)
        
        print(f"找到 {total_images} 张图像")
        
        for i, img_path in enumerate(image_paths):
            result = self.process_image(img_path)
            
            # 打印处理进度
            print(f"已处理: [{i+1}/{total_images}] {img_path}")
            print(f"  检测到 {result['long_count']} 个 l (长指针)")
            print(f"  检测到 {result['short_count']} 个 s (短指针)")
        
        print(f"\n所有图像处理完成! 结果已保存到 {self.output_folder} 文件夹")
        print(f"检测统计结果已保存到 {self.result_file}")
        return self.result_file


class PerformanceAnalyzer:
    """检测性能分析类"""
    
    def __init__(self):
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        matplotlib.rcParams['font.family'] = 'SimHei'
        matplotlib.rcParams['font.size'] = 12
    
    def analyze_performance(self, result_file, output_image="detection_performance-default.png"):
        """
        分析检测性能
        
        Args:
            result_file: 检测结果文件路径
            output_image: 输出图像名称
            
        Returns:
            dict: 性能指标
        """
        # 读取检测结果文件
        df = pd.read_csv(result_file, sep=r'\s+', comment='#', names=['filename', 'long_count', 'short_count'])
        
        # 总样本数
        total_samples = len(df)
        
        # 1. 计算漏检率 - 任一类检测数为0的情况
        missed_samples = df[(df['long_count'] == 0) | (df['short_count'] == 0)]
        missed_rate = len(missed_samples) / total_samples
        
        # 2. 计算完美检测率 - 两类都恰好检测到1个
        perfect_samples = df[(df['long_count'] == 1) & (df['short_count'] == 1)]
        perfect_rate = len(perfect_samples) / total_samples
        
        # 3. 计算误检率 - 分别统计两类
        # Long pointer的误检（检测到>1个）
        long_false_samples = df[df['long_count'] > 1]
        long_false_rate = len(long_false_samples) / total_samples
        
        # Short pointer的误检（检测到>1个）
        short_false_samples = df[df['short_count'] > 1]
        short_false_rate = len(short_false_samples) / total_samples
        
        # 打印统计结果
        print(f"检测性能分析报告")
        print(f"总样本数: {total_samples}")
        print(f"漏检率: {missed_rate:.2%} ({len(missed_samples)}个样本)")
        print(f"完美检测率: {perfect_rate:.2%} ({len(perfect_samples)}个样本)")
        print(f"Long pointer误检率: {long_false_rate:.2%} ({len(long_false_samples)}个样本)")
        print(f"Short pointer误检率: {short_false_rate:.2%} ({len(short_false_samples)}个样本)")
        
        # 详细分析
        print("\n漏检详情:")
        print(f"- 只漏检Long pointer: {len(df[(df['long_count'] == 0) & (df['short_count'] > 0)])}个样本")
        print(f"- 只漏检Short pointer: {len(df[(df['long_count'] > 0) & (df['short_count'] == 0)])}个样本")
        print(f"- 两类都漏检: {len(df[(df['long_count'] == 0) & (df['short_count'] == 0)])}个样本")
        
        print("\n误检详情:")
        print(f"- Long pointer检测到2个: {len(df[df['long_count'] == 2])}个样本")
        print(f"- Long pointer检测到3个或更多: {len(df[df['long_count'] >= 3])}个样本")
        print(f"- Short pointer检测到2个: {len(df[df['short_count'] == 2])}个样本")
        print(f"- Short pointer检测到3个或更多: {len(df[df['short_count'] >= 3])}个样本")
        
        # 可视化性能指标
        plt.figure(figsize=(12, 8))
        
        # 1. 主要性能指标柱状图
        plt.subplot(2, 1, 1)
        labels = ['完美检测率', '漏检率', 'Long误检率', 'Short误检率']
        values = [perfect_rate, missed_rate, long_false_rate, short_false_rate]
        colors = ['green', 'red', 'orange', 'blue']
        
        bars = plt.bar(labels, values, color=colors)
        plt.ylabel('比率')
        plt.title('指针检测性能分析')
        plt.ylim(0, max(values) * 1.2)  # 为了美观调整Y轴范围
        
        # 在柱状图上显示百分比
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom')
        
        # 2. 检测数量分布饼图
        plt.subplot(2, 2, 3)
        long_counts = df['long_count'].value_counts().sort_index()
        plt.pie(long_counts, labels=[f"{i}个" for i in long_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('长指针检测数量分布')
        
        plt.subplot(2, 2, 4)
        short_counts = df['short_count'].value_counts().sort_index()
        plt.pie(short_counts, labels=[f"{i}个" for i in short_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('短指针检测数量分布')
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        plt.show()
        
        print(f"\n分析结果图表已保存为 '{output_image}'")
        
        return {
            'total_samples': total_samples,
            'missed_rate': missed_rate,
            'perfect_rate': perfect_rate,
            'long_false_rate': long_false_rate,
            'short_false_rate': short_false_rate
        }


# 主函数
def main():
    # 指定参数
    model_path = 'E:/YOLO/ImgSegment/runs/segment/train-l/weights/best-l.pt'
    input_folder = "transforms_clocks"
    output_folder = "segmentation_results-l"
    # performance_image = "detection_performance-l.png"
    performance_image = "detection_performance-l.png"
    
    # # 1. 执行检测
    # detector = PointerDetector(
    #     model_path=model_path,
    #     input_folder=input_folder,
    #     output_folder=output_folder
    # )
    # result_file = detector.run()
    
    result_file = "segmentation_results-l\detection_results.txt"
    
    # 2. 分析性能
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_performance(result_file, performance_image)
    
    return metrics


# 程序入口
if __name__ == '__main__':
    metrics = main()