import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 12

# 读取检测结果文件
result_file = "segmentation_results-s/detection_results-s.txt"

# 跳过第一行注释，并设置列名
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
plt.savefig('detection_performance-s.png', dpi=300)
plt.show()

print("\n分析结果图表已保存为 'detection_performance-s.png'")