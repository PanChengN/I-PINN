import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('relative_errors.csv')
# 将 'network_structure' 转换为有特定顺序的分类类型
network_order = ["3x30", "3x50", "3x70", "3x90", "5x30", "5x50", "5x70", "5x90", "7x30", "7x50", "7x70", "7x90"]
df['network_structure'] = pd.Categorical(df['network_structure'], categories=network_order, ordered=True)
# 定义每个模型类型的标记和线型
markers = {"PINN": "o", "IA-PINN": "s", "IAW-PINN": "D", "I-PINN": "^"}
# linestyles = {"PINN": "-", "IFNN_PINN": "--", "AW_PINN": "-.", "IAW_PINN": ":"}
linestyles = {"PINN": ":", "IA-PINN": "--", "IAW-PINN": "-.", "I-PINN": "-"}

# 设置绘图风格
# sns.set(style="darkgrid")

# 创建折线图
plt.figure(figsize=(12, 8))
for model_type, marker in markers.items():
    linestyle = linestyles[model_type]
    subset = df[df['model_type'] == model_type]
    sns.lineplot(data=subset, x="network_structure", y="relative_error", marker=marker, linestyle=linestyle, label=model_type,linewidth=2.5)

# 添加标题和标签
# plt.title("Comparison of Relative Error Across Different Network Structures and Models")
plt.xlabel("Network Structure")
plt.ylabel("Relative Error")
# plt.legend(title="Model Type")
# plt.xticks(rotation=45)
plt.yscale('log')  # 设置 y 轴为对数刻度
plt.xticks(rotation=90)
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.savefig(
    f'./Relative Error Comparison for Different Network Structures and Models.pdf')
plt.show()