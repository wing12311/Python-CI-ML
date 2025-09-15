import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体也设为Times风格
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ---------------------------- 用户自定义参数 ----------------------------
# 字体参数（根据需求直接修改数值）
LABEL_FONTSIZE = 30    # 坐标轴标签（特征名称）字体大小
ANNOT_FONTSIZE = 12    # 热图单元格内数值字体大小
TITLE_FONTSIZE = 14    # 标题字体大小
CBAR_FONTSIZE = 10     # 颜色条标签字体大小

# 布局参数
FIG_WIDTH = 10         # 画布宽度（英寸）
FIG_HEIGHT = 5       # 画布高度（英寸）
ROTATION_ANGLE = 90    # X轴标签旋转角度（0~90）

# ---------------------------- 数据读取 ----------------------------
df = pd.read_csv(r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改1.csv", encoding='gbk')
df.dropna(inplace=True)

# ---------------------------- 绘图 ----------------------------
# 自定义颜色
# ---------------------------- 绘图参数优化 ----------------------------
# 自定义颜色（根据你的热力图蓝色系）
colors = ['#7ac3df','#ECF7FF',  # 极浅蓝色 - 用于低相关性 (0)
    '#C1E1FF',  #
    '#77C0FF',  # 中等蓝色 - 用于中等相关性
    '#2E9AFF',  #
    '#0039A6']
n_bins = 100  # 平滑过渡
cmap_high_contrast = LinearSegmentedColormap.from_list("custom_high_contrast", colors)

# 绘制热力图
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    df.corr(),
    cmap=cmap_high_contrast,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 12, "color": "black"},
    linewidths=0.3,
    square=True,
    cbar_kws={
        "shrink": 1,
        "ticks": [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "orientation": "vertical",  # 根据你的数据范围设置
    },
    vmin=-0.2,
    vmax=1.0

)

# ---------------------------- 通过colorbar对象单独设置字体（新增部分） ----------------------------
cbar = ax.collections[0].colorbar  # 获取colorbar对象
cbar.ax.tick_params(labelsize=15)  # ⚡ 这里控制刻度标签字体大小
plt.xticks(
    rotation=45,
    ha="right",  # 对齐到右侧边缘
    fontsize=15,  # 字体大小
    fontfamily="Arial",
    fontweight="bold",
fontname="Times New Roman"
)
plt.yticks(
    rotation=0,  # Y轴标签保持水平
    fontsize=15,
    fontfamily="Arial",
    fontweight="bold",
fontname="Times New Roman"
)
# 保存设置
SAVE_DIRECTORY = r"C:\Users\s1769\Desktop\依托大的"# 可修改为任何路径
FIGURE_NAME = "Correlation_Heatmap"               # 图片名称
SAVE_FORMATS = ["PDF"]              # 支持多种格式同时保存
DPI = 600                                         # 高质量分辨率
plt.show()