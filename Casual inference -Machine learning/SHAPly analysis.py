import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap
from joblib import load
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# 修正rcParams设置 - 移除多余的逗号
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2,  # 边框宽度
    "xtick.major.width": 1.5,  # X轴主刻度线宽度
    "ytick.major.width": 1.5,  # Y轴主刻度线宽度
    "xtick.major.size": 10,  # X轴主刻度长度
    "ytick.major.size": 5,  # Y轴主刻度长度
})


# 逆缩放函数
def inverse_scaler(y_scaled, y_true):
    scaler = MinMaxScaler()
    scaler.fit(np.reshape(y_true, (len(y_true), 1)))
    y_inverse = scaler.inverse_transform(y_scaled.reshape(-1, 1))
    return y_inverse


# 读取数据
df = pd.read_csv(r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改2.csv", encoding='gbk')
print('original datasize:', df.shape)
df.dropna(inplace=True)
print('adjusted datasize:', df.shape)

# 特征和目标
xx = df.iloc[:, 0:-1]
yy = df.iloc[:, -1]

# 标准化
zscore = StandardScaler()
scaler = MinMaxScaler()
xx_scaled = zscore.fit_transform(xx)
yy_scaled = scaler.fit_transform(np.reshape(yy.values, (len(yy), 1)))


# 数据分割函数
def split(xx_scaled, yy_scaled):
    x_train, x_test_val, y_train, y_test_val = train_test_split(xx_scaled, yy_scaled, shuffle=True, test_size=0.2,
                                                                random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, shuffle=True, test_size=0.2,
                                                    random_state=1)

    y2 = yy_scaled.ravel()
    ind_train = [np.argwhere(y2 == y_train[i])[0].item() for i in range(len(y_train)) if y_train[i] in y2]
    ind_test = [np.argwhere(y2 == y_test[i])[0].item() for i in range(len(y_test)) if y_test[i] in y2]
    ind_val = [np.argwhere(y2 == y_val[i])[0].item() for i in range(len(y_val)) if y_val[i] in y2]

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    y_val = y_val.ravel()

    return x_train, x_test, x_val, y_train, y_test, y_val, xx, yy, np.asarray(ind_train), np.asarray(
        ind_test), np.asarray(ind_val)


x_train, x_test, x_val, y_train, y_test, y_val, x_scaled, y_scaled, ind_train, ind_test, ind_val = split(xx_scaled,
                                                                                                         yy_scaled)

# 载入模型
model_path = r"C:\Users\s1769\PycharmProjects\PythonProject3\xgb_model_1.joblib"
loaded = load(model_path)
xgb = loaded
#%%

# 计算原始SHAP值
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(x_test)

# 特征名称
feature_names = ['Porosity', 'ED', 'AOP', 'Mn', 'S', 'Fe', 'FSP', 'High', 'C/N', 'TNinf', 'T', 'HRT', 'P', 'A', 'F',
                 'B']

# === 取消SHAP值归一化 ===
shap_values_normalized = shap_values  # 使用原始SHAP值

# 计算原始平均绝对SHAP值（重要性值）
mean_abs_shap_original = np.mean(np.abs(shap_values), axis=0)

# 特征排序索引（按重要性降序）
sorted_idx = np.argsort(mean_abs_shap_original)[::1]  # 降序排列
feature_names_sorted = [feature_names[i] for i in sorted_idx]
mean_abs_shap_sorted = mean_abs_shap_original[sorted_idx]
shap_values_sorted = shap_values_normalized[:, sorted_idx]
x_test_sorted = x_test[:, sorted_idx]

# 计算SHAP值范围（用于蜂群图）
shap_min = np.min(shap_values_sorted)
shap_max = np.max(shap_values_sorted)
shap_range = shap_max - shap_min

# 计算重要性范围（用于条形图）
importance_min = 0
importance_max = np.max(mean_abs_shap_sorted) * 1.1  # 添加10%缓冲区


# ====== 修改后的可视化函数 ====== #
def visualize_bee_swarm_only(shap_values_sorted, x_test_sorted, feature_names_sorted,
                             mean_abs_shap_sorted, shap_min, shap_max, importance_max):
    # 创建自定义颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'shap', ['#FFDD8E', '#FFFFFF', '#70CDBE'],
        N=256
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))

    # 显式设置刻度长度 - 确保生效
    ax.tick_params(axis='x', which='major', length=10, width=1.5)
    ax.tick_params(axis='y', which='major', length=5, width=1.5)

    # 为每个特征添加数据点
    feature_order = range(len(feature_names_sorted))
    for i in feature_order:
        shap_vals = shap_values_sorted[:, i]
        # 添加垂直抖动以避免点重叠
        y_jitter = np.random.uniform(i - 0.3, i + 0.3, size=len(shap_vals))

        # 获取当前特征的实际值
        feature_values = x_test_sorted[:, i]

        # 计算当前特征的最小最大值
        feature_min = np.min(feature_values)
        feature_max = np.max(feature_values)

        # 绘制点 - 使用特征值范围
        sc = ax.scatter(
            shap_vals,  # 使用原始SHAP值
            y_jitter,
            c=x_test_sorted[:, i],  # 使用原始特征值着色
            cmap=cmap,
            vmin=feature_min,  # 使用特征最小值
            vmax=feature_max,  # 使用特征最大值
            s=30,
            alpha=0.8,
            edgecolors='face'
        )

    # 设置y轴标签
    ax.set_yticks(range(len(feature_names_sorted)))
    ax.set_yticklabels(feature_names_sorted, fontsize=30, fontweight='bold')

    # 移除右侧边框和刻度
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')  # 确保刻度只在左侧

    # 显式设置Y轴刻度参数
    ax.tick_params(axis='y', which='both', direction='in', length=5, width=1.5, labelsize=25)

    # === 设置蜂群图x轴范围 ===
    # 使用计算出的SHAP值范围
    ax.set_xlim(shap_min - 0.1*shap_range, shap_max + 0.1*shap_range+10)
    ax.set_xlabel('SHAP Value', fontsize=30, fontweight='bold', labelpad=15)

    # 显式设置X轴刻度参数
    ax.tick_params(axis='x', which='both', direction='in', length=5, width=1.5, labelsize=20)

    # 添加浅蓝色背景矩形表示特征重要区间
    max_importance = importance_max
    for i in feature_order:
        # 计算当前特征的矩形宽度（基于特征重要性）
        rel_width = mean_abs_shap_sorted[i] / max_importance * 0.8  # 80%的最大宽度

        # 计算矩形位置（从左侧开始）
        rect_x = shap_min - 0.1*shap_range

        rect = plt.Rectangle(
            (rect_x, i - 0.4),  # x,y位置
            width=rel_width * (shap_max - shap_min + 0.2*shap_range),  # 宽度
            height=0.8,
            facecolor='#7AC3DF',
            alpha=0.3,
            edgecolor='none',
            zorder=-1  # 在底部
        )
        ax.add_patch(rect)

    # 添加0线
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)

    # === 在顶部添加重要性坐标轴（平均绝对SHAP值）===
    ax_top = ax.twiny()  # 共享y轴的顶部x轴
    # 设置顶部x轴范围为0到最大重要性值
    ax_top.set_xlim(0, importance_max)
    ax_top.set_xlabel('Mean Absolute SHAP Value', fontsize=30, fontweight='bold', labelpad=15)

    from matplotlib.ticker import FormatStrFormatter

    # 设置刻度参数（方向、长度、字体等）
    ax_top.tick_params(axis='x', which='both', direction='in',
                       length=5, width=1.5, labelsize=20)

    # 设置刻度标签保留1位小数
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # ======== 颜色条 ========
    cax = fig.add_axes([0.8, 0.25, 0.02, 0.5])  # [x, y, width, height]
    norm = plt.Normalize(vmin=feature_min, vmax=feature_max)  # 使用特征值范围
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Feature Value', fontsize=30, fontweight='bold', labelpad=15)

    # 关键修改：去除所有刻度线和刻度标签
    cbar.ax.yaxis.set_ticks([])  # 移除刻度线
    cbar.ax.yaxis.set_ticklabels([])  # 移除刻度标签

    # 添加高低标记
    cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=20, fontweight='bold')
    cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=20, fontweight='bold')
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    # 整体调整
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为标题和颜色条留出空间
    plt.savefig('Bee_Swarm_Only1.SVG', dpi=600, bbox_inches='tight')
    plt.close()
    print("已保存蜂群图为 Bee_Swarm_Only.SVG")


# 调用修改后的可视化函数
visualize_bee_swarm_only(shap_values_sorted, x_test_sorted, feature_names_sorted,
                          mean_abs_shap_sorted, shap_min, shap_max, importance_max)