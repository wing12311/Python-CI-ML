import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats

# 数据读取与预处理
df = pd.read_csv(r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv", encoding='gbk')
print('原始数据大小:', df.shape)
df.dropna(inplace=True)
print('清洗后数据大小:', df.shape)
print('数据预览:')
print(df.head())

# 定义特征和目标变量（假设最后一列是目标变量）
target_name = df.columns[-1]  # 获取目标变量名称
xx = df.iloc[:, :-1]  # 所有特征列
yy = df.iloc[:, -1]  # 目标列

# 数据标准化
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()

xx_scaled = scaler_X.fit_transform(xx)
yy_scaled = scaler_y.fit_transform(yy.values.reshape(-1, 1)).ravel()


# 数据集划分（训练集60%，验证集20%，测试集20%）
def data_split(x, y, test_size=0.2, val_size=0.2, random_state=42):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


x_train, x_val, x_test, y_train, y_val, y_test = data_split(xx_scaled, yy_scaled)

# 打印数据集信息
print('\n数据集分布:')
print(f"总样本数: {xx.shape[0]}")
print(f"特征数量: {xx.shape[1]}")
print(f"训练集: {x_train.shape[0]} ({(x_train.shape[0] / xx.shape[0]) * 100:.1f}%)")
print(f"验证集: {x_val.shape[0]} ({(x_val.shape[0] / xx.shape[0]) * 100:.1f}%)")
print(f"测试集: {x_test.shape[0]} ({(x_test.shape[0] / xx.shape[0]) * 100:.1f}%)")

# 随机森林模型训练
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_features=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    bootstrap=True
)

rf.fit(x_train, y_train)

# 交叉验证
scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')
print("\n交叉验证结果:")
print(f"平均R2分数: {scores.mean():.4f} (±{scores.std():.4f})")

# 模型预测
y_pred_train = rf.predict(x_train)
y_pred_val = rf.predict(x_val)
y_pred_test = rf.predict(x_test)

# 逆标准化还原数据
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
y_pred_val_orig = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
#  计算全局范围（在反缩放后）
all_actuals = np.concatenate([y_train_orig, y_val_orig, y_test_orig])
all_preds = np.concatenate([y_pred_train_orig, y_pred_val_orig, y_pred_test_orig])

min_val = np.min(all_actuals)
max_val = np.max(all_actuals)

# 11. 归一化到0-1
y_train_norm = (y_train_orig - min_val) / (max_val - min_val)
y_pred_train_norm = (y_pred_train_orig - min_val) / (max_val - min_val)

y_val_norm = (y_val_orig - min_val) / (max_val - min_val)
y_pred_val_norm = (y_pred_val_orig - min_val) / (max_val - min_val)

y_test_norm = (y_test_orig - min_val) / (max_val - min_val)
y_pred_test_norm = (y_pred_test_orig - min_val) / (max_val - min_val)

# 12. 组成列表（归一化后）
actuals_list = [y_train_norm, y_val_norm, y_test_norm]
predicted_list = [y_pred_train_norm, y_pred_val_norm, y_pred_test_norm]

# 修改评估函数
def evaluate_model(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # 手动计算平方根
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{set_name}集评估:")
    print(f"R²: {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return r2, rmse, mae

# 打印评估结果
print("\n模型性能评估:")
_ = evaluate_model(y_train_orig, y_pred_train_orig, "训练")
_ = evaluate_model(y_val_orig, y_pred_val_orig, "验证")
_ = evaluate_model(y_test_orig, y_pred_test_orig, "测试")


def plot_regression_with_bands_all_tests(actuals_list, predicted_list, target_name, cv_r2):
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.2
    })

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置背景色为白色

    dataset_config = {
    'Train': {'color': '#2D80B9', 'label': 'Training Set', 'fontsize': 20},
    'Validation': {'color': '#ED7B7B', 'label': 'Validation Set', 'fontsize': 20},
    'Test': {'color': '#00B050', 'label': 'Test Set', 'fontsize': 20}
}

    for (actual, pred), (name, style) in zip(zip(actuals_list, predicted_list), dataset_config.items()):
        ax.scatter(
            pred, actual,
            c=style['color'],
            marker='o',
            s=60,
            edgecolors='white',
            linewidths=0.4,
            alpha=0.9,
            label=style['label']
        )

    # 线性拟合（整体）
    all_actuals = np.concatenate(actuals_list)
    all_preds = np.concatenate(predicted_list).reshape(-1, 1)
    reg = LinearRegression().fit(all_preds, all_actuals)
    x_fit = np.linspace(all_preds.min(), all_preds.max(), 100)
    y_fit = reg.predict(x_fit.reshape(-1, 1))
    ax.plot(x_fit, y_fit, 'k--', lw=2, label='Linear Fit')
    ax.legend(fontsize=20)
    # 计算置信区间和预测区间
    n = len(all_actuals)
    dof = n - 2
    t_val = stats.t.ppf(0.975, dof)
    residuals = all_actuals - reg.predict(all_preds)
    residual_std = np.sqrt(np.sum(residuals ** 2) / dof)
    x_mean = np.mean(all_preds)
    ssx = np.sum((all_preds - x_mean) ** 2)

    ci = t_val * residual_std * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ssx)
    pi = t_val * residual_std * np.sqrt(1 + 1 / n + (x_fit - x_mean) ** 2 / ssx)

    # 添加置信区间
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color='gray', alpha=0.3, label='95% Confidence Interval')

    # 添加预测区间
    ax.fill_between(x_fit, y_fit - pi, y_fit + pi, color='gray', alpha=0.3, label='Prediction Interval')
    # 图标题

    ax.set_xlabel('', fontsize=20, fontweight='bold')
    ax.set_ylabel('', fontsize=20, fontweight='bold')
    ax.set_xlim(-0.19, 1.2)
    ax.set_ylim(-0.19, 1.2)

    # 刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, labelsize=15)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=2)

    r2_text = f"5-Fold CV\nR²= {cv_r2:.2f}"
    ax.text(0.95, 0.05, r2_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=26, bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3'))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
    })
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=15)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 13. 调用绘图
cv_r2_mean = scores.mean()

# 实际值（逆标准化）
actuals_list = [y_train_norm, y_val_norm, y_test_norm]
predicted_list = [y_pred_train_norm, y_pred_val_norm, y_pred_test_norm]

# 调用绘图
plot_regression_with_bands_all_tests(actuals_list, predicted_list, target_name='TNre(%)', cv_r2=cv_r2_mean)