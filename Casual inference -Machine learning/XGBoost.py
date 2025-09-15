import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator
from sklearn.utils import shuffle
import joblib
import os
from xgboost import XGBRegressor
from scipy.stats import linregress

# 设置全局样式
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 1. 读取数据
df = pd.read_csv(r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv", encoding='gbk')
print('原始数据大小:', df.shape)
df.dropna(inplace=True)
print('清洗后数据大小:', df.shape)
print('数据预览:')
print(df.head())

# 2. 打乱数据
data = shuffle(df, random_state=42).reset_index(drop=True)

# 3. 准备特征和目标变量
y_col = 'TNre(%)'  # 根据实际数据调整
xx = data.drop(columns=[y_col])
yy = data[y_col]

# 4. 数据标准化
scaler_X = StandardScaler()
scaler_y = MinMaxScaler(feature_range=(0, 100))  # 缩放到0-100范围

xx_scaled = scaler_X.fit_transform(xx)
yy_scaled = scaler_y.fit_transform(yy.values.reshape(-1, 1)).ravel()

# 5. 数据集划分
x_train_full, x_test, y_train_full, y_test, train_idx_full, test_idx = train_test_split(
    xx_scaled, yy_scaled, np.arange(len(yy)),
    test_size=0.2, shuffle=True, random_state=42
)

x_train, x_val, y_train, y_val, train_idx, val_idx = train_test_split(
    x_train_full, y_train_full, train_idx_full,
    test_size=0.2, shuffle=True, random_state=42
)

print('\n样本分布:')
print(f"总样本数: {len(yy)}")
print(f"特征数量: {xx.shape[1]}")
print(f"训练集: {len(y_train)} ({len(y_train) / len(yy):.1%})")
print(f"验证集: {len(y_val)} ({len(y_val) / len(yy):.1%})")
print(f"测试集: {len(y_test)} ({len(y_test) / len(yy):.1%})")

# 6. 训练模型
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0.2,
    reg_alpha=30,
    reg_lambda=40,
    random_state=42
)
model.fit(x_train_full, y_train_full)  # 使用全部训练数据

# 创建模型保存目录（若不存在）
model_dir = r"C:\Users\s1769\PycharmProjects\PythonProject1\Models"
os.makedirs(model_dir, exist_ok=True)

# 保存模型
joblib.dump(model, os.path.join(model_dir, 'xgb_model_2.joblib'))

# 7. 预测
y_pred_test = model.predict(x_test)

# 8. 交叉验证R²
model_cv = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0.2,
    reg_alpha=30,
    reg_lambda=40,
    random_state=42
)

cv_scores = cross_val_score(
    estimator=model_cv,
    X=x_train_full,
    y=y_train_full,
    cv=5,
    scoring='r2'
)
cv_r2_mean = cv_scores.mean()

# 9. 反缩放所有子集的实际值和预测值
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

y_pred_train_orig = scaler_y.inverse_transform(model.predict(x_train).reshape(-1, 1)).ravel()
y_pred_val_orig = scaler_y.inverse_transform(model.predict(x_val).reshape(-1, 1)).ravel()
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

# 10. 计算全局范围（在反缩放后）
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

# 13. 目标变量名
target_name = "NH4re(%)"

def evaluate_model(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{set_name}集评估:")
    print(f"R²: {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return r2, rmse, mae

# 打印评估结果
print("\n模型性能评估:")
_ = evaluate_model(y_train_orig, y_pred_train_orig, "训练")
_ = evaluate_model(y_val_orig, y_pred_val_orig, "验证")
_ = evaluate_model(y_test_orig, y_pred_test_orig, "测试")

# 14. 定义绘图函数（带白色背景）
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
        'Train': {'color': '#5A9AD9', 'label': 'Training Set', 'fontsize': 20},
        'Validation': {'color': '#ED7B7B', 'label': 'Validation Set', 'fontsize': 20},
        'Test': {'color': '#F5AA83', 'label': 'Test Set', 'fontsize': 20}
    }

    # 绘制散点
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
    all_preds = np.concatenate(predicted_list)
    slope, intercept, r_value, p_value, std_err = linregress(all_preds, all_actuals)
    x_fit = np.linspace(0, 1, 100)
    y_fit = slope * x_fit + intercept

    # 计算置信区间和预测区间
    n = len(all_actuals)
    dof = n - 2
    t_val = stats.t.ppf(0.975, dof)
    y_pred_line = slope * all_preds + intercept
    residuals = all_actuals - y_pred_line
    residual_std = np.sqrt(np.sum(residuals ** 2) / dof)
    x_mean = np.mean(all_preds)
    ssx = np.sum((all_preds - x_mean) ** 2)

    ci = t_val * residual_std * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ssx)
    pi = t_val * residual_std * np.sqrt(1 + 1 / n + (x_fit - x_mean) ** 2 / ssx)

    # 绘制线性拟合线和区间
    ax.plot(x_fit, y_fit, 'k--', lw=2, label='Linear Fit')

    ax.fill_between(x_fit, y_fit - pi, y_fit + pi, color='gray', alpha=0.3, label='95% Prediction bond')

    # 设置标题和标签

    ax.set_xlim(-0.199, 1.2)
    ax.set_ylim(-0.199, 1.2)

    # 设置刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, labelsize=15)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=2)

    # 添加文本（R²）
    r2_text = f"5-Fold CV\nR²= {cv_r2:.2f}"
    ax.text(0.95, 0.05, r2_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=25, bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3'))

    # 图例（只调用一次）
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=15)

    plt.tight_layout()
    plt.show()

# 15. 调用绘图
plot_regression_with_bands_all_tests(actuals_list, predicted_list, target_name, cv_r2_mean)

y_pred_train_norm

