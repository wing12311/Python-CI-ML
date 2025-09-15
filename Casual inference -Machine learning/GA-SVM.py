import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pygad
from sklearn.linear_model import LinearRegression

# ================= 中文字体配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
file_path = r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv"
data = pd.read_csv(file_path, encoding='gbk')

# 2. 打乱数据
data = shuffle(data, random_state=42).reset_index(drop=True)

# 3. 提取特征和目标
target = 'TNre(%)'
X = data.drop(columns=[target])
y = data[target]

# 4. 划分数据
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 5. 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


# 6. 交叉验证函数
def cross_val_score_svr(x, y, C, gamma, epsilon, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    for train_idx, val_idx in kf.split(x):
        x_train_cv, x_val_cv = x[train_idx], x[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(x_train_cv, y_train_cv)
        preds = model.predict(x_val_cv)
        r2_scores.append(r2_score(y_val_cv, preds))
    return np.mean(r2_scores)


# 7. 遗传算法优化函数
def optimize_svm_with_pygad(x, y):
    def fitness_func(ga, solution, solution_idx):
        C, gamma, epsilon = solution
        score = cross_val_score_svr(x, y, C, gamma, epsilon, n_splits=5)
        return score,

    gene_space = [
        {'low': 0.1, 'high': 100},  # C
        {'low': 0.0001, 'high': 10},  # gamma
        {'low': 0.0001, 'high': 0.5}  # epsilon
    ]

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=3,
        gene_space=gene_space,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10
    )

    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()
    return solution, fitness[0]


# 8. 参数优化
best_params, cv_r2_mean = optimize_svm_with_pygad(
    X_train_scaled,
    pd.Series(y_train_scaled.ravel())
)

# 9. 模型训练
best_svr = SVR(
    kernel='rbf',
    C=best_params[0],
    gamma=best_params[1],
    epsilon=best_params[2]
).fit(X_train_scaled, y_train_scaled.ravel())


# 10. 模型评估函数
def evaluate_model(model, X, y, set_name):
    pred_scaled = model.predict(X)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

    r2 = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)

    print(f"\n{set_name}集评估结果:")
    print(f"R²: {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return actual, pred


# 11. 各数据集评估
train_true, train_pred = evaluate_model(best_svr, X_train_scaled, y_train_scaled, "训练")
val_true, val_pred = evaluate_model(best_svr, X_val_scaled, y_val_scaled, "验证")
test_true, test_pred = evaluate_model(best_svr, X_test_scaled, y_test_scaled, "测试")

# 12. 可视化函数
def plot_regression_with_bands_all_tests(actuals_list, predicted_list, target_name, cv_r2):
    import scipy.stats as stats
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
    ax.set_facecolor('white')

    # 1. 计算所有实际值的最大最小值，用于归一化
    combined_actuals = np.concatenate(actuals_list)
    min_val = combined_actuals.min()
    max_val = combined_actuals.max()

    # 2. 归一化函数
    def normalize(data):
        return (data - min_val) / (max_val - min_val)

    # 3. 归一化所有实际值和预测值
    normalized_actuals = [normalize(act) for act in actuals_list]
    normalized_preds = [normalize(pred) for pred in predicted_list]

    # 4. 绘制散点
    for (actual, pred), (name, style) in zip(zip(normalized_actuals, normalized_preds), {
        'Train': {'color': '#2D80B9', 'label': 'Training Set'},
        'Validation': {'color': '#ED7B7B', 'label': 'Validation Set'},
        'Test': {'color': '#00B050', 'label': 'Test Set'}
    }.items()):
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

    # 5. 线性拟合（用归一化数据）
    all_actuals_norm = np.concatenate(normalized_actuals)
    all_preds_norm = np.concatenate(normalized_preds).reshape(-1, 1)

    reg = LinearRegression().fit(all_preds_norm, all_actuals_norm)
    x_fit = np.linspace(0, 1, 100)
    y_fit = reg.predict(x_fit.reshape(-1, 1))
    ax.plot(x_fit, y_fit, 'k--', lw=2, label='Linear Fit')

    # 6. 计算置信区间和预测区间
    n = len(all_actuals_norm)
    dof = n - 2
    t_val = stats.t.ppf(0.975, dof)

    residuals = all_actuals_norm - reg.predict(all_preds_norm)
    residual_std = np.sqrt(np.sum(residuals ** 2) / dof)

    x_mean = np.mean(all_preds_norm)
    ssx = np.sum((all_preds_norm - x_mean) ** 2)

    ci = t_val * residual_std * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ssx)
    pi = t_val * residual_std * np.sqrt(1 + 1 / n + (x_fit - x_mean) ** 2 / ssx)

    # 7. 添加置信区间和预测区间
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci,
                     color='gray', alpha=0.3, label='95% Confidence Band')
    ax.fill_between(x_fit, y_fit - pi, y_fit + pi,
                     color='gray', alpha=0.1, label='95% Predicted Band')

    # 8. 设置图标题和坐标轴
    ax.set_title(f"", fontsize=20)
    ax.set_xlabel(f'', fontsize=20, fontweight='bold')
    ax.set_ylabel(f'', fontsize=20, fontweight='bold')
    ax.set_xlim(-0.199, 1.2)
    ax.set_ylim(-0.199, 1.2)

    # 9. 刻度
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
            fontsize=25, bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3'))

    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=15)
    plt.grid(False)
    plt.tight_layout()

# 13. 调用绘图函数（只调用一次）
actuals_list = [train_true, val_true, test_true]
predicted_list = [train_pred, val_pred, test_pred]

plot_regression_with_bands_all_tests(actuals_list, predicted_list, target_name='TNre(%)', cv_r2=cv_r2_mean)

# 14. 显示图形
plt.show()
