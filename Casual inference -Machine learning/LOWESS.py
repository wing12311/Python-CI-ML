import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from networkx.classes.filters import show_nodes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as mticker
from statsmodels.nonparametric.smoothers_lowess import lowess
import time
import os
import joblib
from matplotlib.colors import LinearSegmentedColormap  # 导入 LinearSegmentedColormap
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 50,
    'ytick.labelsize': 50,
    'font.weight': 'bold'
})
# 加载数据
def load_data():
    """加载并预处理数据"""
    try:
        data_path = r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv"

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        df = pd.read_csv(data_path, encoding='gbk')
        print(f"原始数据集形状: {df.shape}")

        # 检查数据
        if df.empty:
            raise ValueError("数据加载失败: 数据集为空")

        # 处理缺失值
        original_count = len(df)
        df.dropna(inplace=True)
        print(f"清理后有效样本量: {len(df)} (删除 {original_count - len(df)} 个无效样本)")

        return df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        return None


# 直接使用用户提供的参数训练模型
def train_xgboost_model(X_train, y_train):
    """使用用户提供的参数训练XGBoost模型"""
    print("\n=== 开始训练XGBoost模型（使用用户参数） ===")

    start_time = time.time()

    # 使用用户提供的参数
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

    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train)

    end_time = time.time()
    print(f"模型训练完成! 耗时: {end_time - start_time:.2f}秒")

    # 打印模型参数
    print("\n模型参数配置:")
    params = model.get_params()
    for param, value in params.items():
        print(f"{param}: {value}")

    return model


# 评估模型性能
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    print("\n=== 模型评估 ===")

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"均方误差(MSE): {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f}")
    print(f"决定系数(R²): {r2:.4f}")

    # 绘制实际值 vs 预测值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('实际值 vs 预测值')
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.close()

    # 保存预测结果
    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': y_test - y_pred
    })
    pred_df.to_csv('model_predictions.csv', index=False)

    return rmse, r2


# 特征影响分析
def analyze_feature_impact(model, feature, X_data):
    """
    分析单个特征对预测目标的影响
    返回: 临界点列表
    """
    try:
        # 确保特征存在
        if feature not in X_data.columns:
            available_features = ", ".join(X_data.columns)
            raise ValueError(f"特征 '{feature}' 不在数据集中。可用特征: {available_features}")

        # 获取特征值
        feature_values = X_data[feature].values

        # 计算SHAP值
        print(f"\n计算特征 '{feature}' 的SHAP值...")
        explainer = shap.TreeExplainer(model)
        full_shap_values = explainer.shap_values(X_data)

        # 提取目标特征的SHAP值
        feature_idx = list(X_data.columns).index(feature)
        feature_shap = full_shap_values[:, feature_idx]

        # 创建图形
        plt.figure(figsize=(13, 8))

        # =================== 新增的坐标轴设置 ===================
        # 计算坐标轴限制值
        x_min =0
        x_max = np.max(feature_values) * 1.1

        # 设置坐标轴范围
        plt.xlim(x_min, x_max)
        plt.ylim(np.min(feature_shap) * 1.2, np.max(feature_shap) * 1.3)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(30))
        plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(2))
        # 设置坐标轴标签样式
        plt.tick_params(axis='both', which='major', direction='in',
                        labelsize=25,
                        length=6,  # 主刻度长度
                        width=2,  # 刻度线宽度
                        pad=10)
        #plt.xticks(fontsize=25)
        #plt.yticks(fontsize=25)

        # =================== 坐标轴设置结束 ===================
        colors = [(0.0, '#FFDD8E'),
                  (0.5, '#FFFFFF'),
                  (1.0, '#7AC3DF')]

        cmap_name = 'SciTech'
        sci_tech_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

        sc = plt.scatter(feature_values, feature_shap,
                         c=feature_values,
                         cmap=sci_tech_cmap,  # 关键修正: 直接传递对象
                         vmin=np.min(feature_values),
                         vmax=np.max(feature_values),
                         alpha=0.85,
                         s=55,
                         edgecolor='w',
                         linewidth=0.8)

        # 添加颜色条
        cbar = plt.colorbar(sc)
        cbar.set_label('Feature Value', rotation=270, labelpad=25, fontsize=30, fontweight='bold')
        cbar.ax.tick_params(labelsize=50, length=0)
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(50)  # 设置字体大小
            label.set_fontweight('bold')
        cbar.ax.set_yticklabels([])  # 移除刻度标签

        # 在顶部添加"HIGH"标注
        cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes,
                     fontsize=20, fontweight='bold',
                     ha='center', va='bottom')

        # 在底部添加"LOW"标注
        cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes,
                     fontsize=20, fontweight='bold',
                     ha='center', va='top')

        # LOWESS拟合
        lowess_fit = lowess(feature_shap, feature_values, frac=0.2)
        plt.plot(lowess_fit[:, 0], lowess_fit[:, 1],
                 color='#70CDBE', linewidth=3,
                 label='LOWESS')

        # 标记SHAP=0区域
        plt.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

        # 寻找临界点
        def find_crossings(x, y):
            crossings = []
            for i in range(1, len(y)):
                if (y[i - 1] < 0 and y[i] > 0) or (y[i - 1] > 0 and y[i] < 0):
                    # 线性插值求交点
                    x0 = x[i - 1] - (x[i] - x[i - 1]) * y[i - 1] / (y[i] - y[i - 1])
                    crossings.append(round(x0, 3))
            return crossings

        thresholds = find_crossings(lowess_fit[:, 0], lowess_fit[:, 1])

        # 标注临界点
        for i, th in enumerate(thresholds):
            plt.axvline(x=th, color='grey', linestyle='--', alpha=0.8)

        plt.xlabel(feature, fontsize=35, fontweight='bold')
        plt.ylabel('SHAP Value', fontsize=25, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{feature}_impact.png', dpi=600, bbox_inches='tight')
        plt.show()

        # 保存SHAP值和特征值
        shap_df = pd.DataFrame({
            'Feature_Value': feature_values,
            'SHAP_Value': feature_shap,
            'LOWESS_Fit': lowess_fit[:, 1]
        })
        shap_df.to_csv(f'shap_values_{feature}.csv', index=False)

        return thresholds

    except Exception as e:
        print(f"特征分析失败: {str(e)}")
        return []


# 业务解读函数
def business_interpretation(feature, thresholds, feature_values):
    """基于分析结果生成业务解读"""
    try:
        print(f"\n=== {feature} 影响分析业务报告 ===")
        print(f"特征值范围: {np.min(feature_values):.3f} - {np.max(feature_values):.3f}")
        print(f"特征值中位数: {np.median(feature_values):.3f}")

        if not thresholds:
            print("\n分析结论: 该特征对预测值的影响方向一致，未检测到明显的临界点")
            print("建议操作: 保持当前水平的特征值可能获得稳定预测效果")
            return

        # 排序临界点
        thresholds_sorted = sorted(thresholds)
        print(f"\n检测到 {len(thresholds_sorted)} 个关键临界点:")

        # 计算每个区域的样本分布
        regions = []
        prev_th = np.min(feature_values) - 0.001
        for i, th in enumerate(thresholds_sorted):
            region_samples = np.sum((feature_values >= prev_th) & (feature_values < th))
            regions.append({
                "start": prev_th,
                "end": th,
                "samples": region_samples,
                "effect": "正" if (i % 2 == 0) else "负"
            })
            prev_th = th

        # 添加最后一个区域
        region_samples = np.sum(feature_values >= prev_th)
        regions.append({
            "start": prev_th,
            "end": np.max(feature_values) + 0.001,
            "samples": region_samples,
            "effect": "正" if (len(thresholds_sorted) % 2 == 0) else "负"
        })

        # 打印区域分析
        for i, region in enumerate(regions):
            print(f"\n区域 {i + 1}: {region['start']:.3f} - {region['end']:.3f}")
            print(f"  - 影响方向: {region['effect']}向")
            print(f"  - 样本数量: {region['samples']} ({region['samples'] / len(feature_values) * 100:.1f}%)")

            # 建议
            if region['samples'] < 10:
                print("  - ⚠️ 注意: 此区域样本较少，结论可信度低")

        # 识别最稳定区域
        optimal_regions = [r for r in regions if r['samples'] > 0.1 * len(feature_values)]
        if optimal_regions:
            most_stable = max(optimal_regions, key=lambda x: x['samples'])
            print(f"\n最稳定区域: {most_stable['start']:.3f} - {most_stable['end']:.3f}")
            print(f"  - 包含样本: {most_stable['samples']} ({most_stable['samples'] / len(feature_values) * 100:.1f}%)")
            print(f"  - 影响方向: {most_stable['effect']}向")

        print("\n建议操作:")
        if len(regions) > 1:
            print("1. 将特征控制在主要影响区间:")
            for r in regions:
                if r['samples'] > 0.15 * len(feature_values):
                    print(f"   - {r['start']:.3f}-{r['end']:.3f} ({r['effect']}向影响)")

            if any(r['samples'] < 10 for r in regions):
                print("2. 对于样本稀少的区域，收集更多数据进行验证")
        else:
            print("1. 维持当前特征值范围可获得稳定预测结果")

        print("3. 结合其他特征综合分析，避免单一特征决策")
        print("4. 通过临界点值调整工艺参数")

    except Exception as e:
        print(f"业务解读失败: {str(e)}")


# 主执行函数
def main():
    # 加载数据
    print("加载数据...")
    df = load_data()
    if df is None:
        print("无法加载数据，程序终止")
        return

    # 分割特征和目标（假设最后一列是目标变量）
    if len(df.columns) < 2:
        print("错误: 数据集需要至少包含特征和目标两列数据")
        return

    target_column = df.columns[-1]
    print(f"目标变量: '{target_column}'")

    # 获取特征和目标数据
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    print(f"特征数量: {X.shape[1]}")

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 训练模型（使用用户提供的参数）
    model = train_xgboost_model(X_train, y_train)

    # 保存模型
    joblib.dump(model, 'xgb_model_final.joblib')
    print("模型已保存为 'xgb_model_final.joblib'")

    # 评估模型
    rmse, r2 = evaluate_model(model, X_test, y_test)

    # 在完整数据集上重新训练模型（使用所有可用数据）
    print("\n=== 使用全部数据重新训练模型（提高泛化能力） ===")
    model_full = train_xgboost_model(X, y)
    joblib.dump(model_full, 'xgb_model_full.joblib')
    print("完整模型已保存为 'xgb_model_full.joblib'")

    # 选择分析的特征 - 使用用户关心的特征
    feature_to_analyze = ("Porosity")  # 直接指定为"Porosity"

    if feature_to_analyze not in X.columns:
        print(f"警告: 特征 '{feature_to_analyze}' 不存在，使用第一个特征替代")
        feature_to_analyze = X.columns[0]

    print(f"\n分析特征: '{feature_to_analyze}'")

    # 执行特征影响分析（使用完整数据集训练的模型）
    thresholds = analyze_feature_impact(model_full, feature_to_analyze, X)

    # 业务解读
    business_interpretation(feature_to_analyze, thresholds, X[feature_to_analyze].values)


# 执行主函数
if __name__ == "__main__":
    print("=== 开始分析 ===")
    start_time = time.time()
    main()
    end_time = time.time()
    print("结果文件:")
    print("- 模型文件: xgb_model_full.joblib")
    print(f"- 特征影响图: {feature_to_analyze if 'feature_to_analyze' in locals() else 'feature'}_impact.svg")
    print("- 模型预测结果: model_predictions.csv")
    plt.show()