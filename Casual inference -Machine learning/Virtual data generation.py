# -*- coding: utf-8 -*-
"""多目标虚拟数据生成专用脚本"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import smogn
# 配置环境
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

def install_package(package, import_name=None):
    """自动安装依赖包"""
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"正在安装 {package}...")
        os.system(f"{sys.executable} -m pip install {package}")

# 检查依赖
install_package("smogn")

def generate_virtual_data(
        input_path,
        target_cols,
        output_name="生成样本.csv",  # 固定输出文件名
        target_total=600,
        random_state=42
):
    """
    虚拟数据生成主函数
    参数:
        input_path: 原始数据文件路径
        target_cols: 目标列名列表，如 ['NH4re', 'TNre']
        output_name: 输出文件名（固定）
        target_total: 期望总样本量
        random_state: 随机种子
    """
    # 读取数据
    try:
        raw_df = pd.read_csv(input_path, encoding='gbk')
        print(f"[成功] 已读取数据: {input_path}")
        print(f"原始维度: {raw_df.shape}")
    except Exception as e:
        print(f"[错误] 文件读取失败: {str(e)}")
        return

    # 数据预处理
    df = raw_df.dropna().reset_index(drop=True)
    if len(df) < 10:
        print("[警告] 有效数据不足，无法生成虚拟样本")
        return
    print(f"清洗后有效样本: {len(df)}")

    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    def quality_filter(data):
        """数据质量过滤"""
        valid = data[
            (data[target_cols[0]].between(
                data[target_cols[0]].quantile(0.02),
                data[target_cols[0]].quantile(0.98)
            )) &
            (data[target_cols[1]].between(
                data[target_cols[1]].quantile(0.02),
                data[target_cols[1]].quantile(0.98)
            )) &
            (data['FT'] <= data['high']) &
            (data[['P', 'F', 'A', 'B']].sum(axis=1) <= 100) &
            (data[['S', 'Fe', 'Mn']].sum(axis=1) <= 100) &
            ((data['FSP'] + 0.5 * data['FT']) < 2 * data['high']) &
            (data[numeric_cols] >= 0).all(axis=1)
            ]
        print(f"过滤后保留样本: {len(valid)}/{len(data)}")
        return valid

    # 训练生成模型
    print("正在训练贝叶斯混合模型...")
    try:
        bgmm = BayesianGaussianMixture(
            n_components=min(5, len(df) // 10),
            covariance_type='tied',
            reg_covar=1e-3,
            random_state=random_state
        )
        bgmm.fit(df[numeric_cols])
    except Exception as e:
        print(f"[错误] 模型训练失败: {str(e)}")
        return

    # 生成逻辑
    def generate_samples():
        """多策略样本生成"""
        # 计算生成需求
        existing = len(df)
        remaining = max(target_total - existing, 0)
        if remaining == 0:
            print("[提示] 样本量已达标，无需生成")
            return df

        # 分配生成策略
        synth_samples = {
            'smote': int(remaining * 0.7),
            'gmm': remaining - int(remaining * 0.7)
        }
        print(f"生成计划: SMOTE {synth_samples['smote']} | GMM {synth_samples['gmm']}")

        # SMOTE生成
        def safe_smote(data, n_samples):
            try:
                data_numeric = data[numeric_cols]
                synthetic = smogn.smoter(
                    data=data_numeric,
                    y=target_cols[0],
                    k=7,
                    pert=0.2,
                    samp_method='extreme',
                    rel_coef=0.6,
                    rel_method='auto'
                )
                return synthetic.sample(
                    n=min(n_samples, len(synthetic)),
                    replace=len(synthetic) < n_samples,
                    random_state=random_state
                )
            except Exception as e:
                print(f"[警告] SMOTE生成失败: {str(e)}")
                return pd.DataFrame(columns=numeric_cols)

        # GMM生成
        def safe_gmm(n_samples):
            try:
                samples = bgmm.sample(n_samples * 2)[0]
                gmm_df = pd.DataFrame(samples, columns=numeric_cols)
                return gmm_df.sample(
                    n=min(n_samples, len(gmm_df)),
                    replace=len(gmm_df) < n_samples,
                    random_state=random_state
                )
            except Exception as e:
                print(f"[警告] GMM生成失败: {str(e)}")
                return pd.DataFrame(columns=numeric_cols)

        # 执行生成
        print("正在生成SMOTE样本...")
        smote_df = safe_smote(df, synth_samples['smote'])
        print("正在生成GMM样本...")
        gmm_df = safe_gmm(synth_samples['gmm'])

        # 合并数据
        combined = pd.concat([df, smote_df, gmm_df], ignore_index=True)

        # 过滤合并后的样本
        filtered_final = quality_filter(combined)

        print(f"最终生成样本构成（过滤后）: {len(filtered_final)}个样本")
        return filtered_final

    def save_filtered_samples(virtual_df):
        """保存过滤后的生成样本到桌面"""
        save_path_filtered = os.path.join(os.path.expanduser('~'), 'Desktop', '过滤后生成样本.csv')
        try:
            virtual_df.to_csv(save_path_filtered, index=False)
            print(f"\n[成功] 过滤后生成样本已保存: {save_path_filtered}")
        except Exception as e:
            print(f"[错误] 保存过滤后生成样本失败: {str(e)}")

    # 调用生成虚拟数据函数并保存
    virtual_df = generate_samples()
    save_filtered_samples(virtual_df)

# 示例配置
CONFIG = {
    "input_path": r"C:\Users\s1769\Desktop\真实数据1.csv",  # 替换为您的输入文件路径
    "target_cols": ['NH4re', 'TNre'],  # 替换为您的目标列
    "target_total": 600,
    "random_state": 42
}

# 执行生成
print("=" * 50)
print("虚拟数据生成器启动")
generate_virtual_data(**CONFIG)
print("=" * 50)
