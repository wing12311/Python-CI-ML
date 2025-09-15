# -*- coding: utf-8 -*-
"""
简化版多变量因果分析系统（只验证中介路径）
"""
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from joblib import Parallel, delayed
import logging
import os
import warnings
import statsmodels.api as sm
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')


class CausalAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.cg = None
        self.treatment_vars = []
        self.outcome_vars = []
        self.n_bootstrap = 200

    def load_and_preprocess_data(self):
        self.raw_data = pd.read_csv(
            self.data_path,
            encoding='utf-8',
            engine='python'
        )
        cols_needed = ['ED', 'S', 'Fe', 'Mn', 'AOP', 'P', 'Porosity', 'F', 'B', 'A', 'TNre(%)']
        self.raw_data = self.raw_data[cols_needed].dropna()
        numeric_cols = self.raw_data.select_dtypes(include=np.number).columns
        self.raw_data[numeric_cols] = (self.raw_data[numeric_cols] - self.raw_data[numeric_cols].mean()) / \
                                      self.raw_data[numeric_cols].std()
        return self.raw_data

    def define_variables(self):
        self.treatment_vars = ['ED', 'S', 'Fe', 'AOP', 'Mn', 'P', 'Porosity', 'F', 'B', 'A']
        self.outcome_vars = ['TNre(%)']

    def run_causal_discovery(self, data):
        data_np = data.values
        black_list = []
        for outcome in self.outcome_vars:
            for var in self.treatment_vars + self.outcome_vars:
                if var != outcome:
                    black_list.append((outcome, var))
        self.cg = pc(
            data=data_np,
            node_names=list(data.columns),
            alpha=0.1,
            indep_test='fisherz',
            stable=True,
            uc_rule=1,
            uc_priority=3,
            black_list=black_list,
            show_progress=False
        )
        logging.info("因果结构学习完成")

    def validate_effects(self):
        # 使用双重机器学习（DML）验证中介路径
        class EffectValidator:
            def __init__(self, data, treatment_vars, outcome_vars, n_bootstrap=200):
                self.data = data
                self.treatment_vars = treatment_vars
                self.outcome_vars = outcome_vars
                self.n_bootstrap = n_bootstrap
                self.model = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.6,
                    gamma=0.2,
                    reg_alpha=0,
                    reg_lambda=0,
                    random_state=42
                )

            def _bootstrap_effect(self, treatment, outcome, controls=None):
                try:
                    boot_data = resample(self.data, replace=True, n_samples=len(self.data))
                    X = boot_data.drop(columns=[outcome])
                    y = boot_data[outcome]

                    if controls is not None:
                        X_controls = boot_data[controls]
                        X = pd.concat([X, X_controls], axis=1)

                    self.model.fit(X, y)
                    y_pred = self.model.predict(X)
                    residual_Y = y - y_pred

                    T = boot_data[treatment]
                    X_T = X.copy()
                    if treatment not in X_T.columns:
                        X_T[treatment] = T
                    else:
                        X_T[treatment] = T

                    t_pred = self.model.predict(X_T)
                    residual_T = T - t_pred

                    X_resid = residual_T.values.reshape(-1, 1)
                    X_resid = sm.add_constant(X_resid)
                    model_resid = sm.OLS(residual_Y, X_resid).fit()

                    return model_resid.params.iloc[1]
                except Exception as e:
                    logging.error(f"Bootstrap 过程中出现错误: {str(e)}")
                    return np.nan

            def analyze(self):
                results = []
                for treat in self.treatment_vars:
                    for outcome in self.outcome_vars:
                        effects = Parallel(n_jobs=-1)(
                            delayed(self._bootstrap_effect)(treat, outcome) for _ in range(self.n_bootstrap)
                        )
                        effects = [e for e in effects if not np.isnan(e)]
                        if len(effects) == 0:
                            continue
                        effect_mean = np.mean(effects)
                        ci_lower = np.percentile(effects, 2.5)
                        ci_upper = np.percentile(effects, 97.5)
                        results.append({
                            'Treatment': treat,
                            'Outcome': outcome,
                            'Effect': effect_mean,
                            'CI_Lower': ci_lower,
                            'CI_Upper': ci_upper,
                            'N': len(effects)
                        })
                return pd.DataFrame(results)

        validator = EffectValidator(self.raw_data, self.treatment_vars, self.outcome_vars, self.n_bootstrap)
        results_df = validator.analyze()
        results_df.to_excel('causal_effects_xgboost.xlsx', index=False)
        return results_df

    def perform_mediation_paths(self):
        # 定义路径
        paths = [
            ('ED', 'P', 'TNre(%)'),
            ('ED', 'F', 'TNre(%)'),
            ('ED', 'B', 'TNre(%)'),
            ('ED', 'A', 'TNre(%)'),
            ('S', 'P', 'TNre(%)'),
            ('S', 'F', 'TNre(%)'),
            ('S', 'B', 'TNre(%)'),
            ('S', 'A', 'TNre(%)'),
            ('Fe', 'P', 'TNre(%)'),
            ('Fe', 'F', 'TNre(%)'),
            ('Fe', 'B', 'TNre(%)'),
            ('Fe', 'A', 'TNre(%)'),
            ('Mn', 'P', 'TNre(%)'),
            ('Mn', 'F', 'TNre(%)'),
            ('Mn', 'B', 'TNre(%)'),
            ('Mn', 'A', 'TNre(%)'),
            ('Porosity', 'B', 'TNre(%)'),
            ('Porosity', 'A', 'TNre(%)'),
            ('Porosity', 'P', 'TNre(%)'),
            ('Porosity', 'F', 'TNre(%)'),
            ('AOP', 'P', 'TNre(%)'),
            ('AOP', 'F', 'TNre(%)'),
            ('AOP', 'B', 'TNre(%)'),
            ('AOP', 'A', 'TNre(%)'),
        ]
        print("\n开始路径中介分析：")
        for X, M, Y in paths:
            print(f"\n路径分析：{X} → {M} → {Y}")
            self.mediation_analysis(self.raw_data, X, M, Y)

    def mediation_analysis(self, data, X, M, Y, n_bootstrap=200):
        effects = []
        for _ in range(n_bootstrap):
            sample = resample(data)
            try:
                model_a = sm.OLS(sample[M], sm.add_constant(sample[X])).fit()
                a_coef = model_a.params[1]
                model_b = sm.OLS(sample[Y], sm.add_constant(sample[[X, M]])).fit()
                b_coef = model_b.params[M]
                effects.append(a_coef * b_coef)
            except:
                effects.append(np.nan)
        effects = [e for e in effects if not np.isnan(e)]
        effect_mean = np.mean(effects)
        ci_lower = np.percentile(effects, 2.5)
        ci_upper = np.percentile(effects, 97.5)
        print(f"路径 {X}→{M}→{Y} 中介效应：{effect_mean:.3f}")
        print(f"95% 置信区间：[{ci_lower:.4f}, {ci_upper:.4f}]")
        return effect_mean, (ci_lower, ci_upper)

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv"
    analyzer = CausalAnalyzer(DATA_PATH)
    data = analyzer.load_and_preprocess_data()
    analyzer.define_variables()
    # 不调用因果结构学习（去掉）
    analyzer.run_causal_discovery(data)
    results = analyzer.validate_effects()

    # 新增路径中介分析调用
    analyzer.perform_mediation_paths()

    # 输出结果
    try:
        import tabulate
        print("\n因果效应分析结果：")
        print(results.round(3).to_markdown(index=False))
    except ImportError:
        print("\n因果效应分析结果：")
        print(results.round(3))
