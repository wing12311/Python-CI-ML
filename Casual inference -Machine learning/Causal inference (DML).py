
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
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
        self.treatment_vars = []
        self.outcome_vars = []
        self.n_bootstrap = 300
        self.n_folds = 5

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
        self.treatment_vars = ['ED', 'S', 'Fe', 'AOP', 'Mn',  'Porosity']
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
            alpha=0.01,
            indep_test='fisherz',
            stable=True,
            uc_rule=0,
            uc_priority=2,
            black_list=black_list,
            show_progress=False
        )
        logging.info("因果结构学习完成")

    # 增强版双重机器学习实现
    def validate_effects(self):
        class EnhancedDMLValidator:
            def __init__(self, data, treatment_vars, outcome_vars, n_bootstrap=300):
                self.data = data
                self.treatment_vars = treatment_vars
                self.outcome_vars = outcome_vars
                self.n_bootstrap = n_bootstrap
                self.y_model = XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.6,
                    colsample_bytree=0.7,
                    gamma=0.1,
                    reg_alpha=0.01,
                    reg_lambda=1,
                    random_state=42
                )

                self.t_model = XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.01,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42
                )

            def _dml_estimator(self, treatment, outcome):
                try:
                    # 采样
                    sample = resample(self.data, replace=True, n_samples=len(self.data), random_state=None)
                    # 取出变量
                    X = sample[[col for col in self.data.columns if col not in [treatment, outcome]]]
                    T = sample[treatment].values.reshape(-1, 1)
                    y = sample[outcome].values

                    # 估计残差
                    self.y_model.fit(X, y)
                    y_pred = self.y_model.predict(X)
                    y_resid = y - y_pred

                    self.t_model.fit(X, T)
                    t_pred = self.t_model.predict(X)
                    t_resid = T.flatten() - t_pred.flatten()

                    # 计算效果
                    numerator = np.sum(t_resid * y_resid)
                    denominator = np.sum(t_resid ** 2)
                    if denominator == 0:
                        return np.nan
                    effect = numerator / denominator
                    return effect
                except Exception as e:
                    logging.error(f"DML估计错误: {str(e)}")
                    return np.nan

            def bootstrap_analysis(self):
                results = []
                for treat in self.treatment_vars:
                    for outcome in self.outcome_vars:
                        effects = Parallel(n_jobs=-1)(
                            delayed(self._dml_estimator)(treat, outcome)
                            for _ in range(self.n_bootstrap)
                        )
                        valid_effects = [e for e in effects if not np.isnan(e)]
                        if len(valid_effects) == 0:
                            continue
                        effect_mean = np.mean(valid_effects)
                        ci_lower = np.percentile(valid_effects, 2.5)
                        ci_upper = np.percentile(valid_effects, 97.5)
                        results.append({
                            'Treatment': treat,
                            'Outcome': outcome,
                            'Method': 'DML',
                            'Effect': effect_mean,
                            'CI_Lower': ci_lower,
                            'CI_Upper': ci_upper
                        })
                return pd.DataFrame(results)

        validator = EnhancedDMLValidator(self.raw_data, self.treatment_vars, self.outcome_vars, self.n_bootstrap)
        results_df = validator.bootstrap_analysis()
        results_df.to_excel('enhanced_causal_effects.xlsx', index=False)
        return results_df

    # 只保留线性中介分析
    def perform_mediation_paths(self):
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
            ('Porosity', 'F', 'NH4re(%)'),
            ('AOP', 'P', 'TNre(%)'),
            ('AOP', 'F', 'TNre(%)'),
            ('AOP', 'B', 'TNre(%)'),
            ('AOP', 'A', 'TNre(%)'),
        ]
        results = []
        for path in paths:
            X, M, Y = path
            linear_effect, linear_ci = self.mediation_analysis(X, M, Y)

            results.append({
                'Path': f"{X}→{M}→{Y}",
                'Linear_Effect': linear_effect,
                'Linear_CI': f"[{linear_ci[0]:.3f}, {linear_ci[1]:.3f}]",

            })

        result_df = pd.DataFrame(results)
        result_df.to_excel('mediation_results.xlsx', index=False)
        print(result_df.round(3))
        return result_df

    def mediation_analysis(self, X, M, Y):
        effects = []
        for _ in range(self.n_bootstrap):
            sample = resample(self.raw_data)
            try:
                # 阶段1：X → M
                X_sm = sm.add_constant(sample[X])
                model_m = sm.OLS(sample[M], X_sm).fit()
                a = model_m.params[1]

                # 阶段2：X + M → Y
                XM_sm = sm.add_constant(sample[[X, M]])
                model_y = sm.OLS(sample[Y], XM_sm).fit()
                b = model_y.params[M]

                effects.append(a * b)
            except:
                effects.append(np.nan)

        valid_effects = np.array([e for e in effects if not np.isnan(e)])
        if len(valid_effects) == 0:
            return np.nan, (np.nan, np.nan)

        return np.mean(valid_effects), np.percentile(valid_effects, [2.5, 97.5])

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\s1769\Desktop\数据和模型\虚拟样本516改3.csv"
    analyzer = CausalAnalyzer(DATA_PATH)
    data = analyzer.load_and_preprocess_data()
    analyzer.define_variables()

    # 因果效应分析（DML）
    effect_results = analyzer.validate_effects()

    # 中介路径分析（只保留线性）
    mediation_results = analyzer.perform_mediation_paths()

    # 结果输出
    print("\n双重机器学习结果:")
    print(effect_results.round(3))

    print("\n线性中介分析结果:")
    print(mediation_results.round(3))
