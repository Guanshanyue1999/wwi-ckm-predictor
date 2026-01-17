#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
WWI与CKM综合征关联与预测 - 完整机器学习流水线
================================================================================
作者: 郑赫
学号: 2511110259
单位: 北京大学医学部第一临床医学院
课程: 健康数据科学的Python语言编程基础

本脚本包含:
1. 数据读取与预处理
2. 特征工程与变量构造
3. LASSO特征选择
4. 多模型训练与交叉验证 (Logistic, RF, XGBoost, LightGBM, MLP)
5. 模型评估 (ROC, 校准曲线, DCA, DeLong检验)
6. SHAP可解释性分析
7. 结果可视化与导出
================================================================================
"""

# =============================================================================
# 0. 环境配置与依赖安装
# =============================================================================
import subprocess
import sys
import json

def install_packages():
    """安装所需的Python包"""
    packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'scikit-learn', 'statsmodels', 'pyreadstat',
        'xgboost', 'lightgbm', 'shap', 'imbalanced-learn'
    ]
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# 取消注释以自动安装依赖
# install_packages()

# =============================================================================
# 1. 导入依赖库
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 数据读取
import pyreadstat

# 统计建模
from statsmodels.api import Logit, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 机器学习
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# XGBoost & LightGBM
import xgboost as xgb
import lightgbm as lgb

# SHAP解释
import shap

# 类不平衡处理
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# 2. 数据读取与预处理
# =============================================================================
def load_and_preprocess_data(filepath):
    """
    读取SAV数据并进行预处理
    
    Parameters:
    -----------
    filepath : str
        SAV文件路径
        
    Returns:
    --------
    df : pd.DataFrame
        预处理后的数据框
    """
    print("="*60)
    print("Step 1: 数据读取与预处理")
    print("="*60)
    
    # 读取SAV文件
    df, meta = pyreadstat.read_sav(filepath)
    print(f"原始数据维度: {df.shape}")
    
    # 必需字段
    needed = [
        "IN_2023", "Weight_2023", "WC_2023", 
        "CardiovascularDisease_2023", "CerebrovascularDisease_2023",
        "TG_2023", "LSBP_2023", "RSBP_2023", 
        "LDBP_2023", "RDBP_2023", "CR_2023",
        "AGE_2023", "Sex", "BMI_2023", "HDL_2023", "Height_2023",
        "Smoke_2023", "Drink_2023", "PA_2023", "Diet_2023",
        "TC_2023", "LDL_2023", "FBG_2023",
        "HTN_drugs_2023", "DM_drugs_2023", "DYS_drugs_2023",
        "DM_2023", "HTN_2023", "DYS_2023"
    ]
    
    # 缺失值处理
    df = df.dropna(subset=needed)
    print(f"缺失值处理后数据维度: {df.shape}")
    
    return df

def create_derived_variables(df):
    """
    创建派生变量
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据框
        
    Returns:
    --------
    df : pd.DataFrame
        包含派生变量的数据框
    """
    print("\nStep 2: 创建派生变量")
    print("-"*40)
    
    # 血压均值
    df["SBP_2023"] = (df["LSBP_2023"] + df["RSBP_2023"]) / 2
    df["DBP_2023"] = (df["LDBP_2023"] + df["RDBP_2023"]) / 2
    
    # 肌酐单位转换与eGFR计算 (MDRD公式)
    df["Scr_mg_dL"] = df["CR_2023"] * 0.0113
    df["eGFR_2023"] = (
        186 * (df["Scr_mg_dL"] ** -1.154) * 
        (df["AGE_2023"] ** -0.203) * 
        np.where(df["Sex"] == 2, 0.742, 1) * 1.227
    )
    
    # WWI - 体重调整腰围指数
    df["WWI_2023"] = df["WC_2023"] / np.sqrt(df["Weight_2023"])
    
    # WHtR - 腰围身高比
    df["WHtR_2023"] = df["WC_2023"] / df["Height_2023"]
    
    # VAI - 内脏脂肪指数
    df["VAI_2023"] = np.where(
        df["Sex"] == 1,
        (df["WC_2023"] / (39.68 + 1.88 * df["BMI_2023"])) * 
        (df["TG_2023"] / 1.03) * (1.31 / df["HDL_2023"]),
        (df["WC_2023"] / (39.58 + 1.89 * df["BMI_2023"])) * 
        (df["TG_2023"] / 0.81) * (1.52 / df["HDL_2023"])
    )
    
    # LAP - 脂肪积聚产物
    df["LAP_2023"] = np.where(
        df["Sex"] == 1,
        (df["WC_2023"] - 65) * df["TG_2023"],
        (df["WC_2023"] - 58) * df["TG_2023"]
    )
    
    # CVAI - 中国内脏脂肪指数
    df["CVAI_2023"] = np.where(
        df["Sex"] == 1,
        -267.93 + 0.68 * df["AGE_2023"] + 0.03 * df["BMI_2023"] + 
        4.00 * df["WC_2023"] + 22.00 * np.log10(df["TG_2023"]) - 
        16.32 * df["HDL_2023"],
        -187.32 + 1.71 * df["AGE_2023"] + 4.23 * df["BMI_2023"] + 
        1.12 * df["WC_2023"] + 39.76 * np.log10(df["TG_2023"]) - 
        11.66 * df["HDL_2023"]
    )
    
    # CMI - 心代谢指数
    df["CMI_2023"] = df["WHtR_2023"] * (df["TG_2023"] / df["HDL_2023"])
    
    # BRI - 身体圆形度指数
    df["BRI_2023"] = 364.2 - 365.5 * np.sqrt(
        1 - (df["WC_2023"] / (2 * np.pi)) ** 2 / (0.5 * df["Height_2023"]) ** 2
    )
    
    # ABSI - A Body Shape Index
    df["ABSI_2023"] = (df["WC_2023"] / 100) / (
        df["BMI_2023"] ** (2/3) * np.sqrt(df["Height_2023"] / 100)
    )
    
    # CI - 圆锥指数
    df["CI_2023"] = (df["WC_2023"] / 100) / (
        0.109 * np.sqrt(df["Weight_2023"] / (df["Height_2023"] / 100))
    )
    
    print("已创建的体测指标: WWI, WHtR, VAI, LAP, CVAI, CMI, BRI, ABSI, CI")
    
    return df

def create_outcome_variables(df):
    """
    创建结局变量
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
        
    Returns:
    --------
    df : pd.DataFrame
        包含结局变量的数据框
    """
    print("\nStep 3: 创建结局变量")
    print("-"*40)
    
    # MetS组件 (IDF 2005标准)
    df["Central_Obesity"] = np.where(
        ((df["WC_2023"] >= 90) & (df["Sex"] == 1)) |
        ((df["WC_2023"] >= 80) & (df["Sex"] == 2)), 1, 0
    )
    
    df["Raised_BP"] = np.where(
        (df["SBP_2023"] >= 130) | (df["DBP_2023"] >= 85) | 
        (df["HTN_drugs_2023"] == 1), 1, 0
    )
    
    df["Raised_FBG"] = np.where(
        (df["FBG_2023"] >= 5.6) | (df["DM_2023"] == 1) | 
        (df["DM_drugs_2023"] == 1), 1, 0
    )
    
    df["Raised_TG"] = np.where(
        (df["TG_2023"] > 1.7) | (df["DYS_drugs_2023"] == 1), 1, 0
    )
    
    df["Reduced_HDL"] = np.where(
        ((df["HDL_2023"] < 1.03) & (df["Sex"] == 1)) |
        ((df["HDL_2023"] < 1.29) & (df["Sex"] == 2)), 1, 0
    )
    
    # MetS定义
    mets_components = df[["Raised_BP", "Raised_FBG", "Raised_TG", "Reduced_HDL"]].sum(axis=1)
    df["MetS"] = np.where((df["Central_Obesity"] == 1) & (mets_components >= 2), 1, 0)
    
    # CVD/CbVD/CKD
    df["CVD_2023"] = (df["CardiovascularDisease_2023"] != 0).astype(int)
    df["CbVD_2023"] = (df["CerebrovascularDisease_2023"] != 0).astype(int)
    df["CKD"] = (df["eGFR_2023"] < 60).astype(int)
    df["CVD"] = ((df["CVD_2023"] == 1) | (df["CbVD_2023"] == 1)).astype(int)
    
    # CKM综合征 (简化定义: MetS + CVD/CKD)
    df["CKM"] = ((df["MetS"] == 1) & ((df["CKD"] == 1) | (df["CVD"] == 1))).astype(int)
    
    # 打印结局分布
    outcomes = ["MetS", "CVD", "CbVD_2023", "CKD", "CKM"]
    print("\n结局变量分布:")
    for outcome in outcomes:
        n_pos = df[outcome].sum()
        pct = n_pos / len(df) * 100
        print(f"  {outcome}: n={n_pos} ({pct:.1f}%)")
    
    return df

# =============================================================================
# 3. 特征选择 - LASSO回归
# =============================================================================
def lasso_feature_selection(X, y, cv=10, random_state=42):
    """
    使用LASSO进行特征选择
    
    Parameters:
    -----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    cv : int
        交叉验证折数
    random_state : int
        随机种子
        
    Returns:
    --------
    selected_features : list
        选中的特征列表
    lasso : LassoCV
        拟合后的LASSO模型
    """
    print("\n" + "="*60)
    print("Step 4: LASSO特征选择")
    print("="*60)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LASSO CV
    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # 选择非零系数特征
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    selected_features = coef_df[coef_df['Coefficient'] != 0]['Feature'].tolist()
    
    print(f"最优lambda: {lasso.alpha_:.6f}")
    print(f"选中特征数: {len(selected_features)}/{len(X.columns)}")
    print("\n特征系数 (非零):")
    print(coef_df[coef_df['Coefficient'] != 0].to_string(index=False))
    
    # 绘制LASSO路径图
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_df_plot = coef_df[coef_df['Coefficient'] != 0].copy()
    colors = plt.cm.viridis(np.linspace(0, 1, len(coef_df_plot)))
    bars = ax.barh(coef_df_plot['Feature'], coef_df_plot['Coefficient'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coefficient')
    ax.set_title(f'LASSO Feature Selection (lambda={lasso.alpha_:.4f})')
    plt.tight_layout()
    plt.savefig('lasso_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return selected_features, lasso

# =============================================================================
# 4. 模型定义与训练
# =============================================================================
def get_models(imbalance_ratio=1.0):
    """
    获取待比较的模型字典
    
    Parameters:
    -----------
    imbalance_ratio : float
        正负样本比例
        
    Returns:
    --------
    models : dict
        模型字典
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, cv=10):
    """
    训练和评估多个模型
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练集和测试集特征
    y_train, y_test : array-like
        训练集和测试集标签
    models : dict
        模型字典
    cv : int
        交叉验证折数
        
    Returns:
    --------
    results : dict
        包含各模型评估结果的字典
    """
    print("\n" + "="*60)
    print("Step 5: 模型训练与交叉验证")
    print("="*60)
    
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        print("-"*40)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        
        # 在全部训练集上拟合
        model.fit(X_train, y_train)
        
        # 测试集预测
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # 计算各项指标
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        # ROC曲线数据
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        # 校准曲线数据
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_auc': test_auc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'brier_score': brier,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
        
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  Brier Score: {brier:.4f}")
    
    return results

# =============================================================================
# 5. DeLong检验 - AUC比较
# =============================================================================
def delong_roc_variance(ground_truth, predictions):
    """
    计算AUC的DeLong方差
    """
    order = np.argsort(-predictions)
    predictions_sorted = predictions[order]
    ground_truth_sorted = ground_truth[order]
    
    n_pos = np.sum(ground_truth)
    n_neg = len(ground_truth) - n_pos
    
    pos_indices = np.where(ground_truth_sorted == 1)[0]
    neg_indices = np.where(ground_truth_sorted == 0)[0]
    
    # 计算placement values
    V10 = np.zeros(int(n_pos))
    V01 = np.zeros(int(n_neg))
    
    for i, pos_idx in enumerate(pos_indices):
        V10[i] = np.sum(predictions_sorted[neg_indices] < predictions_sorted[pos_idx]) + \
                 0.5 * np.sum(predictions_sorted[neg_indices] == predictions_sorted[pos_idx])
    V10 = V10 / n_neg
    
    for i, neg_idx in enumerate(neg_indices):
        V01[i] = np.sum(predictions_sorted[pos_indices] > predictions_sorted[neg_idx]) + \
                 0.5 * np.sum(predictions_sorted[pos_indices] == predictions_sorted[neg_idx])
    V01 = V01 / n_pos
    
    auc_value = np.mean(V10)
    s10 = np.var(V10) / n_pos if n_pos > 1 else 0
    s01 = np.var(V01) / n_neg if n_neg > 1 else 0
    variance = s10 + s01
    
    return auc_value, variance

def delong_test(y_true, pred1, pred2):
    """
    DeLong检验比较两个AUC
    
    Returns:
    --------
    z_stat : float
        Z统计量
    p_value : float
        双侧p值
    """
    auc1, var1 = delong_roc_variance(y_true.values, pred1)
    auc2, var2 = delong_roc_variance(y_true.values, pred2)
    
    # 计算协方差
    order1 = np.argsort(-pred1)
    order2 = np.argsort(-pred2)
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    ground_truth = y_true.values
    pos_indices = np.where(ground_truth == 1)[0]
    neg_indices = np.where(ground_truth == 0)[0]
    
    # 简化的协方差估计
    cov = 0.5 * (var1 + var2) * 0.5  # 近似协方差
    
    se = np.sqrt(var1 + var2 - 2*cov)
    z_stat = (auc1 - auc2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, auc1, auc2

def compare_all_models_delong(y_test, results, reference_model='XGBoost'):
    """
    使用DeLong检验比较所有模型与参考模型
    """
    print("\n" + "="*60)
    print(f"DeLong Test (Reference: {reference_model})")
    print("="*60)
    
    ref_pred = results[reference_model]['y_pred_proba']
    
    comparison_results = []
    for name, res in results.items():
        if name != reference_model:
            z, p, auc1, auc2 = delong_test(y_test, ref_pred, res['y_pred_proba'])
            comparison_results.append({
                'Comparison': f"{reference_model} vs {name}",
                f'AUC ({reference_model})': auc1,
                f'AUC ({name})': auc2,
                'Z-statistic': z,
                'P-value': p,
                'Significant': 'Yes' if p < 0.05 else 'No'
            })
            print(f"{reference_model} vs {name}: Z={z:.3f}, P={p:.4f}")
    
    return pd.DataFrame(comparison_results)

# =============================================================================
# 6. 决策曲线分析 (DCA)
# =============================================================================
def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    """
    计算决策曲线的净收益
    """
    n = len(y_true)
    net_benefits = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        # 净收益 = TP/n - FP/n * (pt/(1-pt))
        if thresh < 1:
            net_benefit = tp/n - fp/n * (thresh / (1 - thresh))
        else:
            net_benefit = 0
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)

def plot_decision_curve(y_test, results, outcome_name='CKM'):
    """
    绘制决策曲线
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 治疗所有人
    prevalence = y_test.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax.plot(thresholds, treat_all, 'k--', label='Treat All', linewidth=1.5)
    
    # 治疗无人
    ax.axhline(y=0, color='gray', linestyle=':', label='Treat None', linewidth=1.5)
    
    # 各模型
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, res), color in zip(results.items(), colors):
        nb = calculate_net_benefit(y_test.values, res['y_pred_proba'], thresholds)
        ax.plot(thresholds, nb, label=f"{name} (AUC={res['test_auc']:.3f})", 
                color=color, linewidth=2)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, max(prevalence, 0.3)])
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(f'Decision Curve Analysis - {outcome_name} Prediction', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'dca_{outcome_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# =============================================================================
# 7. 综合评估可视化
# =============================================================================
def plot_comprehensive_evaluation(y_test, results, outcome_name='CKM'):
    """
    绘制4合1综合评估图 (ROC, 校准曲线, DCA, 指标表)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # ========== (A) ROC曲线 ==========
    ax1 = axes[0, 0]
    for (name, res), color in zip(results.items(), colors):
        ax1.plot(res['fpr'], res['tpr'], 
                 label=f"{name} (AUC={res['test_auc']:.3f})",
                 color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax1.set_title('(A) ROC Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== (B) 校准曲线 ==========
    ax2 = axes[0, 1]
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
    for (name, res), color in zip(results.items(), colors):
        ax2.plot(res['prob_pred'], res['prob_true'], 
                 marker='o', label=f"{name} (Brier={res['brier_score']:.3f})",
                 color=color, linewidth=2, markersize=6)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax2.set_ylabel('Fraction of Positives', fontsize=11)
    ax2.set_title('(B) Calibration Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ========== (C) DCA曲线 ==========
    ax3 = axes[1, 0]
    thresholds = np.arange(0.01, 0.99, 0.01)
    prevalence = y_test.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax3.plot(thresholds, treat_all, 'k--', label='Treat All', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle=':', label='Treat None', linewidth=1.5)
    
    for (name, res), color in zip(results.items(), colors):
        nb = calculate_net_benefit(y_test.values, res['y_pred_proba'], thresholds)
        ax3.plot(thresholds, nb, label=f"{name}", color=color, linewidth=2)
    
    ax3.set_xlim([0, 0.8])
    ax3.set_ylim([-0.05, max(prevalence, 0.25)])
    ax3.set_xlabel('Threshold Probability', fontsize=11)
    ax3.set_ylabel('Net Benefit', fontsize=11)
    ax3.set_title('(C) Decision Curve Analysis', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ========== (D) 性能指标表 ==========
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_data = []
    for name, res in results.items():
        metrics_data.append([
            name,
            f"{res['cv_mean']:.3f}+/-{res['cv_std']:.3f}",
            f"{res['test_auc']:.3f}",
            f"{res['test_acc']:.3f}",
            f"{res['test_precision']:.3f}",
            f"{res['test_recall']:.3f}",
            f"{res['test_f1']:.3f}",
            f"{res['brier_score']:.3f}"
        ])
    
    columns = ['Model', 'CV AUC', 'Test AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Brier']
    
    table = ax4.table(
        cellText=metrics_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.18, 0.14, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # 表头样式
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # 交替行颜色
    for i in range(1, len(metrics_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    ax4.set_title('(D) Model Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Comprehensive Model Evaluation - {outcome_name} Prediction', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'comprehensive_evaluation_{outcome_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# =============================================================================
# 8. SHAP可解释性分析
# =============================================================================
def shap_analysis(model, X_train, X_test, feature_names, model_name='XGBoost', outcome_name='CKM'):
    """
    SHAP可解释性分析
    """
    print("\n" + "="*60)
    print(f"SHAP Analysis for {model_name}")
    print("="*60)
    
    # 创建Explainer
    if model_name in ['XGBoost', 'LightGBM', 'Random Forest']:
        explainer = shap.TreeExplainer(model)
    else:
        # 使用KernelExplainer作为后备
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)
    
    # 处理多类别输出
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 取正类的SHAP值
    
    # ========== 1. Summary Plot (Beeswarm) ==========
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {model_name} ({outcome_name})', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'shap_summary_{outcome_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 2. Feature Importance Bar Plot ==========
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name} ({outcome_name})', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'shap_importance_{outcome_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 3. Dependence Plot for WWI ==========
    if 'WWI_2023' in feature_names:
        wwi_idx = list(feature_names).index('WWI_2023')
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(wwi_idx, shap_values, X_test, 
                            feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot - WWI ({outcome_name})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'shap_dependence_wwi_{outcome_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========== 4. 计算平均绝对SHAP值 ==========
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)
    
    print("\nSHAP Feature Importance:")
    print(importance_df.to_string(index=False))
    
    return shap_values, importance_df

# =============================================================================
# 9. Bootstrap验证
# =============================================================================
def bootstrap_validation(model, X, y, n_bootstrap=200, random_state=42):
    """
    Bootstrap内部验证
    """
    print("\n" + "="*60)
    print(f"Bootstrap Validation (B={n_bootstrap})")
    print("="*60)
    
    np.random.seed(random_state)
    n = len(y)
    
    apparent_aucs = []
    optimism_aucs = []
    
    for b in range(n_bootstrap):
        # Bootstrap样本
        indices = np.random.choice(n, size=n, replace=True)
        oob_indices = list(set(range(n)) - set(indices))
        
        if len(oob_indices) < 10:  # 确保OOB样本足够
            continue
            
        X_boot, y_boot = X.iloc[indices], y.iloc[indices]
        X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]
        
        # 训练
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_boot, y_boot)
        
        # 计算AUC
        auc_boot = roc_auc_score(y_boot, model_clone.predict_proba(X_boot)[:, 1])
        auc_oob = roc_auc_score(y_oob, model_clone.predict_proba(X_oob)[:, 1])
        
        apparent_aucs.append(auc_boot)
        optimism_aucs.append(auc_boot - auc_oob)
    
    # 计算校正后的AUC
    mean_apparent = np.mean(apparent_aucs)
    mean_optimism = np.mean(optimism_aucs)
    corrected_auc = mean_apparent - mean_optimism
    
    print(f"Apparent AUC: {mean_apparent:.4f}")
    print(f"Optimism: {mean_optimism:.4f}")
    print(f"Corrected AUC: {corrected_auc:.4f}")
    print(f"95% CI: [{np.percentile(apparent_aucs, 2.5):.4f}, {np.percentile(apparent_aucs, 97.5):.4f}]")
    
    return {
        'apparent_auc': mean_apparent,
        'optimism': mean_optimism,
        'corrected_auc': corrected_auc,
        'ci_lower': np.percentile(apparent_aucs, 2.5),
        'ci_upper': np.percentile(apparent_aucs, 97.5)
    }

# =============================================================================
# 10. 生成结果汇总表
# =============================================================================
def generate_results_table(results, outcome_name='CKM'):
    """
    生成结果汇总表并保存为CSV
    """
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'CV_AUC_Mean': res['cv_mean'],
            'CV_AUC_Std': res['cv_std'],
            'Test_AUC': res['test_auc'],
            'Accuracy': res['test_acc'],
            'Precision': res['test_precision'],
            'Recall': res['test_recall'],
            'F1_Score': res['test_f1'],
            'Brier_Score': res['brier_score']
        })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(f'model_comparison_{outcome_name.lower()}.csv', index=False)
    
    print("\n" + "="*60)
    print("Model Comparison Results")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return df_results

# =============================================================================
# 10.5 保存模型元数据
# =============================================================================
def save_model_metadata(selected_features, X_selected, output_path="model_meta.json"):
    """
    保存特征列表与均值，便于应用端推理对齐
    """
    meta = {
        "selected_features": selected_features,
        "feature_means": X_selected.mean(numeric_only=True).to_dict()
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"模型元数据已保存: {output_path}")

# =============================================================================
# 11. 主函数
# =============================================================================
def main(sav_filepath='Thesis.sav'):
    """
    主函数 - 完整流水线
    """
    print("="*70)
    print("WWI与CKM综合征关联与预测 - 机器学习流水线")
    print("="*70)
    
    # 1. 数据加载与预处理
    df = load_and_preprocess_data(sav_filepath)
    df = create_derived_variables(df)
    df = create_outcome_variables(df)
    
    # 2. 定义特征和目标变量
    feature_cols = [
        'WWI_2023', 'BMI_2023', 'WC_2023', 'WHtR_2023',
        'AGE_2023', 'Sex',
        'SBP_2023', 'DBP_2023',
        'TG_2023', 'HDL_2023', 'LDL_2023', 'TC_2023', 'FBG_2023',
        'eGFR_2023',
        'Smoke_2023', 'Drink_2023', 'PA_2023',
        'HTN_drugs_2023', 'DM_drugs_2023', 'DYS_drugs_2023'
    ]
    
    X = df[feature_cols].copy()
    y = df['CKM'].copy()
    
    # 3. 数据清洗
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n最终分析样本量: n={len(y)}")
    print(f"CKM阳性: n={y.sum()} ({y.mean()*100:.1f}%)")
    
    # 4. LASSO特征选择
    selected_features, lasso_model = lasso_feature_selection(X, y)
    
    # 如果LASSO选择的特征太少，使用全部特征
    if len(selected_features) < 5:
        print("\nLASSO选择特征较少，使用全部特征...")
        selected_features = feature_cols
    
    X_selected = X[selected_features]

    # 保存模型元数据，供应用端推理使用
    save_model_metadata(selected_features, X_selected)
    
    # 5. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转回DataFrame以保留特征名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
    
    # 7. 计算类不平衡比例
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n类不平衡比例: {imbalance_ratio:.2f}:1")
    
    # 8. 获取模型并训练
    models = get_models(imbalance_ratio)
    results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, models
    )
    
    # 9. DeLong检验
    delong_df = compare_all_models_delong(y_test, results, reference_model='XGBoost')
    delong_df.to_csv('delong_comparison.csv', index=False)
    
    # 10. 综合评估可视化
    plot_comprehensive_evaluation(y_test, results, outcome_name='CKM')
    
    # 11. SHAP分析 (使用XGBoost)
    best_model_name = max(results, key=lambda x: results[x]['test_auc'])
    print(f"\n最佳模型: {best_model_name} (AUC={results[best_model_name]['test_auc']:.4f})")
    
    shap_values, shap_importance = shap_analysis(
        results['XGBoost']['model'],
        X_train_scaled, X_test_scaled,
        selected_features,
        model_name='XGBoost',
        outcome_name='CKM'
    )
    shap_importance.to_csv('shap_importance.csv', index=False)
    
    # 12. Bootstrap验证
    bootstrap_results = bootstrap_validation(
        results['XGBoost']['model'],
        X_selected, y,
        n_bootstrap=200
    )
    
    # 13. 生成结果汇总表
    results_df = generate_results_table(results, outcome_name='CKM')
    
    # 14. 保存最佳模型
    import joblib
    joblib.dump(results['XGBoost']['model'], 'best_model_xgboost.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\n" + "="*70)
    print("流水线完成!")
    print("="*70)
    print("\n生成的文件:")
    print("  - lasso_coefficients.png")
    print("  - comprehensive_evaluation_ckm.png")
    print("  - dca_ckm.png")
    print("  - shap_summary_ckm.png")
    print("  - shap_importance_ckm.png")
    print("  - shap_dependence_wwi_ckm.png")
    print("  - model_comparison_ckm.csv")
    print("  - delong_comparison.csv")
    print("  - shap_importance.csv")
    print("  - best_model_xgboost.pkl")
    print("  - scaler.pkl")
    print("  - model_meta.json")
    
    return df, results, shap_values

# =============================================================================
# 运行主程序
# =============================================================================
if __name__ == "__main__":
    # 设置数据文件路径
    # 如果在Codex中运行，可能需要调整路径
    import os
    
    # 尝试多个可能的路径
    possible_paths = [
        'Thesis.sav',
        './Thesis.sav',
        '/mnt/user-data/uploads/1768556386965_Thesis.sav',
        '../Thesis.sav'
    ]
    
    sav_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sav_path = path
            break
    
    if sav_path:
        df, results, shap_values = main(sav_path)
    else:
        print("错误: 未找到数据文件 Thesis.sav")
        print("请将数据文件放置在当前目录或指定正确路径")
        print("\n可用路径示例:")
        for p in possible_paths:
            print(f"  - {p}")
