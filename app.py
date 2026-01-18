#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
多种中心性肥胖指标与CKM综合征预测 - Streamlit网页应用
课程项目展示网页 - 北京大学医学部健康数据科学Python编程
学生：郑赫 (2511110259)
个人主页：https://guanshanyue1999.github.io/
================================================================================

部署方式：
1. 本地运行: streamlit run app.py
2. Streamlit Cloud: 连接GitHub仓库后一键部署 (share.streamlit.io)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="多种中心性肥胖指标与CKM综合征预测系统",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 自定义CSS样式
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=Source+Serif+4:wght@600;700&display=swap');
:root {
    --ink: #0f1b2d;
    --muted: #5b6b7a;
    --accent: #2f6fed;
    --accent-2: #1aa7a1;
    --card: #ffffff;
    --paper: #f7f6f1;
    --shadow: 0 10px 30px rgba(15, 27, 45, 0.08);
}
.stApp {
    background:
        radial-gradient(900px 400px at 85% -10%, rgba(187, 212, 255, 0.55) 0%, rgba(187, 212, 255, 0) 60%),
        radial-gradient(800px 500px at 10% 10%, rgba(217, 245, 238, 0.6) 0%, rgba(217, 245, 238, 0) 55%),
        linear-gradient(180deg, #f7f8fb 0%, #f2f0ea 100%);
}
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Noto Sans SC', sans-serif;
    color: var(--ink);
}
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f6f3ee 100%);
    border: 1px solid rgba(15, 27, 45, 0.08);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow);
    animation: fadeUp 0.6s ease-out both;
}
.hero-title {
    font-family: 'Source Serif 4', 'Noto Serif SC', serif;
    font-size: 3.1rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: 0.3px;
    margin: 0;
}
.hero-sub {
    font-size: 1.15rem;
    color: var(--muted);
    margin-top: 0.45rem;
}
.hero-tags {
    margin-top: 0.8rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.hero-tag {
    background: #eef3ff;
    color: #2f4fb0;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    border: 1px solid rgba(47, 79, 176, 0.2);
}
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.8rem;
    margin-top: 0.9rem;
}
.info-card {
    background: var(--card);
    border: 1px solid rgba(15, 27, 45, 0.08);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: var(--shadow);
    animation: fadeUp 0.6s ease-out both;
}
.info-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 1rem;
    color: var(--ink);
}
.info-card p {
    margin: 0;
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.4;
}
.metric-card {
    background: linear-gradient(135deg, #2f6fed 0%, #1aa7a1 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 20px rgba(47, 111, 237, 0.2);
}
.risk-high {
    background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
}
.risk-low {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.info-box {
    background-color: #f0f7ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1E3A5F;
    margin: 1rem 0;
}
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #f7f8fb 0%, #eef1f6 100%);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 辅助函数
# ============================================================================
MODEL_PATH = "best_model_xgboost.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "model_meta.json"

# 训练管线中的特征名（用于模型推理对齐）
MODEL_FEATURES = [
    "WWI_2023",
    "BMI_2023",
    "WC_2023",
    "WHtR_2023",
    "AGE_2023",
    "Sex",
    "SBP_2023",
    "DBP_2023",
    "TG_2023",
    "HDL_2023",
    "LDL_2023",
    "TC_2023",
    "FBG_2023",
    "eGFR_2023",
    "Smoke_2023",
    "Drink_2023",
    "PA_2023",
    "HTN_drugs_2023",
    "DM_drugs_2023",
    "DYS_drugs_2023",
]

@st.cache_resource(show_spinner=False)
def load_model_assets():
    """加载模型、标准化器与元数据（若存在）"""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        return None, None, None, str(exc)

    meta = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except json.JSONDecodeError:
            meta = None

    return model, scaler, meta, None

def resolve_feature_order(meta, scaler, model):
    if meta and "selected_features" in meta:
        return meta["selected_features"]
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return MODEL_FEATURES

def predict_with_model(model, scaler, feature_values, meta):
    feature_order = resolve_feature_order(meta, scaler, model)
    defaults = (meta or {}).get("feature_means", {})
    aligned = {}
    missing = []

    for name in feature_order:
        if name in feature_values:
            aligned[name] = feature_values[name]
        elif name in defaults:
            aligned[name] = defaults[name]
        else:
            aligned[name] = 0
            missing.append(name)

    feature_frame = pd.DataFrame([aligned], columns=feature_order)
    scaled = scaler.transform(feature_frame)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[:, 1][0]
    else:
        prob = float(model.predict(scaled)[0])

    return float(np.clip(prob, 0.0, 1.0)), missing

def calculate_wwi(waist_cm, weight_kg):
    """计算WWI (Weight-Adjusted Waist Index)"""
    if weight_kg > 0:
        return waist_cm / np.sqrt(weight_kg)
    return 0

def calculate_whtr(waist_cm, height_cm):
    """计算WHtR (Waist-to-Height Ratio)"""
    if height_cm > 0:
        return waist_cm / height_cm
    return 0

def calculate_bmi(weight_kg, height_cm):
    """计算BMI"""
    if height_cm > 0:
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    return 0

def calculate_egfr(creatinine_umol, age, is_female):
    """计算eGFR (MDRD公式)"""
    scr_mg_dl = creatinine_umol * 0.0113
    if scr_mg_dl > 0:
        egfr = 186 * (scr_mg_dl ** -1.154) * (age ** -0.203)
        if is_female:
            egfr *= 0.742
        egfr *= 1.227  # 中国人群校正
        return egfr
    return 90  # 默认值

def predict_ckm_risk(features):
    """
    CKM风险预测（简化版本，模型文件不可用时作为备选）
    
    基于Logistic回归简化模型的系数（示例）
    """
    # 简化的风险评分模型（基于文献和研究结果）
    wwi = features['wwi']
    age = features['age']
    sbp = features['sbp']
    fbg = features['fbg']
    egfr = features['egfr']
    sex = features['sex']  # 1=男, 2=女
    
    # 归一化处理
    wwi_norm = (wwi - 11.0) / 0.8  # 基于人群均值和标准差
    age_norm = (age - 72) / 6
    sbp_norm = (sbp - 136) / 17
    fbg_norm = (fbg - 5.5) / 1.5
    egfr_norm = (egfr - 90) / 30
    
    # 简化的线性组合（基于文献OR值）
    log_odds = (
        -3.5 +  # 截距
        0.58 * wwi_norm +  # WWI效应
        0.35 * age_norm +  # 年龄效应
        0.25 * sbp_norm +  # 血压效应
        0.40 * fbg_norm +  # 血糖效应
        -0.30 * egfr_norm +  # eGFR效应(保护因素)
        (0.15 if sex == 1 else 0)  # 男性风险略高
    )
    
    # 转换为概率
    risk_prob = 1 / (1 + np.exp(-log_odds))
    
    return risk_prob

def get_risk_category(prob):
    """根据预测概率划分风险等级"""
    if prob < 0.1:
        return "低风险", "#28a745"
    elif prob < 0.3:
        return "中低风险", "#17a2b8"
    elif prob < 0.5:
        return "中等风险", "#ffc107"
    elif prob < 0.7:
        return "中高风险", "#fd7e14"
    else:
        return "高风险", "#dc3545"

# ============================================================================
# 主界面
# ============================================================================
def main():
    # 标题与说明
    st.markdown("""
    <div class="hero">
        <div class="hero-title">多种中心性肥胖指标与CKM综合征预测系统</div>
        <div class="hero-sub">基于机器学习的心肾代谢综合征风险评估工具</div>
        <div class="hero-tags">
            <span class="hero-tag">多指标比较</span>
            <span class="hero-tag">XGBoost 预测模型</span>
            <span class="hero-tag">可解释性 SHAP</span>
            <span class="hero-tag">老年人群数据</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <h4>使用说明</h4>
            <p>在左侧输入体型、代谢与肾功能信息，系统自动计算 WC/WHtR/WWI/BMI 并给出 CKM 风险评估。</p>
        </div>
        <div class="info-card">
            <h4>模型说明</h4>
            <p>模型基于 XGBoost，特征包含体型指标 + 血压血脂血糖 + eGFR + 生活方式与用药；测试集 AUC ≈ 0.93。</p>
        </div>
        <div class="info-card">
            <h4>注意事项</h4>
            <p>结果用于科研与健康管理参考，不替代临床诊断；建议结合医生意见综合评估。</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>模型输入指标：</b>
        体型（WC、WHtR、WWI、BMI）；
        代谢/肾功能（SBP、DBP、TG、HDL、LDL、TC、FBG、eGFR）；
        生活方式与用药（吸烟、饮酒、体力活动、降压/降糖/降脂）；
        基本信息（年龄、性别）。
    </div>
    """, unsafe_allow_html=True)

    model, scaler, meta, load_error = load_model_assets()
    model_ready = model is not None and scaler is not None and load_error is None
    
    # 侧边栏 - 用户信息输入
    with st.sidebar:
        st.header("📋 请输入您的健康信息")
        
        st.subheader("👤 基本信息")
        age = st.slider("年龄 (岁)", 40, 100, 70)
        sex = st.radio("性别", ["男", "女"], horizontal=True)
        sex_code = 1 if sex == "男" else 2
        wc_threshold = 90 if sex_code == 1 else 80
        
        st.subheader("📏 体格测量")
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("身高 (cm)", 140.0, 200.0, 165.0, 0.5)
            weight = st.number_input("体重 (kg)", 30.0, 150.0, 65.0, 0.5)
        with col2:
            waist = st.number_input("腰围 (cm)", 50.0, 150.0, 85.0, 0.5)
        
        st.subheader("💉 血压血糖")
        col3, col4 = st.columns(2)
        with col3:
            sbp = st.number_input("收缩压 (mmHg)", 80, 220, 135)
            dbp = st.number_input("舒张压 (mmHg)", 40, 140, 78)
        with col4:
            fbg = st.number_input("空腹血糖 (mmol/L)", 2.0, 20.0, 5.5, 0.1)
        
        st.subheader("🧪 肾功能")
        creatinine = st.number_input("血肌酐 (μmol/L)", 30.0, 500.0, 80.0, 1.0)
        
        st.subheader("🧬 血脂")
        col5, col6 = st.columns(2)
        with col5:
            tg = st.number_input("甘油三酯 (mmol/L)", 0.3, 10.0, 1.5, 0.1)
        with col6:
            hdl = st.number_input("HDL-C (mmol/L)", 0.3, 3.0, 1.3, 0.1)

        col7, col8 = st.columns(2)
        with col7:
            ldl = st.number_input("LDL-C (mmol/L)", 0.5, 6.0, 2.6, 0.1)
        with col8:
            tc = st.number_input("总胆固醇 (mmol/L)", 2.0, 10.0, 4.8, 0.1)

        with st.expander("生活方式与用药（可选）", expanded=False):
            col9, col10 = st.columns(2)
            with col9:
                smoke = st.checkbox("吸烟", value=False)
                drink = st.checkbox("饮酒", value=False)
                pa = st.checkbox("规律运动", value=False)
            with col10:
                htn_drugs = st.checkbox("降压药", value=False)
                dm_drugs = st.checkbox("降糖药", value=False)
                dys_drugs = st.checkbox("降脂药", value=False)

        st.subheader("模型预测")
        use_model = st.checkbox("使用预训练模型（如可用）", value=model_ready, disabled=not model_ready)
        if load_error:
            st.warning("模型加载失败，已使用简化评分。")
        elif not model_ready:
            st.caption("未检测到模型文件，将使用简化评分。")
    
    # 计算派生指标
    wwi = calculate_wwi(waist, weight)
    whtr = calculate_whtr(waist, height)
    bmi = calculate_bmi(weight, height)
    egfr = calculate_egfr(creatinine, age, sex_code == 2)

    model_features = {
        "WWI_2023": wwi,
        "BMI_2023": bmi,
        "WC_2023": waist,
        "WHtR_2023": whtr,
        "AGE_2023": age,
        "Sex": sex_code,
        "SBP_2023": sbp,
        "DBP_2023": dbp,
        "TG_2023": tg,
        "HDL_2023": hdl,
        "LDL_2023": ldl,
        "TC_2023": tc,
        "FBG_2023": fbg,
        "eGFR_2023": egfr,
        "Smoke_2023": int(smoke),
        "Drink_2023": int(drink),
        "PA_2023": int(pa),
        "HTN_drugs_2023": int(htn_drugs),
        "DM_drugs_2023": int(dm_drugs),
        "DYS_drugs_2023": int(dys_drugs),
    }
    
    # 主要内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 风险评估", "📊 指标解读", "📈 数据可视化", "ℹ️ 关于项目"])
    
    # ========== Tab 1: 风险评估 ==========
    with tab1:
        st.header("CKM综合征风险评估结果")
        
        # 计算风险
        features = {
            'wwi': wwi,
            'age': age,
            'sbp': sbp,
            'fbg': fbg,
            'egfr': egfr,
            'sex': sex_code
        }
        use_model_now = use_model and model_ready
        if use_model_now:
            try:
                risk_prob, missing_features = predict_with_model(
                    model,
                    scaler,
                    model_features,
                    meta
                )
                if missing_features:
                    st.warning(f"模型特征缺失，已使用默认值填充: {', '.join(missing_features)}")
            except Exception:
                st.warning("模型预测失败，已使用简化评分。")
                use_model_now = False
                risk_prob = predict_ckm_risk(features)
        else:
            risk_prob = predict_ckm_risk(features)

        mode_label = "预训练模型" if use_model_now else "简化评分"
        st.caption(f"当前评分方式：{mode_label}")
        risk_category, risk_color = get_risk_category(risk_prob)
        
        # 展示结果
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # 风险仪表盘
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CKM风险评分", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 48}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 10], 'color': '#c8e6c9'},
                        {'range': [10, 30], 'color': '#fff9c4'},
                        {'range': [30, 50], 'color': '#ffe0b2'},
                        {'range': [50, 70], 'color': '#ffccbc'},
                        {'range': [70, 100], 'color': '#ffcdd2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_prob * 100
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # 风险等级
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {risk_color}20; 
                    border-radius: 10px; border: 2px solid {risk_color};">
            <h2 style="color: {risk_color}; margin: 0;">风险等级：{risk_category}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 个性化建议
        st.subheader("💡 健康建议")
        
        suggestions = []
        if waist >= wc_threshold:
            suggestions.append("⚠️ 腰围偏高，提示中心性肥胖风险增加，建议控制总能量摄入并加强运动。")
        if whtr >= 0.5:
            suggestions.append("⚠️ WHtR偏高（≥0.5），提示腹型肥胖风险增加。")
        if wwi > 11.5:
            suggestions.append("⚠️ WWI偏高，可作为中心性肥胖的补充提示指标。")
        if bmi >= 28:
            suggestions.append("⚠️ BMI提示肥胖，建议在医生指导下进行体重管理。")
        if sbp >= 140 or dbp >= 90:
            suggestions.append("⚠️ 血压偏高，建议低盐饮食，定期监测血压，必要时就医。")
        if fbg >= 7.0:
            suggestions.append("⚠️ 空腹血糖偏高，建议进行糖耐量检查，控制碳水化合物摄入。")
        if tg >= 1.7:
            suggestions.append("⚠️ 甘油三酯偏高，建议控制油脂摄入并增加运动。")
        if (sex_code == 1 and hdl < 1.03) or (sex_code == 2 and hdl < 1.29):
            suggestions.append("⚠️ HDL-C偏低，建议改善饮食结构并规律运动。")
        if egfr < 60:
            suggestions.append("⚠️ eGFR提示肾功能减退，建议肾内科就诊评估。")
        
        if not suggestions:
            suggestions.append("✅ 您的各项指标在正常范围内，请继续保持健康的生活方式！")
        
        for suggestion in suggestions:
            st.info(suggestion)
    
    # ========== Tab 2: 指标解读 ==========
    with tab2:
        st.header("📊 您的健康指标")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="WC",
                value=f"{waist:.1f}",
                delta="正常" if waist < wc_threshold else "偏高",
                delta_color="inverse" if waist >= wc_threshold else "off"
            )
            st.caption("参考范围：男<90 / 女<80")
        
        with col2:
            st.metric(
                label="WWI",
                value=f"{wwi:.2f}",
                delta=f"{wwi - 11.0:.2f}" if wwi != 11.0 else None,
                delta_color="inverse"
            )
            st.caption("参考范围：10.5-11.5")
        
        with col3:
            st.metric(
                label="BMI",
                value=f"{bmi:.1f}",
                delta="正常" if 18.5 <= bmi < 24 else ("偏高" if bmi >= 24 else "偏低"),
                delta_color="off"
            )
            st.caption("参考范围：18.5-23.9")
        
        with col4:
            st.metric(
                label="WHtR",
                value=f"{whtr:.3f}",
                delta="正常" if whtr < 0.5 else "偏高",
                delta_color="inverse" if whtr >= 0.5 else "off"
            )
            st.caption("参考范围：<0.5")
        
        with col5:
            st.metric(
                label="eGFR",
                value=f"{egfr:.1f}",
                delta="正常" if egfr >= 90 else ("轻度下降" if egfr >= 60 else "中度下降"),
                delta_color="off" if egfr >= 90 else "inverse"
            )
            st.caption("参考范围：≥90")
        
        st.divider()
        
        # 指标详细说明
        st.subheader("📖 指标说明")
        
        with st.expander("中心性肥胖指标（WC / WHtR / WWI / BMI）", expanded=True):
            st.markdown("""
            **WC（腰围）**：直接反映腹部脂肪堆积。常用阈值：男性≥90 cm、女性≥80 cm。  
            **WHtR（腰围身高比）**：标准化后适合跨人群比较，常用阈值为 0.5。  
            **WWI（体重调整腰围指数）**：WWI = 腰围(cm) / √体重(kg)，在控制体重影响的同时刻画中心性肥胖，可作为补充指标。  
            **BMI（体质指数）**：反映总体肥胖程度，参考范围 18.5–23.9。  
            
            本系统将多指标联合输入模型进行预测，避免单一指标的信息局限。
            """)
        
        with st.expander("CKM综合征"):
            st.markdown("""
            **定义：** 心肾代谢综合征(Cardiovascular-Kidney-Metabolic Syndrome)是美国心脏协会
            2023年提出的整合性概念，强调心血管疾病、慢性肾脏病与代谢危险因素的相互关联。
            
            **分期：**
            - Stage 0: 无代谢风险因素
            - Stage 1: 超重/肥胖或功能失调性脂肪组织
            - Stage 2: 代谢危险因素或中度CKD
            - Stage 3: 亚临床CVD或高危CKD
            - Stage 4: 临床CVD
            
            **参考文献：** Ndumele CE, et al. Circulation. 2023;148:1636-1664.
            """)
    
    # ========== Tab 3: 数据可视化 ==========
    with tab3:
        st.header("📈 您的指标可视化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 雷达图
            categories = ['WC', 'WHtR', 'WWI', 'BMI', '血压', '血糖', 'eGFR']
            
            # 归一化到0-100
            values = [
                min(waist / 120 * 100, 100),
                min(whtr / 0.7 * 100, 100),
                min(wwi / 14 * 100, 100),
                min(bmi / 35 * 100, 100),
                min(sbp / 180 * 100, 100),
                min(fbg / 10 * 100, 100),
                min(egfr / 120 * 100, 100)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='您的指标',
                line_color='#1f77b4'
            ))
            
            # 添加参考范围
            reference = [wc_threshold/120*100, 0.5/0.7*100, 11/14*100, 24/35*100, 140/180*100, 6.1/10*100, 90/120*100]
            fig.add_trace(go.Scatterpolar(
                r=reference,
                theta=categories,
                fill='toself',
                name='参考上限',
                line_color='#ff7f0e',
                opacity=0.3
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="健康指标雷达图"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Indicator distribution (selectable)
            indicator_choice = st.selectbox("Select indicator", ["WC", "WHtR", "WWI", "BMI"])
            np.random.seed(42)
            if indicator_choice == "WC":
                population = np.random.normal(84.0, 9.0, 1000)
                user_value = waist
                x_title = "WC (cm)"
            elif indicator_choice == "WHtR":
                population = np.random.normal(0.53, 0.06, 1000)
                user_value = whtr
                x_title = "WHtR"
            elif indicator_choice == "BMI":
                population = np.random.normal(23.7, 3.5, 1000)
                user_value = bmi
                x_title = "BMI"
            else:
                population = np.random.normal(11.0, 0.8, 1000)
                user_value = wwi
                x_title = "WWI"
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=population,
                nbinsx=30,
                name='Population',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            fig.add_vline(
                x=user_value,
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your {indicator_choice}: {user_value:.2f}"
            )
            
            fig.update_layout(
                title=f"{indicator_choice} distribution (your position)",
                xaxis_title=x_title,
                yaxis_title="Count",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # CKM风险因素贡献
        st.subheader("CKM风险因素贡献度（示意）")
        st.caption("提示：该图用于展示指标偏离程度的相对贡献，不代表因果权重或临床效应大小。")
        
        central_score = (
            max(0, (waist - wc_threshold) * 0.6)
            + max(0, (whtr - 0.5) * 120)
            + max(0, (wwi - 10.5) * 8)
            + max(0, (bmi - 24) * 2)
        )
        
        contributions = {
            '中心性肥胖指标升高': central_score,
            '年龄': max(0, (age - 60) * 0.8),
            '高血压': max(0, (sbp - 120) * 0.3),
            '血糖升高': max(0, (fbg - 5.0) * 8),
            'eGFR下降': max(0, (90 - egfr) * 0.5),
        }
        
        fig = px.bar(
            x=list(contributions.values()),
            y=list(contributions.keys()),
            orientation='h',
            title="各因素对CKM风险的贡献（示意）",
            labels={'x': '风险贡献度', 'y': '风险因素'},
            color=list(contributions.values()),
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== Tab 4: 关于项目 ==========
    with tab4:
        st.header("ℹ️ 关于本项目")
        
        st.markdown("""
        ### 研究背景
        
        本项目是北京大学医学部**健康数据科学的Python语言编程基础**课程的结课作业。
        研究基于中国南方某社区8742名老年人的横断面数据，系统比较多种中心性肥胖指标
        （WC、WHtR、WWI、BMI）与心肾代谢综合征(CKM)的关联与预测价值。
        
        ### 研究方法
        
        1. **关联分析**：多因素Logistic回归、限制性立方样条、亚组分析
        2. **预测建模**：LASSO特征选择 + 多模型比较（Logistic Regression, Random Forest, 
           XGBoost, LightGBM, MLP）
        3. **模型评估**：10折交叉验证、ROC-AUC、校准曲线、DCA决策曲线
        4. **模型解释**：SHAP可解释性分析
        
        ### 主要发现
        
        - WC与WHtR对CKM的单指标判别效能优于WWI
        - WWI与CKM呈显著正相关，作为补充指标可提升联合模型表现
        - XGBoost模型预测CKM的AUC达到0.93左右
        - SHAP分析显示WC/WHtR贡献更大，WWI仍提供补充信息
        
        ### 作者信息
        
        - **学生**：郑赫
        - **学号**：2511110259
        - **学院**：第一临床医学院
        - **个人主页**：[https://guanshanyue1999.github.io/](https://guanshanyue1999.github.io/)
        
        ### 免责声明
        
        ⚠️ 本工具仅供学术研究和健康科普使用，不能替代专业医疗诊断。
        如有健康问题，请咨询专业医生。
        """)
        
        st.divider()
        
        # 参考文献
        st.subheader("📚 主要参考文献")
        st.markdown("""
        1. Ndumele CE, et al. Cardiovascular-Kidney-Metabolic Health: A Presidential Advisory 
           From the American Heart Association. *Circulation*. 2023;148:1636-1664.
        2. Park Y, et al. A Novel Adiposity Index as an Integrated Predictor of Cardiometabolic 
           Disease Morbidity and Mortality. *Sci Rep*. 2018;8:16753.
        3. Ding C, et al. Association of weight-adjusted-waist index with all-cause and 
           cardiovascular mortality. *Nutr Metab Cardiovasc Dis*. 2022;32:1210-1217.
        """)

# ============================================================================
# 运行应用
# ============================================================================
if __name__ == "__main__":
    main()
