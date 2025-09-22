import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

# 设置页面配置
st.set_page_config(
    page_title="AKI Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 抑制警告信息
warnings.filterwarnings('ignore')

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 20px 0;
}
.feature-info {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# 1) 模型加载函数
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """加载训练好的模型"""
    try:
        model = joblib.load(path)
        st.success("✅ 模型加载成功！")
        return model
    except FileNotFoundError:
        st.error(f"❌ 找不到模型文件：{path}")
        st.info("请确保模型文件 'rf.pkl' 在正确的路径中")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.stop()

# 2) 特征规格定义
feature_specs = {
    "Preoperative_waiting_time_plus_7d": {
        "type": "categorical",
        "options": {"No delay": 0, "Delay > 7 days": 1},
        "default": "No delay",
        "description": "术前等待时间是否超过7天"
    },
    "Cardiovascular_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "是否有心血管合并症"
    },
    "Lung_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "是否有肺部合并症"
    },
    "Operation_time_plus_230min": {
        "type": "categorical",
        "options": {"≤230 min": 0, ">230 min": 1},
        "default": "≤230 min",
        "description": "手术时间是否超过230分钟"
    },
    "NO._Levels": {
        "type": "categorical",
        "options": {"Level 2": 0, "Level 3": 1, "Level 4": 2, "Level >4": 3},
        "default": "Level 2",
        "description": "手术级别"
    },
    "Infectious_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "是否有感染性并发症"
    },
    "Major_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "是否有主要并发症"
    }
}

# 3) 特征顺序（与训练时保持一致）
feature_order = [
    "Preoperative_waiting_time_plus_7d",
    "Cardiovascular_comorbidities", 
    "Lung_comorbidities",
    "Operation_time_plus_230min",
    "NO._Levels",
    "Infectious_complications",
    "Major_complications"
]

# 4) 构建背景数据集
@st.cache_data
def build_background_df():
    """构建用于SHAP解释的背景数据集"""
    row = []
    for feat in feature_order:
        spec = feature_specs[feat]
        if spec["type"] == "categorical":
            default_label = spec["default"]
            row.append(float(spec["options"][default_label]))
        else:
            row.append(float(spec["default"]))
    return pd.DataFrame([row], columns=feature_order).astype(float)

# 5) 主界面
def main():
    # 页面标题
    st.markdown('<h1 class="main-header">🏥 急性肾损伤(AKI)预测模型</h1>', unsafe_allow_html=True)
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown("### 📊 模型信息")
        st.info("本模型用于预测患者发生急性肾损伤(AKI)的风险")
        
        st.markdown("### 📋 使用说明")
        st.markdown("""
        1. 在左侧输入患者的临床特征
        2. 点击"开始预测"按钮
        3. 查看预测结果和SHAP解释
        """)
    
    # 加载模型
    try:
        model = load_model("rf.pkl")
    except:
        st.stop()
    
    # 构建背景数据
    background_df = build_background_df()
    
    # 特征输入界面
    st.markdown('<h2 class="sub-header">📝 请输入患者特征</h2>', unsafe_allow_html=True)
    
    # 使用列布局
    col1, col2 = st.columns(2)
    
    numeric_values = []
    
    for i, feat in enumerate(feature_order):
        spec = feature_specs[feat]
        
        # 交替使用两列
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            if spec["type"] == "categorical":
                labels = list(spec["options"].keys())
                idx = labels.index(spec["default"])
                
                choice = st.selectbox(
                    feat.replace("_", " ").title(),
                    options=labels,
                    index=idx,
                    key=feat,
                    help=spec.get("description", "")
                )
                
                code_val = spec["options"][choice]
                numeric_values.append(float(code_val))
            else:
                # 数值型特征的处理
                v = st.number_input(
                    f"{feat.replace('_', ' ').title()} ({spec['min']}–{spec['max']})",
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    key=feat,
                    help=spec.get("description", "")
                )
                numeric_values.append(float(v))
    
    # 预测按钮
    st.markdown("---")
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
    
    with col_pred2:
        predict_button = st.button("🔮 开始预测", type="primary", use_container_width=True)
    
    # 预测逻辑
    if predict_button:
        # 构建输入数据
        X_df = pd.DataFrame([numeric_values], columns=feature_order).astype(float)
        
        # 数据验证
        if X_df.isnull().any().any():
            st.error("❌ 输入数据中存在缺失值，请检查所有特征是否已正确填写")
            return
        
        # 执行预测
        with st.spinner("正在进行预测..."):
            try:
                # 预测概率
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_df)[0]
                    classes = getattr(model, "classes_", [0, 1])
                    
                    if 1 in classes:
                        pos_idx = list(classes).index(1)
                        pos_proba = float(proba[pos_idx]) * 100
                    else:
                        pos_proba = float(np.max(proba)) * 100
                    
                    # 显示预测结果
                    st.markdown('<h2 class="sub-header">🎯 预测结果</h2>', unsafe_allow_html=True)
                    
                    # 根据概率设置风险等级和颜色
                    if pos_proba < 30:
                        risk_level = "低风险"
                        color = "#28a745"  # 绿色
                    elif pos_proba < 70:
                        risk_level = "中等风险" 
                        color = "#ffc107"  # 黄色
                    else:
                        risk_level = "高风险"
                        color = "#dc3545"  # 红色
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: {color};">AKI发生概率: {pos_proba:.2f}%</h3>
                        <h4 style="color: {color};">风险等级: {risk_level}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # 仅分类预测
                    y_pred = model.predict(X_df)[0]
                    result_text = "高风险" if y_pred == 1 else "低风险"
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>预测结果: {result_text}</h3>
                        <p>注：当前模型不提供概率预测</p>
                    </div>
                    """, unsafe_allow_html=True)
                    pos_proba = None
            
            except Exception as e:
                st.error(f"❌ 预测过程中出现错误：{e}")
                return
        
        # SHAP可视化
        st.markdown('<h2 class="sub-header">📊 模型解释 (SHAP)</h2>', unsafe_allow_html=True)
        
        try:
            with st.spinner("正在生成SHAP解释..."):
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model, data=background_df)
                shap_values = explainer.shap_values(X_df)
                
                # 处理多类别情况
                if isinstance(shap_values, list):
                    classes = getattr(model, "classes_", list(range(len(shap_values))))
                    if 1 in classes:
                        class_idx = list(classes).index(1)
                    else:
                        class_idx = 1 if len(shap_values) > 1 else 0
                    
                    sv_row = shap_values[class_idx][0]
                    expected = explainer.expected_value[class_idx]
                else:
                    sv_row = shap_values[0]
                    expected = explainer.expected_value
                
                # SHAP力图
                st.markdown("#### 🔍 特征贡献分析 (Force Plot)")
                fig_force = plt.figure(figsize=(12, 3))
                shap.force_plot(
                    base_value=expected,
                    shap_values=sv_row,
                    features=X_df.iloc[0, :],
                    feature_names=[f.replace("_", " ") for f in X_df.columns],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig_force)
                plt.close()
                
                # SHAP条形图
                st.markdown("#### 📈 特征重要性 (Bar Plot)")
                fig_bar = plt.figure(figsize=(10, 6))
                shap.bar_plot(
                    sv_row, 
                    feature_names=[f.replace("_", " ") for f in X_df.columns],
                    show=False
                )
                plt.title("Feature Importance (SHAP Values)")
                st.pyplot(fig_bar)
                plt.close()
                
                # 特征解释说明
                st.markdown("#### 💡 结果解释")
                st.markdown("""
                <div class="feature-info">
                <b>如何理解SHAP图表：</b><br>
                • <b>Force Plot</b>: 显示各特征如何推动预测远离或接近基线值<br>
                • <b>Bar Plot</b>: 显示各特征对预测结果的绝对贡献大小<br>
                • <b>红色</b>: 增加AKI风险的因素<br>
                • <b>蓝色</b>: 降低AKI风险的因素
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"⚠️ SHAP可视化生成失败：{e}")
            st.info("SHAP解释功能暂时不可用，但预测结果仍然有效")
    
    # 页面底部信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    ⚠️ 本工具仅供医疗辅助参考，不能替代专业医疗判断
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
