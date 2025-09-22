import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 1) 载入模型（加兜底）
try:
    model = joblib.load("rf.pkl")
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

# 2) 特征映射
feature_specs = {
    "Preoperative_waiting_time_plus_7d": {
        "type": "categorical",
        "options": {"No delay": 0, "Delay > 7 days": 1},
        "default": "No delay",
    },
    "Cardiovascular_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
    },
    "Lung_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
    },
    "Operation_time_plus_230min": {
        "type": "categorical",
        "options": {"≤230 min": 0, ">230 min": 1},
        "default": "≤230 min",
    },
    "NO._Levels": {
        "type": "categorical",  # 修正拼写
        "options": {"Level 2": 0, "Level 3": 1, "Level 4": 2, "Level >4": 3},
        "default": "Level 2",
    },
    "Infectious_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
    },
    "Major_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
    },
}

# 3) 特征顺序（与训练一致）
feature_order = [
    "Preoperative_waiting_time_plus_7d",
    "Cardiovascular_comorbidities",
    "Lung_comorbidities",
    "Operation_time_plus_230min",
    "NO._Levels",
    "Infectious_complications",
    "Major_complications",
]

st.title("Prediction Model with SHAP Visualization")
st.header("Enter feature values")

# 4) 采集输入
numeric_values = []
for feat in feature_order:
    spec = feature_specs[feat]
    if spec["type"].startswith("categor"):
        labels = list(spec["options"].keys())
        idx = labels.index(spec["default"])
        choice = st.selectbox(feat, options=labels, index=idx)
        code_val = spec["options"][choice]
        numeric_values.append(code_val)
    else:
        v = st.number_input(
            f"{feat} ({spec['min']}–{spec['max']})",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
        )
        numeric_values.append(float(v))

X_df = pd.DataFrame([numeric_values], columns=feature_order)

if st.button("Predict"):
    # 预测
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        proba = model.predict_proba(X_df)[0]
        cls = list(getattr(model, "classes_", range(len(proba))))
        pos_idx = cls.index(1) if 1 in cls else int(np.argmax(proba))
        pos_proba = float(proba[pos_idx]) * 100
        pred_text = f"Based on feature values, predicted possibility of AKI is {pos_proba:.2f}%"
    else:
        y_pred = model.predict(X_df)[0]
        pred_text = f"Model predicts class: {y_pred}（模型不提供概率）"

    # 文本展示（直接渲染，不落盘）
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, pred_text, fontsize=16, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    st.pyplot(fig)

    # SHAP（尽量在概率空间解释）
    try:
        explainer = shap.TreeExplainer(model, model_output="probability")
        shap_values = explainer.shap_values(X_df)

        if isinstance(shap_values, list):
            classes = list(getattr(model, "classes_", range(len(shap_values))))
            class_idx = classes.index(1) if 1 in classes else (1 if len(shap_values) > 1 else 0)
            sv_row = shap_values[class_idx][0]
            expected = explainer.expected_value[class_idx]
        else:
            sv_row = shap_values[0]
            expected = explainer.expected_value

        # 力图（matplotlib 渲染）
        shap.force_plot(
            base_value=expected,
            shap_values=sv_row,
            features=X_df.iloc[0, :],
            matplotlib=True,
        )
        st.pyplot(plt.gcf())

        # 条形图（传入 list 名称）
        st.write("Top features (bar):")
        shap.bar_plot(sv_row, feature_names=list(X_df.columns), show=False)
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"SHAP 可视化失败：{e}")
        st.info("若模型不是树模型，可改用 KernelExplainer 或 LinearExplainer；或检查 SHAP 版本。")
