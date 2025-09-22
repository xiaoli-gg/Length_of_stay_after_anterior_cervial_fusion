import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import sklearn

# 1) 载入模型（缓存 + 兜底）
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    try:
        mdl = joblib.load(path)
        return mdl
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        st.stop()

model = load_model("rf.pkl")

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
        "type": "categorical",
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
    # 例：数值特征
    # "Age": {"type": "numerical", "min": 60, "max": 95, "default": 72},
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
    # "Age",
]

st.title("Prediction Model with SHAP Visualization")
st.header("Enter feature values")

# —— 构造一个 background_df：用各特征默认值编码成一行 —— #
def build_background_df():
    row = []
    for feat in feature_order:
        spec = feature_specs[feat]
        if spec["type"].startswith("categor"):
            default_label = spec["default"]
            row.append(float(spec["options"][default_label]))
        else:
            row.append(float(spec["default"]))
    return pd.DataFrame([row], columns=feature_order).astype(float)

background_df = build_background_df()

# 4) 采集输入
numeric_values = []
for feat in feature_order:
    spec = feature_specs[feat]
    if spec["type"].startswith("categor"):
        labels = list(spec["options"].keys())
        idx = labels.index(spec["default"])
        choice = st.selectbox(feat, options=labels, index=idx, key=feat)
        code_val = spec["options"][choice]
        numeric_values.append(code_val)
    else:
        v = st.number_input(
            f"{feat} ({spec['min']}–{spec['max']})",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
            key=feat,
        )
        numeric_values.append(float(v))

X_df = pd.DataFrame([numeric_values], columns=feature_order)

# 强校验 + 统一 float
bad_cols = [c for c in X_df.columns if not np.issubdtype(X_df[c].dtype, np.number)]
if bad_cols:
    st.error(f"这些特征不是数值型：{bad_cols}。请检查编码或默认值。")
    st.stop()
X_df = X_df.astype(float)

if st.button("Predict"):
    # 预测（优先概率）
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)[0]
        cls = list(getattr(model, "classes_", range(len(proba))))
        pos_idx = cls.index(1) if 1 in cls else int(np.argmax(proba))
        pos_proba = float(proba[pos_idx]) * 100
        pred_text = f"Based on feature values, predicted possibility of AKI is {pos_proba:.2f}%"
    else:
        y_pred = model.predict(X_df)[0]
        pred_text = f"Model predicts class: {y_pred}（模型不提供概率）"

    # 文本展示
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, pred_text, fontsize=16, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    st.pyplot(fig)

    # --- SHAP 可视化：使用 interventional + 概率空间 ---
    try:
        explainer = shap.TreeExplainer(
            model,
            data=background_df,                           # 提供背景数据
            feature_perturbation="interventional",       # 关键：允许 probability 输出
            model_output="probability"                   # 在概率空间解释
        )
        shap_values = explainer.shap_values(X_df)

        if isinstance(shap_values, list):
            classes = list(getattr(model, "classes_", range(len(shap_values))))
            class_idx = classes.index(1) if 1 in classes else (1 if len(shap_values) > 1 else 0)
            sv_row = shap_values[class_idx][0]
            expected = explainer.expected_value[class_idx]
        else:
            sv_row = shap_values[0]
            expected = explainer.expected_value

        # 力图
        shap.force_plot(
            base_value=expected,
            shap_values=sv_row,
            features=X_df.iloc[0, :],
            matplotlib=True,
        )
        st.pyplot(plt.gcf())

        # 条形图
        st.write("Top features (bar):")
        shap.bar_plot(sv_row, feature_names=list(X_df.columns), show=False)
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"SHAP 可视化失败：{e}")
        st.info("若仍报错，可：1) 升级/固定 SHAP 版本；2) 改用 shap.Explainer(masker=background_df, model_output='probability')；"
                "3) 或退回 model_output='raw' 并在图注说明是 log-odds。")
