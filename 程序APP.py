import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1) 载入模型
model = joblib.load("rf.pkl")

# 2) 特征映射（显示标签 -> 训练时的数值编码）
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
        "type": "categororical",  # ← 如果写错会报错，注意单词
        "options": {
            "Level 2": 0,
            "Level 3": 1,
            "Level 4": 2,
            "Level >4": 3
        },
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
    # 若有数值特征，像这样：
    # "Age": {"type": "numerical", "min": 60, "max": 95, "default": 72},
}

# 3) 训练时的特征顺序（必须与训练完全一致！）
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

# 4) 采集输入：显示文字 -> 数值编码；并按 feature_order 组织
numeric_values = []
for feat in feature_order:
    spec = feature_specs[feat]
    if spec["type"].startswith("categor"):  # categorical
        labels = list(spec["options"].keys())
        idx = labels.index(spec["default"])
        choice = st.selectbox(feat, options=labels, index=idx)
        code_val = spec["options"][choice]
        numeric_values.append(code_val)
    else:  # numerical
        v = st.number_input(
            f"{feat} ({spec['min']}–{spec['max']})",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
        )
        numeric_values.append(float(v))

# 5) 用 DataFrame（含列名、顺序正确）喂给模型
X_df = pd.DataFrame([numeric_values], columns=feature_order)

# ---------------------------------
# 预测 & 文本输出 & SHAP
# ---------------------------------
if st.button("Predict"):
    # 预测
    y_pred = model.predict(X_df)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)[0]
        # 若标签是 {0,1}，正类概率一般是 proba[1]
        if hasattr(model, "classes_"):
            cls = list(model.classes_)
            pos_idx = cls.index(1) if 1 in cls else np.argmax(proba)
        else:
            pos_idx = 1 if len(proba) > 1 else 0
        pos_proba = float(proba[pos_idx]) * 100
    else:
        pos_proba = np.nan

    # 结果文案
    text = f"Based on feature values, predicted possibility of AKI is {pos_proba:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha="center", va="center",
            fontname="Times New Roman", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig("prediction_text.png", bbox_inches="tight", dpi=300)
    st.image("prediction_text.png")

    # --- SHAP 可视化（RandomForest 二分类） ---
    try:
        # 对树模型使用 TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        # 兼容 shap 的两种返回：list（二分类按类返回）或 ndarray
        if isinstance(shap_values, list):
            # 取正类索引
            if hasattr(model, "classes_") and 1 in list(model.classes_):
                class_idx = list(model.classes_).index(1)
            else:
                class_idx = 1 if len(shap_values) > 1 else 0
            sv_row = shap_values[class_idx][0]
            expected = explainer.expected_value[class_idx]
        else:
            sv_row = shap_values[0]
            expected = explainer.expected_value

        # 生成力图并保存
        shap_fig = shap.force_plot(
            base_value=expected,
            shap_values=sv_row,
            features=X_df.iloc[0, :],
            matplotlib=True,
        )
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=600)
        st.image("shap_force_plot.png")

        # 还可以加条形图/蜂群图（单样本不适合 summary_plot）
        st.write("Top features (bar):")
        shap.bar_plot(sv_row, feature_names=X_df.columns, show=False)
        plt.tight_layout()
        plt.savefig("shap_bar.png", bbox_inches="tight", dpi=300)
        st.image("shap_bar.png")

    except Exception as e:
        st.error(f"SHAP plotting failed: {e}")
        st.info("确保当前模型为树模型或使用 KernelExplainer/LinearExplainer 适配其它模型。")
