import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings

# Page configuration
st.set_page_config(
    page_title="Length of Stay Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom CSS styling
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

# 1) Model loading function with better error handling
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Load the trained model"""
    try:
        model = joblib.load(path)
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {path}")
        st.info("Please ensure the model file 'rf.pkl' is in the correct path")
        st.stop()
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# 2) Feature specifications
feature_specs = {
    "Preoperative_waiting_time_plus_7d": {
        "type": "categorical",
        "options": {"No delay": 0, "Delay > 7 days": 1},
        "default": "No delay",
        "description": "Whether preoperative waiting time exceeds 7 days"
    },
    "Cardiovascular_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Presence of cardiovascular comorbidities"
    },
    "Lung_comorbidities": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Presence of lung comorbidities"
    },
    "Operation_time_plus_230min": {
        "type": "categorical",
        "options": {"≤230 min": 0, ">230 min": 1},
        "default": "≤230 min",
        "description": "Whether operation time exceeds 230 minutes"
    },
    "NO._Levels": {
        "type": "categorical",
        "options": {"Level 2": 0, "Level 3": 1, "Level 4": 2, "Level >4": 3},
        "default": "Level 2",
        "description": "Number of surgical levels"
    },
    "Infectious_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Presence of infectious complications"
    },
    "Major_complications": {
        "type": "categorical",
        "options": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Presence of major complications"
    }
}

# 3) Feature order (consistent with training)
feature_order = [
    "Preoperative_waiting_time_plus_7d",
    "Cardiovascular_comorbidities", 
    "Lung_comorbidities",
    "Operation_time_plus_230min",
    "NO._Levels",
    "Infectious_complications",
    "Major_complications"
]

# 4) Build background dataset for SHAP
@st.cache_data
def build_background_df():
    """Build background dataset for SHAP interpretation"""
    row = []
    for feat in feature_order:
        spec = feature_specs[feat]
        if spec["type"] == "categorical":
            default_label = spec["default"]
            row.append(float(spec["options"][default_label]))
        else:
            row.append(float(spec["default"]))
    return pd.DataFrame([row], columns=feature_order).astype(float)

# 5) Input processing function
def process_user_inputs():
    """Process user inputs and convert to numerical values"""
    numeric_values = []
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    for i, feat in enumerate(feature_order):
        spec = feature_specs[feat]
        
        # Alternate between columns
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            if spec["type"] == "categorical":
                labels = list(spec["options"].keys())
                idx = labels.index(spec["default"])
                
                choice = st.selectbox(
                    feat.replace("_", " ").replace("NO.", "Number of").title(),
                    options=labels,
                    index=idx,
                    key=feat,
                    help=spec.get("description", "")
                )
                
                # Convert choice to numeric value
                code_val = float(spec["options"][choice])
                numeric_values.append(code_val)
            else:
                # For numerical features
                v = st.number_input(
                    f"{feat.replace('_', ' ').title()} ({spec['min']}–{spec['max']})",
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    key=feat,
                    help=spec.get("description", "")
                )
                numeric_values.append(float(v))
    
    return numeric_values

# 6) Main application
def main():
    # Page title
    st.markdown('''
    <h1 class="main-header">🏥 Length of Stay Prediction Model</h1>
    <p style="text-align: center; font-size: 1.1rem; color: #666;">
    Interpretable Random Forest Model for Predicting Length of Stay After<br>
    First Elective Open Anterior Cervical Fusion in Elderly Patients
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info("This model predicts the length of hospital stay for elderly patients undergoing anterior cervical fusion surgery")
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter patient clinical features on the main panel
        2. Click the "Predict Length of Stay" button
        3. Review the prediction results and SHAP explanations
        """)
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Target Population**: Elderly patients (≥60 years)  
        **Procedure**: First elective open anterior cervical fusion  
        **Outcome**: Hospital length of stay  
        **Model Type**: Random Forest with SHAP interpretability
        """)
    
    # Load model
    try:
        model = load_model("rf.pkl")
    except:
        st.stop()
    
    # Build background data
    background_df = build_background_df()
    
    # Feature input section
    st.markdown('<h2 class="sub-header">📝 Patient Clinical Features</h2>', unsafe_allow_html=True)
    
    # Process user inputs
    numeric_values = process_user_inputs()
    
    # Create input DataFrame
    X_df = pd.DataFrame([numeric_values], columns=feature_order).astype(float)
    
    # Prediction button
    st.markdown("---")
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
    
    with col_pred2:
        predict_button = st.button("🔮 Predict Length of Stay", type="primary", use_container_width=True)
    
    # Prediction logic
    if predict_button:
        # Data validation
        if X_df.isnull().any().any():
            st.error("❌ Missing values detected in input data. Please check all features are properly filled.")
            return
        
        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                # Prediction with probability (if available)
                if hasattr(model, "predict_proba"):
                    # Get prediction probabilities
                    proba = model.predict_proba(X_df)[0]
                    classes = getattr(model, "classes_", list(range(len(proba))))
                    
                    # Get predicted class
                    pred_class = model.predict(X_df)[0]
                    pred_proba = float(np.max(proba)) * 100
                    
                    # Display prediction results
                    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    # Interpret the prediction (assuming binary classification: 0=short stay, 1=long stay)
                    if pred_class == 0:
                        stay_category = "Short Stay"
                        color = "#28a745"  # Green
                        interpretation = "Patient is predicted to have a shorter length of stay"
                    else:
                        stay_category = "Long Stay" 
                        color = "#dc3545"  # Red
                        interpretation = "Patient is predicted to have a longer length of stay"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: {color};">Predicted Category: {stay_category}</h3>
                        <h4 style="color: {color};">Confidence: {pred_proba:.2f}%</h4>
                        <p>{interpretation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # Classification only
                    y_pred = model.predict(X_df)[0]
                    result_text = "Long Stay" if y_pred == 1 else "Short Stay"
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Length of Stay: {result_text}</h3>
                        <p>Note: Current model does not provide probability predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please check your input values and try again.")
                return
        
        # SHAP visualization
        st.markdown('<h2 class="sub-header">📊 Model Interpretation (SHAP)</h2>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Generating SHAP explanations..."):
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model, data=background_df)
                shap_values = explainer.shap_values(X_df)
                
                # Handle multi-class case
                if isinstance(shap_values, list):
                    classes = getattr(model, "classes_", list(range(len(shap_values))))
                    # Use the positive class (1) if available, otherwise use the predicted class
                    if len(classes) > 1:
                        class_idx = 1  # Assuming 1 represents "long stay"
                    else:
                        class_idx = 0
                    
                    sv_row = shap_values[class_idx][0]
                    expected = explainer.expected_value[class_idx]
                else:
                    sv_row = shap_values[0]
                    expected = explainer.expected_value
                
                # Create feature names for display
                feature_names_display = [f.replace("_", " ").replace("NO.", "Num.") for f in X_df.columns]
                
                # SHAP force plot
                st.markdown("#### 🔍 Feature Contribution Analysis (Force Plot)")
                fig_force = plt.figure(figsize=(12, 3))
                shap.force_plot(
                    base_value=expected,
                    shap_values=sv_row,
                    features=X_df.iloc[0, :],
                    feature_names=feature_names_display,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig_force)
                plt.close()
                
                # SHAP bar plot
                st.markdown("#### 📈 Feature Importance (Bar Plot)")
                fig_bar = plt.figure(figsize=(10, 6))
                shap.bar_plot(
                    sv_row, 
                    feature_names=feature_names_display,
                    show=False
                )
                plt.title("Feature Importance for Length of Stay Prediction")
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
                
                # Interpretation guide
                st.markdown("#### 💡 How to Interpret Results")
                st.markdown("""
                <div class="feature-info">
                <b>Understanding SHAP Plots:</b><br>
                • <b>Force Plot</b>: Shows how each feature pushes the prediction away from or towards the baseline<br>
                • <b>Bar Plot</b>: Shows the absolute contribution magnitude of each feature<br>
                • <b>Red/Positive values</b>: Features that increase the likelihood of longer hospital stay<br>
                • <b>Blue/Negative values</b>: Features that decrease the likelihood of longer hospital stay<br>
                • <b>Baseline value</b>: Average prediction across all patients
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"⚠️ SHAP visualization failed: {e}")
            st.info("SHAP interpretation is temporarily unavailable, but the prediction result is still valid.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    ⚠️ This tool is for clinical decision support only and should not replace professional medical judgment.<br>
    Model performance should be validated in your specific clinical setting.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
