import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings

# Page configuration
st.set_page_config(
    page_title="Length of Stay Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.section-header {
    font-size: 1.4rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
    padding: 10px 0;
    border-bottom: 2px solid #f0f0f0;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.risk-box {
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
}
.low-risk {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #c3e6cb;
}
.medium-risk {
    background-color: #fff3cd;
    color: #856404;
    border: 2px solid #ffeaa7;
}
.high-risk {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
.input-section {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    height: 100vh;
    overflow-y: auto;
}
.output-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    height: 100vh;
    overflow-y: auto;
}
.feature-group {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Model loading function
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Load the trained model"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# Feature specifications
feature_specs = {
    "Preoperative_waiting_time_plus_7d": {
        "type": "categorical",
        "options": ["No delay", "Delay > 7 days"],
        "mapping": {"No delay": 0, "Delay > 7 days": 1},
        "default": "No delay",
        "description": "Preoperative waiting time",
        "group": "Preoperative Factors"
    },
    "Cardiovascular_comorbidities": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Cardiovascular comorbidities",
        "group": "Comorbidities"
    },
    "Lung_comorbidities": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Lung comorbidities",
        "group": "Comorbidities"
    },
    "Operation_time_plus_230min": {
        "type": "categorical",
        "options": ["≤230 min", ">230 min"],
        "mapping": {"≤230 min": 0, ">230 min": 1},
        "default": "≤230 min",
        "description": "Operation duration",
        "group": "Surgical Factors"
    },
    "NO._Levels": {
        "type": "categorical",
        "options": ["Level 2", "Level 3", "Level 4", "Level >4"],
        "mapping": {"Level 2": 0, "Level 3": 1, "Level 4": 2, "Level >4": 3},
        "default": "Level 2",
        "description": "Number of surgical levels",
        "group": "Surgical Factors"
    },
    "Infectious_complications": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Infectious complications",
        "group": "Complications"
    },
    "Major_complications": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Major complications",
        "group": "Complications"
    }
}

# Feature order
feature_order = [
    "Preoperative_waiting_time_plus_7d",
    "Cardiovascular_comorbidities", 
    "Lung_comorbidities",
    "Operation_time_plus_230min",
    "NO._Levels",
    "Infectious_complications",
    "Major_complications"
]

def create_probability_gauge(probability):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probability of Long Stay (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 16}
    )
    return fig

def create_probability_bar(probability):
    """Create a horizontal bar chart for probability"""
    fig = go.Figure()
    
    # Determine color based on probability
    if probability < 30:
        color = '#28a745'
        risk_level = 'Low Risk'
    elif probability < 70:
        color = '#ffc107'
        risk_level = 'Medium Risk'
    else:
        color = '#dc3545'
        risk_level = 'High Risk'
    
    fig.add_trace(go.Bar(
        y=['Probability'],
        x=[probability],
        orientation='h',
        marker_color=color,
        text=[f'{probability:.1f}%'],
        textposition='inside',
        textfont=dict(size=20, color='white'),
        name=risk_level
    ))
    
    fig.update_layout(
        title=f"Length of Stay Risk Assessment: {risk_level}",
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        yaxis=dict(showticklabels=False),
        height=150,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    return fig

def convert_inputs_to_dataframe(user_inputs):
    """Convert user inputs to numerical DataFrame"""
    numeric_values = []
    
    for feat in feature_order:
        spec = feature_specs[feat]
        user_choice = user_inputs[feat]
        
        if spec["type"] == "categorical":
            numeric_value = spec["mapping"][user_choice]
            numeric_values.append(float(numeric_value))
        else:
            numeric_values.append(float(user_choice))
    
    df = pd.DataFrame([numeric_values], columns=feature_order)
    return df.astype(float)

def main():
    # Page title
    st.markdown('<h1 class="main-header">🏥 Length of Stay Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Interpretable Random Forest Model for Predicting Length of Stay After First Elective Open Anterior Cervical Fusion in Elderly Patients</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model("rf.pkl")
    
    # Create main layout: Left (Input) | Right (Output)
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # =========================
    # LEFT COLUMN: INPUT SECTION  
    # =========================
    with left_col:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📝 Patient Information</h2>', unsafe_allow_html=True)
        
        user_inputs = {}
        
        # Group features by category
        feature_groups = {}
        for feat in feature_order:
            group = feature_specs[feat]["group"]
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(feat)
        
        # Display features by group
        for group_name, features in feature_groups.items():
            st.markdown(f'<div class="feature-group">', unsafe_allow_html=True)
            st.markdown(f"**{group_name}**")
            
            for feat in features:
                spec = feature_specs[feat]
                
                if spec["type"] == "categorical":
                    try:
                        default_idx = spec["options"].index(spec["default"])
                    except ValueError:
                        default_idx = 0
                    
                    choice = st.selectbox(
                        spec["description"],
                        options=spec["options"],
                        index=default_idx,
                        key=f"input_{feat}"
                    )
                    user_inputs[feat] = choice
                else:
                    v = st.number_input(
                        spec["description"],
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["default"]),
                        key=f"input_{feat}"
                    )
                    user_inputs[feat] = v
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        st.markdown("---")
        predict_button = st.button("🔮 Predict Length of Stay", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================
    # RIGHT COLUMN: OUTPUT SECTION
    # =========================
    with right_col:
        st.markdown('<div class="output-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        if predict_button:
            try:
                # Convert inputs to DataFrame
                X_df = convert_inputs_to_dataframe(user_inputs)
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    if hasattr(model, "predict_proba"):
                        # Get probabilities
                        proba = model.predict_proba(X_df)[0]
                        classes = getattr(model, "classes_", [0, 1])
                        
                        # Get predicted class and probabilities
                        pred_class = model.predict(X_df)[0]
                        
                        # Assuming binary classification: 0=short stay, 1=long stay
                        if 1 in classes:
                            long_stay_prob = float(proba[list(classes).index(1)]) * 100
                            short_stay_prob = float(proba[list(classes).index(0)]) * 100
                        else:
                            long_stay_prob = float(proba[1]) * 100 if len(proba) > 1 else float(proba[0]) * 100
                            short_stay_prob = 100 - long_stay_prob
                        
                        # Display main prediction result
                        if pred_class == 0:
                            outcome = "Short Stay"
                            main_prob = short_stay_prob
                        else:
                            outcome = "Long Stay"
                            main_prob = long_stay_prob
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted Outcome</h2>
                            <h1>{outcome}</h1>
                            <h3>Confidence: {main_prob:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk assessment
                        if long_stay_prob < 30:
                            risk_class = "low-risk"
                            risk_text = "Low Risk of Extended Stay"
                        elif long_stay_prob < 70:
                            risk_class = "medium-risk" 
                            risk_text = "Medium Risk of Extended Stay"
                        else:
                            risk_class = "high-risk"
                            risk_text = "High Risk of Extended Stay"
                        
                        st.markdown(f"""
                        <div class="risk-box {risk_class}">
                            {risk_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed probabilities
                        st.markdown("### 📊 Detailed Probabilities")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="Short Stay Probability",
                                value=f"{short_stay_prob:.1f}%",
                                delta=f"{short_stay_prob - 50:.1f}%" if short_stay_prob != 50 else None
                            )
                        with col2:
                            st.metric(
                                label="Long Stay Probability", 
                                value=f"{long_stay_prob:.1f}%",
                                delta=f"{long_stay_prob - 50:.1f}%" if long_stay_prob != 50 else None
                            )
                        
                        # Probability visualization - Gauge
                        st.markdown("### 🎯 Risk Assessment Gauge")
                        gauge_fig = create_probability_gauge(long_stay_prob)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Probability visualization - Bar
                        st.markdown("### 📈 Probability Distribution")
                        bar_fig = create_probability_bar(long_stay_prob)
                        st.plotly_chart(bar_fig, use_container_width=True)
                        
                        # Interpretation guide
                        st.markdown("### 💡 Clinical Interpretation")
                        if long_stay_prob < 30:
                            interpretation = "Patient is likely to have a normal length of stay. Standard discharge planning is appropriate."
                        elif long_stay_prob < 70:
                            interpretation = "Patient has moderate risk for extended stay. Consider enhanced monitoring and discharge planning."
                        else:
                            interpretation = "Patient is at high risk for extended hospitalization. Recommend proactive interventions and multidisciplinary planning."
                        
                        st.info(interpretation)
                    
                    else:
                        # Classification only (no probabilities)
                        y_pred = model.predict(X_df)[0]
                        outcome = "Long Stay" if y_pred == 1 else "Short Stay"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted Outcome</h2>
                            <h1>{outcome}</h1>
                            <p>Note: Probability information not available</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.error("Please check your inputs and try again.")
        
        else:
            # Default state when no prediction has been made
            st.info("👈 Please enter patient information and click 'Predict Length of Stay' to see results.")
            
            # Show example visualization
            st.markdown("### 📊 Example Risk Assessment")
            example_fig = create_probability_gauge(45)
            st.plotly_chart(example_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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
