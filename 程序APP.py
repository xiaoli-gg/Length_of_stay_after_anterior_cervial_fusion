import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# Custom CSS styling
st.markdown("""
<style>
/* Remove default Streamlit padding and margins */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

/* Hide Streamlit header and footer */
header[data-testid="stHeader"] {
    display: none;
}

.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.5rem;
    margin-top: 0;
}
.section-header {
    font-size: 1.4rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
    margin-top: 0;
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
    margin-top: 0;
}
.output-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 0;
}
.feature-group {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.metric-container {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 10px 0;
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
    """Create a gauge chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create semicircle gauge
    theta1, theta2 = 0, np.pi
    center = (0.5, 0.5)
    radius = 0.4
    
    # Background arc
    theta = np.linspace(theta1, theta2, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, 'lightgray', linewidth=20)
    
    # Color segments
    segments = [
        (0, 30, '#28a745'),    # Green (Low risk)
        (30, 70, '#ffc107'),   # Yellow (Medium risk) 
        (70, 100, '#dc3545')   # Red (High risk)
    ]
    
    for start, end, color in segments:
        start_angle = np.pi * (1 - start/100)
        end_angle = np.pi * (1 - end/100)
        theta_seg = np.linspace(start_angle, end_angle, 50)
        x_seg = center[0] + radius * np.cos(theta_seg)
        y_seg = center[1] + radius * np.sin(theta_seg)
        ax.plot(x_seg, y_seg, color, linewidth=20)
    
    # Needle
    needle_angle = np.pi * (1 - probability/100)
    needle_x = center[0] + (radius-0.05) * np.cos(needle_angle)
    needle_y = center[1] + (radius-0.05) * np.sin(needle_angle)
    ax.plot([center[0], needle_x], [center[1], needle_y], 'black', linewidth=4)
    ax.plot(center[0], center[1], 'ko', markersize=8)
    
    # Labels
    ax.text(center[0], center[1]-0.15, f'{probability:.1f}%', 
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(center[0], center[1]-0.25, 'Long Stay Probability', 
            ha='center', va='center', fontsize=14)
    
    # Risk zone labels
    ax.text(0.15, 0.65, 'Low\nRisk', ha='center', va='center', fontsize=10, color='green')
    ax.text(0.5, 0.85, 'Medium Risk', ha='center', va='center', fontsize=10, color='orange')
    ax.text(0.85, 0.65, 'High\nRisk', ha='center', va='center', fontsize=10, color='red')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Risk Assessment Gauge', fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig

def create_probability_bars(short_prob, long_prob):
    """Create probability bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Short Stay', 'Long Stay']
    probabilities = [short_prob, long_prob]
    colors = ['#28a745' if short_prob > long_prob else '#90EE90', 
              '#dc3545' if long_prob > short_prob else '#FFB6C1']
    
    bars = ax.barh(categories, probabilities, color=colors, alpha=0.8, height=0.6)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', ha='left', va='center', fontweight='bold', fontsize=14)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Prediction Probabilities', fontsize=16, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add reference line at 50%
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.text(51, 1.5, '50%', rotation=90, va='bottom', ha='left', color='gray')
    
    plt.tight_layout()
    return fig

def create_risk_visualization(probability):
    """Create a simple risk level visualization"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Risk zones
    zones = [
        (0, 30, '#28a745', 'Low Risk'),
        (30, 70, '#ffc107', 'Medium Risk'),
        (70, 100, '#dc3545', 'High Risk')
    ]
    
    y_pos = 0.5
    height = 0.3
    
    for start, end, color, label in zones:
        rect = patches.Rectangle((start, y_pos), end-start, height, 
                               facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(start + (end-start)/2, y_pos + height/2, label, 
                ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Current probability marker
    marker_y = y_pos + height + 0.1
    ax.plot(probability, marker_y, 'v', color='black', markersize=15)
    ax.text(probability, marker_y + 0.15, f'{probability:.1f}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Probability of Long Stay (%)', fontsize=12)
    ax.set_title('Risk Level Assessment', fontsize=16, pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
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
    st.markdown('<p style="text-align: center; color: #666; margin: 0 0 1rem 0; line-height: 1.4;">Interpretable Random Forest Model for Predicting Length of Stay After First Elective Open Anterior Cervical Fusion in Elderly Patients</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model("rf.pkl")
    
    # Create main layout: Left (Input) | Right (Output)
    left_col, right_col = st.columns([1, 1], gap="medium")
    
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
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>Short Stay Probability</h4>
                                <h2 style="color: #28a745;">{short_stay_prob:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>Long Stay Probability</h4>
                                <h2 style="color: #dc3545;">{long_stay_prob:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualization 1: Risk Gauge
                        st.markdown("### 🎯 Risk Assessment Gauge")
                        gauge_fig = create_probability_gauge(long_stay_prob)
                        st.pyplot(gauge_fig)
                        plt.close()
                        
                        # Visualization 2: Probability Bars
                        st.markdown("### 📈 Probability Comparison")
                        bar_fig = create_probability_bars(short_stay_prob, long_stay_prob)
                        st.pyplot(bar_fig)
                        plt.close()
                        
                        # Visualization 3: Risk Level
                        st.markdown("### 🚦 Risk Level Indicator")
                        risk_fig = create_risk_visualization(long_stay_prob)
                        st.pyplot(risk_fig)
                        plt.close()
                        
                        # Clinical interpretation
                        st.markdown("### 💡 Clinical Interpretation")
                        if long_stay_prob < 30:
                            interpretation = "**Low Risk**: Patient is likely to have a normal length of stay. Standard discharge planning is appropriate."
                        elif long_stay_prob < 70:
                            interpretation = "**Moderate Risk**: Patient has moderate risk for extended stay. Consider enhanced monitoring and discharge planning."
                        else:
                            interpretation = "**High Risk**: Patient is at high risk for extended hospitalization. Recommend proactive interventions and multidisciplinary planning."
                        
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
            st.pyplot(example_fig)
            plt.close()
        
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
