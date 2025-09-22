import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# Page configuration
st.set_page_config(
    page_title="Length of Stay Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.0rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.prediction-result {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.risk-assessment {
    padding: 15px;
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
.sidebar .sidebar-content {
    padding-top: 2rem;
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
        "description": "Preoperative waiting time"
    },
    "Cardiovascular_comorbidities": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Cardiovascular comorbidities"
    },
    "Lung_comorbidities": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Lung comorbidities"
    },
    "Operation_time_plus_230min": {
        "type": "categorical",
        "options": ["≤230 min", ">230 min"],
        "mapping": {"≤230 min": 0, ">230 min": 1},
        "default": "≤230 min",
        "description": "Operation duration"
    },
    "NO._Levels": {
        "type": "categorical",
        "options": ["Level 2", "Level 3", "Level 4", "Level >4"],
        "mapping": {"Level 2": 0, "Level 3": 1, "Level 4": 2, "Level >4": 3},
        "default": "Level 2",
        "description": "Number of surgical levels"
    },
    "Infectious_complications": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Infectious complications"
    },
    "Major_complications": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "mapping": {"No": 0, "Yes": 1},
        "default": "No",
        "description": "Major complications"
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

def create_probability_visualization(short_prob, long_prob):
    """Create horizontal bar chart for probabilities"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    categories = ['Short Stay', 'Long Stay']
    probabilities = [short_prob, long_prob]
    colors = ['#28a745', '#dc3545']
    
    bars = ax.barh(categories, probabilities, color=colors, alpha=0.8, height=0.6)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}', ha='left', va='center', 
                fontweight='bold', fontsize=16, color='black')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Classes', fontsize=14, fontweight='bold')
    ax.set_title('Prediction Probability for Patient', fontsize=18, fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_prediction_text(probability, outcome):
    """Create prediction text visualization"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    if outcome == "Long Stay":
        text = f"Predicted probability of Long Hospital Stay: {probability:.2f}%"
    else:
        text = f"Predicted probability of Short Hospital Stay: {probability:.2f}%"
    
    ax.text(0.5, 0.5, text, fontsize=18, ha='center', va='center',
            fontweight='bold', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_risk_gauge(probability):
    """Create risk assessment gauge"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create semicircle gauge
    theta = np.linspace(0, np.pi, 100)
    center = (0.5, 0.3)
    radius = 0.3
    
    # Background arc
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, 'lightgray', linewidth=20)
    
    # Risk zones
    zones = [
        (0, 30, '#28a745', 'Low Risk'),
        (30, 70, '#ffc107', 'Medium Risk'),
        (70, 100, '#dc3545', 'High Risk')
    ]
    
    for start, end, color, label in zones:
        start_angle = np.pi * (1 - start/100)
        end_angle = np.pi * (1 - end/100)
        theta_zone = np.linspace(start_angle, end_angle, 50)
        x_zone = center[0] + radius * np.cos(theta_zone)
        y_zone = center[1] + radius * np.sin(theta_zone)
        ax.plot(x_zone, y_zone, color, linewidth=20)
    
    # Needle
    needle_angle = np.pi * (1 - probability/100)
    needle_x = center[0] + (radius-0.05) * np.cos(needle_angle)
    needle_y = center[1] + (radius-0.05) * np.sin(needle_angle)
    ax.plot([center[0], needle_x], [center[1], needle_y], 'black', linewidth=4)
    ax.plot(center[0], center[1], 'ko', markersize=8)
    
    # Labels
    ax.text(center[0], center[1]-0.15, f'{probability:.1f}%', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(center[0], center[1]-0.25, 'Risk of Long Stay', 
            ha='center', va='center', fontsize=14)
    
    # Risk zone labels
    ax.text(0.2, 0.45, 'Low\nRisk', ha='center', va='center', fontsize=12, 
            color='green', fontweight='bold')
    ax.text(0.5, 0.65, 'Medium Risk', ha='center', va='center', fontsize=12, 
            color='orange', fontweight='bold')
    ax.text(0.8, 0.45, 'High\nRisk', ha='center', va='center', fontsize=12, 
            color='red', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Risk Assessment Gauge', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">🏥 AI-Assisted Prediction of Length of Stay After First Elective Open Anterior Cervical Fusion in Elderly Patients</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model("rf.pkl")
    
    # Sidebar for input
    st.sidebar.header("📝 Enter the following feature values:")
    st.sidebar.markdown("---")
    
    # Collect feature inputs from sidebar
    feature_values = []
    user_inputs = {}
    
    for feature in feature_order:
        spec = feature_specs[feature]
        
        if spec["type"] == "categorical":
            try:
                default_idx = spec["options"].index(spec["default"])
            except ValueError:
                default_idx = 0
            
            choice = st.sidebar.selectbox(
                f"{spec['description']} (Select a value)",
                options=spec["options"],
                index=default_idx,
                key=f"sidebar_{feature}"
            )
            
            # Convert to numerical value
            numeric_value = spec["mapping"][choice]
            feature_values.append(numeric_value)
            user_inputs[feature] = choice
        else:
            # For numerical features (if any)
            v = st.sidebar.number_input(
                f"{spec['description']} ({spec['min']} - {spec['max']})",
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                key=f"sidebar_{feature}"
            )
            feature_values.append(v)
            user_inputs[feature] = v
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### All rights reserved")
    st.sidebar.markdown("##### Contact: your-email@example.com")
    st.sidebar.markdown("##### Length of Stay Prediction Model")
    
    # Main content area
    # Predict button
    if st.button("🔮 Predict Length of Stay", type="primary", use_container_width=True):
        try:
            # Prepare input for model
            features = np.array([feature_values])
            feature_df = pd.DataFrame(features, columns=feature_order)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                if hasattr(model, "predict_proba"):
                    # Get probabilities
                    predicted_proba = model.predict_proba(features)[0]
                    predicted_class = model.predict(features)[0]
                    
                    # Get probabilities for both classes
                    classes = getattr(model, "classes_", [0, 1])
                    
                    if 0 in classes and 1 in classes:
                        short_stay_prob = float(predicted_proba[list(classes).index(0)])
                        long_stay_prob = float(predicted_proba[list(classes).index(1)])
                    else:
                        short_stay_prob = predicted_proba[0]
                        long_stay_prob = predicted_proba[1] if len(predicted_proba) > 1 else 1 - predicted_proba[0]
                    
                    # Determine main outcome
                    if predicted_class == 0:
                        outcome = "Short Stay"
                        main_probability = short_stay_prob * 100
                    else:
                        outcome = "Long Stay"
                        main_probability = long_stay_prob * 100
                    
                    # Display prediction text
                    text_fig = create_prediction_text(main_probability, outcome)
                    st.pyplot(text_fig)
                    plt.close()
                    
                    # Display main result box
                    if outcome == "Long Stay":
                        if main_probability < 30:
                            risk_class = "low-risk"
                            risk_text = "Low Risk of Extended Stay"
                        elif main_probability < 70:
                            risk_class = "medium-risk"
                            risk_text = "Medium Risk of Extended Stay"
                        else:
                            risk_class = "high-risk"
                            risk_text = "High Risk of Extended Stay"
                    else:
                        risk_class = "low-risk"
                        risk_text = "Low Risk of Extended Stay"
                    
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>Predicted Outcome: {outcome}</h2>
                        <h3>Confidence: {main_probability:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="risk-assessment {risk_class}">
                        {risk_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create two columns for visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Probability Comparison")
                        prob_fig = create_probability_visualization(short_stay_prob, long_stay_prob)
                        st.pyplot(prob_fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("### 🎯 Risk Assessment")
                        gauge_fig = create_risk_gauge(long_stay_prob * 100)
                        st.pyplot(gauge_fig)
                        plt.close()
                    
                    # Clinical interpretation
                    st.markdown("### 💡 Clinical Recommendation")
                    if long_stay_prob * 100 < 30:
                        interpretation = """
                        **Low Risk Patient**: 
                        - Standard discharge planning is appropriate
                        - Normal post-operative monitoring protocols
                        - Expected typical recovery timeline
                        """
                    elif long_stay_prob * 100 < 70:
                        interpretation = """
                        **Moderate Risk Patient**: 
                        - Enhanced monitoring and assessment recommended
                        - Consider proactive discharge planning
                        - Monitor for potential complications
                        """
                    else:
                        interpretation = """
                        **High Risk Patient**: 
                        - Intensive monitoring and multidisciplinary planning required
                        - Proactive interventions to prevent complications
                        - Extended care planning may be necessary
                        """
                    
                    st.info(interpretation)
                
                else:
                    # Classification only (no probabilities)
                    predicted_class = model.predict(features)[0]
                    outcome = "Long Stay" if predicted_class == 1 else "Short Stay"
                    
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>Predicted Outcome: {outcome}</h2>
                        <p>Note: Probability information not available with current model</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.error("Please check your inputs and try again.")
    
    else:
        # Default display when no prediction is made
        st.info("👈 Please enter patient information in the sidebar and click 'Predict Length of Stay' to see results.")
        
        # Show sample visualization
        st.markdown("### 📊 Sample Risk Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            sample_prob_fig = create_probability_visualization(0.65, 0.35)
            st.pyplot(sample_prob_fig)
            plt.close()
        
        with col2:
            sample_gauge_fig = create_risk_gauge(35)
            st.pyplot(sample_gauge_fig)
            plt.close()
        
        st.markdown("""
        ### About This Tool
        This AI-assisted prediction tool uses a Random Forest machine learning model to predict 
        the likelihood of extended hospital stay after anterior cervical fusion surgery in elderly patients.
        
        **Key Features:**
        - Interpretable predictions with confidence levels
        - Risk assessment visualization
        - Clinical recommendations based on risk levels
        - User-friendly interface for clinical decision support
        
        **Disclaimer:** This tool is for clinical decision support only and should not replace 
        professional medical judgment. Model performance should be validated in your specific clinical setting.
        """)

if __name__ == "__main__":
    main()
