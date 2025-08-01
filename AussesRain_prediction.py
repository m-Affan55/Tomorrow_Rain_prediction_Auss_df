import pandas as pd
import streamlit as st
import joblib
from PIL import Image

# Loading saved model dictionary
aussie_Rain = joblib.load("Aussie_Rain.joblib")
model = aussie_Rain['model']
imputer = aussie_Rain['imputer']
scaler = aussie_Rain['scaler']
encoder = aussie_Rain['encoder']
input_cols = aussie_Rain['input_cols']
numerical_cols = aussie_Rain['numerical_cols']
categorical_cols = aussie_Rain['categorical_cols']

# ye CSS for better styling kaisa phir
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header {
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        display: block;
        margin: 0 auto;
        width: 200px;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Result styling */
    .result-box {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .rain {
        background: rgba(66, 165, 245, 0.2);
        border: 2px solid #42a5f5;
        color: #0d47a1;
    }
    
    .sun {
        background: rgba(255, 214, 0, 0.2);
        border: 2px solid #ffd600;
        color: #ff6d00;
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Responsive layout */
    @media (max-width: 768px) {
        .input-section {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<div class="header"><h1>🌧 Australian Rain Predictor</h1><h3>Predict Tomorrow\'s Rainfall with Machine Learning</h3></div>', unsafe_allow_html=True)

# Weather illustration
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/1146/1146869.png", width=150)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; color:  black;">
    <p>Enter current weather conditions to predict if it will rain tomorrow in Australia.</p>
</div>
""", unsafe_allow_html=True)

# Input section with improved layout
st.markdown("""<div class="input-section">
            <h2 style = "color: Black; display:flex; justify-content:center " >Input Section</h2>""", unsafe_allow_html=True)

# Split inputs into two columns for better organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Numerical Measurements")
    user_input = {}
    for cols in numerical_cols:
        user_input[cols] = st.number_input(
            f"{cols}",
            value=0.1,
            key=f"num_{cols}"
        )

with col2:
    st.subheader("📊 Categorical Data")
    for col_idx, col in enumerate(categorical_cols):
        categories = list(encoder.categories_[col_idx])
        user_input[col] = st.selectbox(
            f"{col}",
            options=categories,
            key=f"cat_{col}"
        )

st.markdown('</div>', unsafe_allow_html=True)

# Prediction button centered
pre = st.button("Predict Rainfall", key="predict_button")

if pre:
    # Process the input
    df = pd.DataFrame([user_input])
    df[numerical_cols] = imputer.transform(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    encoded_feature = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out(categorical_cols))
    final_input = pd.concat([df[numerical_cols], encoded_df], axis=1)
    
    # Make prediction
    prediction = model.predict(final_input)
    
    # Display result with enhanced UI
    if prediction[0] == 1:
        st.markdown(
            f'<div class="result-box rain">'
            f'<h2>🌧️ Prediction Result</h2>'
            f'<p style="font-size: 2rem;">High chance of rain tomorrow!</p>'
            f'<p>Remember to bring your umbrella! ☔</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box sun">'
            f'<h2>☀️ Prediction Result</h2>'
            f'<p style="font-size: 2rem;">No rain expected tomorrow!</p>'
            f'<p>Perfect day to go outside! 🌞</p>'
            f'</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.9rem;">
    <p>Rain Prediction Model | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)