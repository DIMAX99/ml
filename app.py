import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import skew
import pickle

# Set page configuration
st.set_page_config(page_title="Land Type Prediction", page_icon="🌍", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('finalized_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'finalized_model.pkl' not found. Please ensure the model file is in the project directory.")
        return None

model = load_model()

# Title and description
st.title("🌍 Land Type Prediction from NDVI Values")
st.markdown("Enter NDVI (Normalized Difference Vegetation Index) values for each month to predict land type.")

# Create columns for better layout
col1, col2 = st.columns(2)

# NDVI input fields
st.subheader("📅 Monthly NDVI Values")
st.markdown("*Enter NDVI values (typically range from -1 to 1)*")

# Create input fields for each month (excluding December)
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November']

ndvi_values = []

# Create two columns for input fields
col1, col2 = st.columns(2)

for i, month in enumerate(months):
    if i < 6:
        with col1:
            value = st.number_input(
                f"{month} NDVI", 
                min_value=0, 
                max_value=10000, 
                value=0, 
                step=1,
                key=f"ndvi_{month.lower()}",
                format="%d",
                label_visibility="visible"
            )
    else:
        with col2:
            value = st.number_input(
                f"{month} NDVI", 
                min_value=0, 
                max_value=10000, 
                value=0, 
                step=1,
                key=f"ndvi_{month.lower()}",
                format="%d",
                label_visibility="visible"
            )
    ndvi_values.append(value)

# Add some spacing
st.markdown("---")

# Predict button
if st.button("🔮 Predict Land Type", type="primary", use_container_width=True):
    if model is None:
        st.error("Cannot make prediction without the trained model.")
        st.stop()
    
    # Create DataFrame for feature engineering
    input_df = pd.DataFrame([ndvi_values], columns=months)
    
    # Create month mean features (excluding December - month 12)
    for i, month in enumerate(months, 1):
        if i != 7:  # Skip month 7 as it's not in the feature list
            input_df[f'month_{i}_mean'] = input_df[month]
    
    # Define seasonal features (excluding December)
    season_features = {
        'Winter': ['January', 'February'],  # Removed December
        'Spring': ['March', 'April', 'May'], 
        'Summer': ['June', 'July', 'August'],
        'Fall': ['September', 'October', 'November']
    }
    
    # Calculate seasonal means
    for season, cols in season_features.items():
        input_df[f'{season}_mean'] = input_df[cols].mean(axis=1)
    
    # Calculate seasonal differences
    input_df['Winter_minus_Spring'] = input_df['Spring_mean'] - input_df['Winter_mean']
    input_df['Summer_minus_Spring'] = input_df['Summer_mean'] - input_df['Spring_mean']
    input_df['Summer_minus_Fall'] = input_df['Fall_mean'] - input_df['Summer_mean']
    input_df['Spring_minus_Winter'] = input_df['Spring_mean'] - input_df['Winter_mean']
    input_df['Spring_minus_Summer'] = input_df['Summer_mean'] - input_df['Spring_mean']
    
    # Select only the features expected by the model
    feature_columns = [
        'month_1_mean', 'month_2_mean', 'month_3_mean', 'month_4_mean', 
        'month_5_mean', 'month_6_mean', 'month_8_mean', 'month_9_mean', 
        'month_10_mean', 'month_11_mean', 'Winter_mean', 'Spring_mean', 
        'Summer_mean', 'Fall_mean', 'Winter_minus_Spring', 'Summer_minus_Spring',
        'Summer_minus_Fall', 'Spring_minus_Winter', 'Spring_minus_Summer'
    ]
    
    # Prepare input for model
    model_input = input_df[feature_columns]
    
    # Make prediction using the loaded model
    try:
        prediction = model.predict(model_input)[0]
        prediction_proba = model.predict_proba(model_input)[0]
        confidence = max(prediction_proba)
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Land Type", prediction)
        
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        
        with col3:
            st.metric("Average NDVI", f"{np.mean(ndvi_values):.3f}")
        
        # Display class probabilities
        st.subheader("🎯 Class Probabilities")
        classes = model.classes_
        prob_df = pd.DataFrame({
            'Land Type': classes,
            'Probability': prediction_proba
        }).sort_values('Probability', ascending=False)
        st.dataframe(prob_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
    
    # Display seasonal analysis
    st.subheader("🌤️ Seasonal Analysis")
    seasonal_col1, seasonal_col2 = st.columns(2)
    
    with seasonal_col1:
        seasonal_means = {
            'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
            'Mean NDVI': [
                input_df['Winter_mean'].iloc[0],
                input_df['Spring_mean'].iloc[0], 
                input_df['Summer_mean'].iloc[0],
                input_df['Fall_mean'].iloc[0]
            ]
        }
        seasonal_df = pd.DataFrame(seasonal_means)
        st.dataframe(seasonal_df, use_container_width=True)
    
    with seasonal_col2:
        seasonal_diffs = {
            'Seasonal Difference': [
                'Winter - Spring',
                'Summer - Spring', 
                'Summer - Fall',
                'Spring - Winter',
                'Spring - Summer'
            ],
            'Value': [
                input_df['Winter_minus_Spring'].iloc[0],
                input_df['Summer_minus_Spring'].iloc[0],
                input_df['Summer_minus_Fall'].iloc[0], 
                input_df['Spring_minus_Winter'].iloc[0],
                input_df['Spring_minus_Summer'].iloc[0]
            ]
        }
        diffs_df = pd.DataFrame(seasonal_diffs)
        st.dataframe(diffs_df, use_container_width=True)
    
    # Display input data summary
    st.subheader("📊 Input Data Summary")
    df = pd.DataFrame({
        'Month': months,
        'NDVI Value': ndvi_values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.line_chart(df.set_index('Month')['NDVI Value'])

# Sidebar with information
st.sidebar.header("ℹ️ About NDVI")
st.sidebar.markdown("""
**NDVI (Normalized Difference Vegetation Index)** ranges from -1 to 1:
  Add Noisy data to detect vegetation health.

**Land Types:**
- 🌊 Water
- 🏗️ Impervious  
- 🌾 Grass
- 🚜 Farm
- 🍎 Orchard
- 🌲 Forest

*Higher NDVI values typically indicate denser vegetation.*
""")

# Footer
st.markdown("---")
st.markdown("*Note: Replace the placeholder prediction function with your trained machine learning model.*")
