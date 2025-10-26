import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Wholesale Customers Prediction", page_icon="📦", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    with open('model2.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.title("Wholesale Customers Channel Prediction")
    st.write("Enter customer data to predict the sales channel (Hotel/Restaurant/Cafe vs Retail)")
    
    # Add information boxes
    with st.expander("ℹ️ About the Prediction"):
        st.write("""
        **Target Variable - Channel:**
        - **0**: Hotel/Restaurant/Cafe
        - **1**: Retail
        
        **Region Codes:**
        - **1**: Lisbon
        - **2**: Oporto  
        - **3**: Other regions
        """)
    
    # Load model
    model = load_model()
    
    # Create input fields for all features in the wholesale customers dataset
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Categories")
        fresh = st.number_input("Fresh Products", min_value=0, value=12000, 
                               help="Annual spending on fresh products (fruits, vegetables, meat, fish)")
        milk = st.number_input("Milk Products", min_value=0, value=5000,
                              help="Annual spending on milk products (dairy)")
        grocery = st.number_input("Grocery Products", min_value=0, value=7500,
                                 help="Annual spending on grocery products (packaged goods)")
    
    with col2:
        st.subheader("Additional Categories & Location")
        frozen = st.number_input("Frozen Products", min_value=0, value=3000,
                                help="Annual spending on frozen products")
        detergents_paper = st.number_input("Detergents/Paper", min_value=0, value=2000,
                                          help="Annual spending on detergents and paper products")
        delicassen = st.number_input("Delicatessen Products", min_value=0, value=1500,
                                    help="Annual spending on delicatessen products (specialty foods)")
        
        # Region input with better labels
        region_options = {1: "1 - Lisbon", 2: "2 - Oporto", 3: "3 - Other Regions"}
        region = st.selectbox("Region", options=[1, 2, 3], 
                             format_func=lambda x: region_options[x],
                             help="Geographic region of the customer")
    
    # Create prediction button
    if st.button("Make Prediction", type="primary"):
        # Prepare input data in the exact order used during training
        input_data = pd.DataFrame({
            'Region': [region],
            'Fresh': [fresh],
            'Milk': [milk],
            'Grocery': [grocery],
            'Frozen': [frozen],
            'Detergents_Paper': [detergents_paper],
            'Delicassen': [delicassen]
        })
        
        # Ensure column order matches training data
        input_data = input_data[['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
        
        # Debug: Show input data structure
        st.write("Debug - Input Data:")
        st.write(input_data)
        
        try:
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = None
            
            # Try to get prediction probabilities if available
            try:
                prediction_proba = model.predict_proba(input_data)
            except:
                pass
            
            # Debug: Show raw prediction values
            st.write("Debug - Raw Prediction:", prediction)
            if prediction_proba is not None:
                st.write("Debug - Raw Probabilities:", prediction_proba)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            # Enhanced result display
            if prediction[0] == 0:
                st.success("🏨 Predicted Channel: **Hotel/Restaurant/Cafe** (Channel 0)")
                st.write("This customer is predicted to be from the **Horeca** (Hotel/Restaurant/Cafe) sector.")
            else:
                st.success("🛒 Predicted Channel: **Retail** (Channel 1)")
                st.write("This customer is predicted to be from the **Retail** sector.")
            
            if prediction_proba is not None:
                st.write("**Confidence Levels:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hotel/Restaurant/Cafe", f"{prediction_proba[0][0]:.1%}")
                with col2:
                    st.metric("Retail", f"{prediction_proba[0][1]:.1%}")

            # Display input summary
            st.subheader("Input Summary")
            st.dataframe(input_data.transpose(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check if the model file exists and features match the training data.")

if __name__ == "__main__":
    main()
