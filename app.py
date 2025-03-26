# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from model_utils import train_model, preprocess_input, load_model_and_encoders

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Load or train model
    if 'model' not in st.session_state:
        model_file = 'loan_model.pkl'
        if os.path.exists(model_file):
            try:
                model, label_encoder, scaler, accuracy = load_model_and_encoders()
                st.session_state['model'] = model
                st.session_state['label_encoder'] = label_encoder
                st.session_state['scaler'] = scaler
                st.session_state['accuracy'] = accuracy
                st.info("Model loaded successfully from 'loan_model.pkl'")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return
        else:
            st.warning("Model file 'loan_model.pkl' not found. Training new model...")
            with st.spinner("Training model... This may take a few moments."):
                try:
                    model, label_encoder, scaler, accuracy = train_model()
                    st.session_state['model'] = model
                    st.session_state['label_encoder'] = label_encoder
                    st.session_state['scaler'] = scaler
                    st.session_state['accuracy'] = accuracy
                    st.success("Model trained and saved successfully!")
                except Exception as e:
                    st.error(f"Failed to train model: {str(e)}")
                    return

    # Sidebar
    with st.sidebar:
        st.header("📋 Project Overview")
        st.info("""
            This application predicts loan approval using an ensemble Voting Classifier 
            combining Random Forest, XGBoost, and AdaBoost models.
        """)
        st.header("Features Used")
        st.write("- Gender\n- Married\n- Dependents\n- Education\n- Self_Employed\n- ApplicantIncome\n- CoapplicantIncome\n- LoanAmount\n- Loan_Amount_Term\n- Credit_History\n- Property_Area")
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{st.session_state['accuracy']*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("Enter the applicant's details to predict loan approval status")

    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0.0, step=100.0, value=5000.0)

    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=100.0, value=2000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0, value=150.0)
        loan_term = st.number_input("Loan Term (days)", min_value=0.0, step=30.0, value=360.0)
        credit_history = st.selectbox("Credit History", ["1", "0"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Prediction
    if st.button("Predict Loan Status"):
        with st.spinner("Processing..."):
            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }
            
            try:
                prediction = preprocess_input(input_data, st.session_state['model'], 
                                            st.session_state['label_encoder'], 
                                            st.session_state['scaler'])
                result = "Approved" if prediction[0] == 1 else "Rejected"
                st.success(f"Loan Status Prediction: **{result}**")
                
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Model Accuracy", f"{st.session_state['accuracy']*100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

    # Model details expander
    with st.expander("🔍 Model Details & Processing Steps"):
        st.subheader("Model Architecture")
        st.write("""
            - **Ensemble Method**: Voting Classifier (Hard Voting)
            - **Models Used**:
                1. Random Forest (400 estimators, entropy criterion)
                2. XGBoost (1000 estimators, max_depth=8)
                3. AdaBoost (500 estimators with Decision Tree base)
            - **Preprocessing**: 
                - Missing values filled with mode/mean
                - Categorical encoding with LabelEncoder
                - Feature scaling with StandardScaler
        """)

if __name__ == "__main__":
    main()