# model_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    # Load data
    try:
        data = pd.read_csv('LoanApprovalPrediction.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Dataset 'LoanApprovalPrediction.csv' not found in the current directory")
    
    # Drop Loan_ID
    data.drop(['Loan_ID'], axis=1, inplace=True)
    
    # Handle missing values
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    
    # Split features and target
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define models
    forest = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=1, n_jobs=-1)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=8, min_child_weight=6, gamma=0.1)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    ada = AdaBoostClassifier(estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)
    
    # Voting classifier
    eclf = VotingClassifier(estimators=[('forest', forest), ('xgb', xgb1), ('adaboost', ada)], voting='hard')
    eclf.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = eclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model with accuracy
    with open('loan_model.pkl', 'wb') as model_file:
        pickle.dump((eclf, label_encoder, scaler, accuracy), model_file)
    
    return eclf, label_encoder, scaler, accuracy

def load_model_and_encoders():
    with open('loan_model.pkl', 'rb') as model_file:
        model, label_encoder, scaler, accuracy = pickle.load(model_file)
    return model, label_encoder, scaler, accuracy

def preprocess_input(input_data, model, label_encoder, scaler):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Map categorical variables
    categorical_mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
        "Credit_History": {"1": 1, "0": 0},
        "Property_Area": {"Urban": 2, "Rural": 0, "Semiurban": 1}
    }
    
    # Apply mappings
    for col, mapping in categorical_mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping)
            if input_df[col].isna().any():  # Handle unmapped values
                input_df[col].fillna(mapping[list(mapping.keys())[0]], inplace=True)
    
    # Ensure all columns are numeric
    input_df = input_df.astype(float)
    
    # Scale features
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        raise ValueError(f"Scaling error: {str(e)}")
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction