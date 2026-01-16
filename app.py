import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page Config
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# Load Resources
@st.cache_resource
def load_resources():
    # Ensure these files are in the same directory
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

try:
    model, scaler, encoders = load_resources()
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

# Title
st.title("HR Analytics: Employee Attrition Prediction")
st.markdown("Predict whether an employee is at risk of leaving the organization.")

# Sidebar for Inputs
st.sidebar.header("Employee Details")

def user_input_features():
    # Collect user inputs
    age = st.sidebar.slider("Age", 18, 60, 30)
    daily_rate = st.sidebar.number_input("Daily Rate", 100, 2000, 800)
    distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)
    
    # Dropdowns for categorical data
    # We use the classes_ from encoders to ensure matching values
    dept = st.sidebar.selectbox("Department", encoders['Department'].classes_)
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5])
    gender = st.sidebar.selectbox("Gender", encoders['Gender'].classes_)
    
    job_role = st.sidebar.selectbox("Job Role", encoders['JobRole'].classes_)
    satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    marital = st.sidebar.selectbox("Marital Status", encoders['MaritalStatus'].classes_)
    
    income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    companies = st.sidebar.slider("Num Companies Worked", 0, 10, 1)
    overtime = st.sidebar.selectbox("OverTime", encoders['OverTime'].classes_)
    
    stock = st.sidebar.slider("Stock Option Level", 0, 3, 0)
    total_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
    years_at_company = st.sidebar.slider("Years At Company", 0, 40, 5)
    years_in_role = st.sidebar.slider("Years In Current Role", 0, 20, 3)
    
    # Store in a dictionary
    data = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance,
        'Department': dept,
        'Education': education,
        'Gender': gender,
        'JobRole': job_role,
        'JobSatisfaction': satisfaction,
        'MaritalStatus': marital,
        'MonthlyIncome': income,
        'NumCompaniesWorked': companies,
        'OverTime': overtime,
        'StockOptionLevel': stock,
        'TotalWorkingYears': total_years,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        
        # Default values for features not collected in UI
        'BusinessTravel': 'Travel_Rarely',
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 3,
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'MonthlyRate': 14000,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': 3,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 2
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Display User Input
st.subheader("Employee Data")
st.write(input_df)

# Preprocess Input for Prediction
if st.button("Predict Attrition Risk"):
    # 1. Encode Categoricals
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # 2. Reorder Columns (CRITICAL FIX)
    # The model expects columns in this exact order:
    expected_order = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 
        'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 
        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
        'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager'
    ]
    
    # Apply reordering
    input_df = input_df[expected_order]

    # 3. Scale
    input_df_scaled = scaler.transform(input_df)

    # 4. Predict
    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)[0][1]
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"High Risk of Attrition! (Probability: {probability:.2f})")
        st.write("Recommendation: Review compensation, workload, and career growth opportunities.")
    else:
        st.success(f"Low Risk of Attrition. (Probability: {probability:.2f})")