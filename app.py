import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("Employee Attrition Analysis and Prediction")
st.write("Welcome to the Employee Attrition Prediction App. This app helps HR to analyze employee data and predict if an employee is likely to leave the company.")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Employee-Attrition - Employee-Attrition (1).csv')
    return data

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Data Overview', 'Exploratory Data Analysis', 'Predict Attrition'])

if options == 'Data Overview':
    st.header("Data Overview")
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())
    
    st.write("Summary Statistics:")
    st.dataframe(df.describe())

elif options == 'Exploratory Data Analysis':
    st.header("Exploratory Data Analysis")
    st.write("Let's look at some charts to understand the data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition Count")
        fig, ax = plt.subplots()
        sns.countplot(x='Attrition', data=df, ax=ax, palette="Set2")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Age Distribution by Attrition")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='Age', hue='Attrition', kde=True, ax=ax, palette="Set1")
        st.pyplot(fig)

    st.subheader("Job Satisfaction vs Attrition")
    fig, ax = plt.subplots()
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, ax=ax, palette="pastel")
    st.pyplot(fig)

elif options == 'Predict Attrition':
    st.header("Predict Employee Attrition")
    st.write("Enter the employee details below to predict if they will leave.")
    
    # Load model and encoders
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    except Exception as e:
        st.error("Model not found. Please make sure train.py has been executed.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    # We will just ask for a few important features to keep it simple for the user
    # And fill the rest with mean/mode values
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
        distance = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
        
    with col2:
        env_sat = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
        job_sat = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
        work_life = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
        
    with col3:
        years_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        years_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
        monthly_inc = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)

    # Add other required fields with default values
    input_data = {}
    for col in df.columns:
        if col not in ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']:
            input_data[col] = df[col].mode()[0] if df[col].dtype == 'O' else df[col].median()
            
    # Update with user inputs
    input_data['Age'] = age
    input_data['DailyRate'] = daily_rate
    input_data['DistanceFromHome'] = distance
    input_data['EnvironmentSatisfaction'] = env_sat
    input_data['JobSatisfaction'] = job_sat
    input_data['WorkLifeBalance'] = work_life
    input_data['YearsAtCompany'] = years_company
    input_data['YearsInCurrentRole'] = years_role
    input_data['MonthlyIncome'] = monthly_inc

    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col, le in encoders.items():
        if col in input_df.columns:
            # Handle unseen labels by setting them to the first class if not found
            if input_df[col].iloc[0] in le.classes_:
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = 0

    if st.button("Predict"):
        prediction = model.predict(input_df)
        pred_label = encoders['Attrition'].inverse_transform(prediction)[0]
        
        if pred_label == 'Yes':
            st.error("Prediction: The employee is likely to LEAVE (Attrition = Yes).")
        else:
            st.success("Prediction: The employee is likely to STAY (Attrition = No).")
