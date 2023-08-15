import pandas as pd
import joblib
import streamlit as st

featuress = pd.read_csv("Data/New_Data.csv")
target = featuress.drop(columns=['Salary'])

st.set_page_config(page_title="Salary Prediction", )
st.title('Job Salary Prediction', '\n')
st.image('Image/pexels-tima-miroshnichenko-6694543.jpg', width=700)


st.markdown(""" 
##### The goal of this application is to estimate your salary based on your qualifications.
##### To predict your salary, just follow these steps:
##### 1. Enter your qualifications.
##### 2. Press the "Predict" button and wait your expected salary.
""")


def user_input_features():

    st.sidebar.write('# Fill this form please..')

    Gender = st.sidebar.radio("Gender",
                                options=(Gender for Gender in featuress.Gender.unique()))


    Education_Level	 = st.sidebar.selectbox("Education Level",
                                    options=(Education_Level for Education_Level in featuress.Education_Level.unique()))


    Job_Title = st.sidebar.selectbox("Job Title",
                                    options=(Job_Title for Job_Title in featuress.Job_Title.unique()))


    Age = st.sidebar.number_input('Enter your age', 0, 60)

    Years_of_Experience	 = st.sidebar.slider('How many years of experience do you have ?', 0, 35)
    
    data = {
        "Age": Age,
        "Gender": Gender,
        "Education_Level": Education_Level,
        "Job_Title": Job_Title,
        "Years_of_Experience": Years_of_Experience
        }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

df = pd.concat([input_df,target],axis=0)

df['Education_Level'] = df['Education_Level'].map({"High School":1, "Bachelor's Degree":2, "Master's Degree":3, "PhD": 4})
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df = pd.get_dummies(df, ['Job_Title'])


RF_MODEL_PATH = joblib.load("Models/model.h5")
RF_SCALER_PATH = joblib.load("Models/scaler.h5")

scaled_data = RF_SCALER_PATH.transform(df)
prediction = RF_MODEL_PATH.predict(scaled_data)

if st.sidebar.button('Predict'):
    st.sidebar.success(f'# The Salary of your Qaulifications is : {round(prediction[0])}')

