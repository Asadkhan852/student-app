import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open('student_lr_final_model.pkl','rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
     data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
     df = pd.DataFrame([data])
     df_transformed = scaler.transform(df)
     return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hr = st.number_input("Hours studied",min_value=1,max_value=10,value=5)
    ps = st.number_input("Previous Score",min_value=40,max_value=100,value=70)   
    Ex = st.selectbox("Extracurricular Activities",['Yes','No'])
    Sh = st.number_input("Sleeping Hours",min_value=4,max_value=10,value=7)  
    Qp = st.number_input("number of question paper solved",min_value=0,max_value=10,value=5)  

    if st.button('predict-your_score'):
        user_data = {
            'Hours Studied' : hr,
            'Previous Scores':ps,
            'Extracurricular Activities':Ex	,
            'Sleep Hours':Sh,
            'Sample Question Papers Practiced':Qp
        }
        prediction = predict_data(user_data)
        st.success(f"your prediction result is {prediction}")

if __name__ == "__main__":
    main()