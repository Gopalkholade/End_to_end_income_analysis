import streamlit as st
import json
import pickle

with open("file.json","r") as f:
    dictin=json.load(f)

with open("models/model.pkl","rb") as f:
    model=pickle.loads(f.read())

st.header("**:red[Income Prediction]** :money_with_wings:")
st.subheader("_The webapp to predict income with Adult income DataSet._")

st.write("**Enter the data:**")

col1,col2,col3=st.columns(3)
with col1:
    age=st.number_input('Age',min_value=10,max_value=90,value=30,step=5)
    race=st.selectbox("Race",dictin['race'])
    sex=st.selectbox("Sex",dictin['sex'])
    marital_status=st.selectbox("Marital Status",dictin['marital_status'])
    education=st.selectbox("Education",dictin['education'])
    
with col2:
    education_years=st.number_input('Education Years',min_value=1,max_value=30,value=4,step=1)
    relationship=st.selectbox("Relationship",dictin['relationship'])
    native_country=st.selectbox("Native Country",dictin['native_country'])
    workclass=st.selectbox("Workclass",dictin['workclass'])
    
with col3:
    occupation=st.selectbox("Occupation",dictin['occupation'])
    capital_gain=st.number_input('Capital Gain',min_value=0,max_value=99999,value=30,step=5)
    capital_loss=st.number_input('Capital Loss',min_value=0,max_value=99999,value=30,step=5)
    hours_per_week=st.number_input('Work Hours(Per week)',min_value=1,max_value=99,value=30,step=5)

myarr=[age,workclass,education,education_years,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country]
def prediction(model,arr=list):
    import pandas as pd
    import numpy as np
    arr=np.array(arr).reshape(1,-1)
    df=pd.DataFrame(data=arr,columns=['age', 'workclass', 'education', 'education_years', 'marital_status','occupation', 'relationship', 'race', 'sex', 'capital_gain','capital_loss', 'hours_per_week', 'native_country'])
    prediction=model.predict(df)
    if prediction[0] == 1:
        return [">50K"]
    elif prediction[0] ==0:
        return ["<50K"]
    else:
        return ["Error"]
    
if __name__=="__main__":
    if st.button("Click To Predict"):
        pred=prediction(model,arr=myarr)
        st.write(f"Success! Your prediction is '{pred[0]}'")
    else:
        pass