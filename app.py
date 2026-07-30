import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## loading the model
model=tf.keras.models.load_model('model.h5')

## load encoders ans scaler files

with open('lable_encoder_gender.pkl','rb') as file:
    lable_encoder_gender= pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## STREAMLIT APP

st.title('Customer Churn Prediction')

## USING INPUTS , INPUTS THAT WILL BE GIVEN

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', lable_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare input data (create dictionary)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [lable_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## ENCODE GEOGRAPHICAL VALUE (ONE HOT)

geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()

geo_encode_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)
input_data = input_data.drop('Geography', axis=1)

## Compile one hot encoded data with input data

input_data=pd.concat([input_data.reset_index(drop=True),geo_encode_df],axis=1)

## Scale the input data

input_data_scaled=scaler.transform(input_data)

## Prediction Churn

prediction=model.predict(input_data_scaled)
prediction_probab= prediction[0][0]


st.write(f'Churn Probab: {prediction_probab:.2f}')

if prediction_probab > 0.5:
    st.write('will churn')
else:
    st.write('will not churn')