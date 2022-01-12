# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:07:37 2022

@author: Pushp jain
"""

import numpy as np
import pickle
import streamlit as st

#Loading the model
loaded_model = pickle.load(open('D:/DESKTOP/Data Science Projects/Breast_Cancer_Classification_System/trained_model_breast_cancer.sav', 'rb'))

#creating function for prediction
def breast_cancer_classification(input_data):
    input_data_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_np_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
      return 'Breast Cancer detected (Malignant Tumour)'
    else:
      return 'Breast Cancer not detected (Benign Tumour)'
  

def main():
    #Giving title
    st.title('Breast Cancer Classification System')
    
    #Getting input from user
    input_fields = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
    input_data_1 = []
    for i in range(len(input_fields)):
        sample_input = st.text_input('Enter value of '+ input_fields[i] + ':')
        input_data_1.append(sample_input)
    
    # Pregnancies = st.text_input('Number of Pregnancies')
    # Glucose = st.text_input('Blood Glucose Level')
    # BloodPressure = st.text_input('Blood Pressure Value')
    # SkinThickness = st.text_input('Skin Thickness Value')
    # Insulin = st.text_input('Blood Insulin Level')
    # BMI = st.text_input('Body Mass Index (BMI) Value')
    # DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    # Age = st.text_input('Enter Age')
    
    #For prediction
    diagnosis = ''
    if st.button('Breast Cancer Classification Result'):
        diagnosis = breast_cancer_classification(input_data_1)

    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    