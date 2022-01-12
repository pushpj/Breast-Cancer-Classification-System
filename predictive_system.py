# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:54:10 2022

@author: Pushp jain
"""

import numpy as np
import pickle

#Loading the model
loaded_model = pickle.load(open('D:/DESKTOP/Data Science Projects/Breast_Cancer_Classification_System/trained_model_breast_cancer.sav', 'rb'))

input_data = (18.22,18.7,120.3,1033,0.1148,0.1485,0.1772,0.106,0.2092,0.0631,0.8337,1.593,4.877,98.81,0.003899,0.02961,0.02817,0.009222,0.02674,0.005126,20.6,24.13,135.1,1321,0.128,0.2297,0.2623,0.1325,0.3021,0.07987)
input_data_np_array = np.asarray(input_data)
input_data_reshaped = input_data_np_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
if prediction[0] == 0:
  print('Breast Cancer detected (Malignant Tumour)')
else:
  print('Breast Cancer not detected (Benign Tumour)')
