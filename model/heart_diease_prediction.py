import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""### DATA COLLECTION AND PREPROCESSING"""

heart_data = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Heart Diease Prediction/data/heart_disease_data.csv')

heart_data.isnull().sum()

heart_data['target'].value_counts()

"""1 --> Defective Heart
0 --> Healthy Heart
"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

"""#### MODEL TRAINING"""

model = LogisticRegression()

model.fit(X_train, Y_train)

"""#### MODEL EVALUATION"""

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on testing data: ', test_data_accuracy)

"""### PREDICTIVE SYSTEM"""

input_data = (57,0,0,120,354,0,1,163,1,0.6,2,0,2)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The person does not have a heart disease')
else:
  print('The person has heart disease')

