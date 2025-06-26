import pickle

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
# print(diabetes_dataset.head())
diabetes_dataset.describe()
# print(diabetes_dataset.describe())
diabetes_dataset['Outcome'].value_counts()
# print(diabetes_dataset['Outcome'].value_counts())
diabetes_dataset.groupby('Outcome').mean()
X = diabetes_dataset.drop(columns='Outcome',axis =1)
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)
# print(X.shape,X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
# print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
# print('Accuracy score of the test data : ', test_data_accuracy)
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_num_array = np.asarray(input_data)
input_data_reshaped = input_data_as_num_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

filename = 'diabetes_model_save.sav'
pickle.dump(classifier,open(filename,'wb'))
loaded_model = pickle.load(open('diabetes_model_save.sav','rb'))
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_num_array = np.asarray(input_data)
input_data_reshaped = input_data_as_num_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('the person is diabetic')


