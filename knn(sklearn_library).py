#this is the implementation of k nearest neighbor algorithm using scikit learn library (neighbors.KNeighborsClassifier())

#data is breast cancer data available at uci.edu
import numpy as np                                                                  #importing required packages and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

data = pd.read_csv('breast-cancer-wisconsin.data.txt')

data.replace('?', -99999, inplace=True)                                             #replace missing values with -99999(outlier)
data.drop(['samp_code_no'], 1, inplace=True)                                        #dropping the first column which represents the id


X_vals = np.array(data.drop(['class'], 1))                                          #converting to numpy arrays and seperating x from y(class)
y_vals = np.array(data['class'])

X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2)  #sepearating training and testing data

classifier = neighbors.KNeighborsClassifier()                                       #defining classifier as k nearest neighbor
classifier.fit(X_train, y_train)                                                    #training classifier on training data

accuracy = classifier.score(X_test, y_test)                                         #testing
print(accuracy)

eg_data = np.array([[3,4,2,2,3,1,1,1,1], [3,4,5,6,7,1,2,2,1], [1,2,3,4,5,6,7,8,9]]) #creating new data to predict  on trained classifier

print(classifier.predict(eg_data))                                                  #result on new data
