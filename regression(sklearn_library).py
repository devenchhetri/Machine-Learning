#data is google stocks data available freely at quandl.com
#data is directly imported from quandl

import math, quandl, datetime                                                               #importing required packages and modules
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data = quandl.get('WIKI/GOOGL')                                                             #importing data
data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]            #selecting specific features from the list of features in the dataset

data['high_low_diff'] = (data['Adj. High'] - data['Adj. Low']) / data['Adj. Low'] * 100     #creating new features which might be more related to the output(decision feature-price)
data['day_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Close'] * 100

data = data[['Adj. Close', 'high_low_diff', 'day_change', 'Adj. Volume']]                   #keeping specific features in the data for processing/training
data.fillna(-99999, inplace=True)                                                           #replace missing values with -99999(outlier)

forecast = int(math.ceil(0.005*len(data)))                                                  #forecast stores 0.5% of the length of total data which is used later

data['label'] = data['Adj. Close'].shift(-forecast)                                         #choosing label as Adj Close field, but after 16 days, by shifting,ie.after 0.5% of the length of total data

X = np.array(data.drop(['label'], 1))                                                       #defining X value as all the fields except label field and converting to numpy array
X = preprocessing.scale(X)                                                                  #scaling X
X_new = X[-forecast:]                                                                       #seperating the samples which does not have label after the shift
X = X[:-forecast]

data.dropna(inplace=True)
y = np.array(data['label'])                                                                 #converting to numpy array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)                   #splitting training and testing data

classifier = LinearRegression(n_jobs=-1)                                                    #defining model as Linear regression with max number of threads
classifier.fit(X_train, y_train)                                                            #training model

accuracy = classifier.score(X_test, y_test)                                                 #checking accuracy of model

forecast_new = classifier.predict(X_new)                                                    #predicting lable for X value for which there were no labels
print(forecast_new, accuracy, forecast)

data['Forecast'] = np.nan                                                                   #adding a new feature/column to store predicted value (assigning nan for now)

last_date = data.iloc[-1].name                                                              #last_date will store date of the last item in dataset(data)
last_unix = last_date.timestamp()                                                           #taking timestamp
next_unix = last_unix + 86400                                                               #hard coding to increment by 1 day

for i in forecast_new:
    next_date = datetime.datetime.fromtimestamp(next_unix)                                  #converting to date from timestamp
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] +[i]                 #setting all columns to nan, except forecast which will hold predicted values

data['Adj. Close'].plot()                                                                   #plotting Adj Close values and forecasted values with matplotlib
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
