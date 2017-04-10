#this is the implementation of k nearest neighbor algorithm from scratch

#data is breast cancer data available at uci.edu
import numpy as np                                                      #importing required packages and modules
import pandas as pd
import warnings, random
from collections import Counter

def knn(dataset, predict, k=3):                                         #here, we define the main k nearest neighbor algorithm
    dist = []
    for group in dataset:
        for feat in dataset[group]:
            euc_dist = np.linalg.norm(np.array(feat)-np.array(predict)) #calculating euclidean distance using numpy
            dist.append([euc_dist, group])

    vote = [i[1] for i in sorted(dist)[:k]]                             #vote array will hold the k nearest groups from sorted dist array
    result = Counter(vote).most_common(1)[0][0]                         #result will hold the class which is maximum in vote array
    return result


data = pd.read_csv('breast-cancer-wisconsin.data.txt')
data.replace('?', -99999, inplace=True)                                 #replace missing values with -99999(outlier)
data.drop(['samp_code_no'], 1, inplace=True)                            #dropping the first column which represents the id
dataset = data.astype(float).values.tolist()                            #converting all value to float and then to list
random.shuffle(dataset)                                                 #shuffling the datasset

test_size = 0.25
training_set = {2:[], 4:[]}
testing_set = {2:[], 4:[]}
training_data = dataset[:-int((len(dataset))*test_size)]                #segregating training and testing data
testing_data = dataset[-int((len(dataset))*test_size):]

for i in training_data:
    training_set[i[-1]].append(i[:-1])                                  #training and testing set will be dictionary containing arrays

for i in testing_data:
    testing_set[i[-1]].append(i[:-1])

match=0
total=0

for group in testing_set:
    for each in testing_set[group]:
        vote = knn(training_set, each, k=3)                             #result(class) for each sample will be returned to variable(vote)
        if group == vote:
            match +=1                                                   #counting number of matches to calculate accuracy
        total +=1

print('accuracy =', match/total)                                        #accuracy = number of matches/total number of samples



