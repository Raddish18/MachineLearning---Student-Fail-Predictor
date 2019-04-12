import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


d1 = pd.read_csv("student-mat.csv",delimiter=',',dtype=None,skiprows=[0], encoding='utf-8', usecols={13,14,26,27}, names=['studytime','failures', 'Dalc', 'Walc'])
d2 = pd.read_csv("student-por.csv",delimiter=',',dtype=None,skiprows=[0], encoding='utf-8', usecols={13,14,26,27}, names=['studytime','failures', 'Dalc', 'Walc'])

myData = pd.concat([d1, d2])

# Just finding out percentages for the dataset (e.g Percentage of students that drink heavily)
def percentages():
    count = 0
    # Counting percentage of people that failed
    for x in myData['failures']:
        if x >= 1:
            count = count + 1
    print(count / len(myData))

    count = 0
    #
    for x in myData['Dalc']:
        if x > 2:
            count = count + 1
    print(count / len(myData))

    count = 0
    for x in myData['Walc']:
        if x > 3:
            count = count + 1
    print(count / len(myData))

    count = 0
    for x in myData['studytime']:
        if x <= 1:
            count = count + 1
    print(count / len(myData))



# This is a basic naive bayes algorithm I started out with to get a base line
def naiveBayes():
    assert myData['Walc'].notnull().all()

    myData['total_alcohol'] = myData['Dalc'] + myData['Walc']
    myData['total_alcohol'] = myData['total_alcohol'].astype(int)

    threshold = sum(myData.total_alcohol) / len(myData.total_alcohol)
    myData['alcohol_level'] = [1 if i > threshold else 0 for i in myData.total_alcohol]

    totalFails = 0
    # Counting percentage of people that failed
    for x in myData['failures']:
        if x >= 1:
            totalFails = totalFails + 1
    failP = totalFails/len(myData)


    high = 0
    for x in myData['alcohol_level']:
        if x == 1:
            high = high + 1
    highP = high/len(myData)

    highGivenfail = 0
    # print(myData['alcohol_level'])
    for index, row in myData.iterrows():
        # print(row['alcohol_level'], row['failures'])
        if row['alcohol_level'] == 1:
            if row['failures'] > 0:
                highGivenfail = highGivenfail + 1

    hGfP = highGivenfail/totalFails


    # print("(",hGfP,"*",failP,")","/",highP)
    naiveB = (hGfP*failP)/highP
    print(naiveB)

#This function uses SkLearn's Bernoulli Naive Bayes
def naiveB():
    myData['total_alcohol'] = myData['Dalc'] + myData['Walc']
    myData['total_alcohol'] = myData['total_alcohol'].astype(int)

    threshold = sum(myData.total_alcohol) / len(myData.total_alcohol)
    myData['alcohol_level'] = [1 if i > threshold else 0 for i in myData.total_alcohol]

    # myData.values.reshape(-1, 1)

    y = myData['failures']
    X = myData.drop(['failures'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    nb = BernoulliNB(binarize=True)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    # print(y_pred)
    # print(y_test)
    print(accuracy_score(y_test, y_pred))

# SKLearn's K Nearest Neighbour algorithm
def knn():
    myData['total_alcohol'] = myData['Dalc'] + myData['Walc']
    myData['total_alcohol'] = myData['total_alcohol'].astype(int)

    threshold = sum(myData.total_alcohol) / len(myData.total_alcohol)
    myData['alcohol_level'] = [1 if i > threshold else 0 for i in myData.total_alcohol]

    y = myData['failures']
    X = myData.drop(['failures'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    kNN = KNeighborsClassifier(n_neighbors=5)

    kNN.fit(X_train, y_train)
    y_pred = kNN.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def main():
    print("Regular Naive Bayes:")
    naiveBayes()
    print("Scikit Bernoulli Naive Bayes:")
    naiveB()
    print("Scikit K Nearest Neighbour:")
    knn()

main()




