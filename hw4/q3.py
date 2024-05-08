import numpy as np
import pandas as pd
import perceptron
from perceptron import file_to_numpy
from perceptron import calc_mistakes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def NaiveBayes(xTrain, yTrain, xTest, yTest):
    """
    Train a MultinomialNB model on the dataset
    and report the performance
    Parameters
    ----------
    xTrain
    yTrain
    xTest
    yTest

    Returns
    -------
    results: model's number of mistakes on training set and testing set
    """
    results = {}
    model = MultinomialNB()
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTrain)
    results.update({'train': calc_mistakes(yHat, yTrain)})
    yHat = model.predict(xTest)
    results.update({'test': calc_mistakes(yHat, yTest)})
    return results

def LR(xTrain, yTrain, xTest, yTest):
    """
    Train a Logistic Regression model on the dataset
    and report the performance
    Parameters
    ----------
    xTrain
    yTrain
    xTest
    yTest

    Returns
    -------
    results: model's number of mistakes on training set and testing set
    """
    results = {}
    model = LogisticRegression(max_iter=500)
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTrain)
    results.update({'train': calc_mistakes(yHat, yTrain)})
    yHat = model.predict(xTest)
    results.update({'test': calc_mistakes(yHat, yTest)})
    return results

def main():
    xTrain_bi = file_to_numpy('xTrain_binary.csv')
    yTrain_bi = file_to_numpy('yTrain_binary.csv').reshape(-1)
    xTest_bi = file_to_numpy('xTest_binary.csv')
    yTest_bi = file_to_numpy('yTest_binary.csv').reshape(-1)

    xTrain_c = file_to_numpy('xTrain_count.csv')
    yTrain_c = file_to_numpy('yTrain_count.csv').reshape(-1)
    xTest_c = file_to_numpy('xTest_count.csv')
    yTest_c = file_to_numpy('yTest_count.csv').reshape(-1)

    # Naive Bayes
    results = NaiveBayes(xTrain_bi, yTrain_bi, xTest_bi, yTest_bi)
    print('Naive Bayes trained on binary dataset: ')
    print(results)

    results = NaiveBayes(xTrain_c, yTrain_c, xTest_c, yTest_c)
    print('Naive Bayes trained on count dataset: ')
    print(results)

    print('-----------------------------------------------------------------------------------')

    # Logistic Regression
    results = LR(xTrain_bi, yTrain_bi, xTest_bi, yTest_bi)
    print('Logistic Regression trained on binary dataset: ')
    print(results)

    results = LR(xTrain_c, yTrain_c, xTest_c, yTest_c)
    print('Logistic Regression trained on count dataset: ')
    print(results)


if __name__ == "__main__":
    main()
