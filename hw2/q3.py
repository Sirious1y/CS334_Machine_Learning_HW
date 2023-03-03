import q2
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def kfold_knn(xFeat, y):
    """
    Use 5-fold cross validation to find the best k for K-NN

    Parameters
    ----------
    xFeat : nd-array with shape n x d
        Features of the dataset
    y : 1-array with shape n x 1
        Labels of the dataset

    Returns
    -------
    best_k : int
        the k that gives the highest AUC
    """
    kf = KFold(n_splits=5)
    knn_result = []
    y = y['label']
    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(estimator=knn, X=xFeat, y=y, scoring='roc_auc', cv=kf)
        knn_result.append([k, np.mean(scores)])

    knn_result = pd.DataFrame(knn_result, columns=['K', 'AUC'])
    print('K-NN: ')
    print(knn_result)
    print('-----------------------------------------------------------------------')
    best_k = knn_result.iloc[knn_result['AUC'].idxmax(), 0]

    return best_k


def kfold_dt(xFeat, y):
    """
    Use 5-fold cross validation to find the best Max_Depth
    and best Min_Sample_Leaf for Decision Tree

    Parameters
    ----------
    xFeat : nd-array with shape n x d
        Features of the dataset
    y : 1-array with shape n x 1
        Labels of the dataset

    Returns
    -------
    best_md : int
        the Max_Depth that gives the highest AUC
    best_msl : int
        the Min_Sample_Leaf that gives the highest AUC
    """
    kf = KFold(n_splits=5)
    dt_md_result = []
    dt_msl_result = []
    y = y['label']

    # find optimal Max_Depth
    for md in range(1, 16):
        dt = DecisionTreeClassifier(max_depth=md)
        scores = cross_val_score(estimator=dt, X=xFeat, y=y, scoring='roc_auc', cv=kf)
        dt_md_result.append([md, np.mean(scores)])

    # find optimal Min_Sample_Leaf
    for msl in range(0, 401, 25):
        if msl == 0:
            msl += 1
        dt = DecisionTreeClassifier(min_samples_leaf=msl)
        scores = cross_val_score(estimator=dt, X=xFeat, y=y, scoring='roc_auc', cv=kf)
        dt_msl_result.append([msl, np.mean(scores)])

    dt_md_result = pd.DataFrame(dt_md_result, columns=['Max_Depth', 'AUC'])
    dt_msl_result = pd.DataFrame(dt_msl_result, columns=['Min_Sample_Leaf', 'AUC'])
    print('Decision Tree Max_Depth: ')
    print(dt_md_result)
    print('-----------------------------------------------------------------------')
    print('Decision Tree Min_Sample_Leaf: ')
    print(dt_msl_result)
    print('-----------------------------------------------------------------------')
    best_md = dt_md_result.iloc[dt_md_result['AUC'].idxmax(), 0]
    best_msl = dt_msl_result.iloc[dt_msl_result['AUC'].idxmax(), 0]

    return best_md, best_msl


def opt_knn(xTrain, yTrain, xTest, yTest, best_k):
    """
    Train a K-NN with the best k found previously on the whole dataset,
    95% of the dataset, 90% of the dataset, and 80% of the dataset, separately.
    And report the accuracies and AUCs

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.
    best_k : int
        the best k for K-NN

    Returns
    -------
    result : DataFrame with columns=['model', 'Acc', 'AUC']
        the Accs and AUCs for each K-NN model
    """
    result = []
    yTrain = yTrain['label']
    yTest = yTest['label']

    # train on full dataset
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['knn_full', acc, auc])

    # remove 5%
    knn = KNeighborsClassifier(n_neighbors=best_k)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.05)
    knn.fit(new_xTrain, new_yTrain)
    yPred = knn.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['knn_95%', acc, auc])

    # remove 10%
    knn = KNeighborsClassifier(n_neighbors=best_k)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.10)
    knn.fit(new_xTrain, new_yTrain)
    yPred = knn.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['knn_90%', acc, auc])

    # remove 20%
    knn = KNeighborsClassifier(n_neighbors=best_k)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.20)
    knn.fit(new_xTrain, new_yTrain)
    yPred = knn.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['knn_80%', acc, auc])
    result = pd.DataFrame(result, columns=['model', 'Acc', 'AUC'])
    return result

def opt_dt(xTrain, yTrain, xTest, yTest, best_md, best_msl):
    """
    Train a Decision Tree with the best md and msl found previously,
    on the whole dataset, 95% of the dataset, 90% of the dataset,
    and 80% of the dataset, separately. And report the accuracies and AUCs

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.
    best_md : int
        the best Max_Depth for Decision Tree
    best_msl : int
        the best Min_Sample_Leaf for Decision Tree

    Returns
    -------
    result : DataFrame with columns=['model', 'Acc', 'AUC']
        the Accs and AUCs for each Decision Tree model
    """
    result = []
    yTrain = yTrain['label']
    yTest = yTest['label']

    # train on full dataset
    dt = DecisionTreeClassifier(max_depth=best_md, min_samples_leaf=best_msl)
    dt.fit(xTrain, yTrain)
    yPred = dt.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['dt_full', acc, auc])

    # remove 5%
    dt = DecisionTreeClassifier(max_depth=best_md, min_samples_leaf=best_msl)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.05)
    dt.fit(new_xTrain, new_yTrain)
    yPred = dt.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['dt_95%', acc, auc])

    # remove 10%
    dt = DecisionTreeClassifier(max_depth=best_md, min_samples_leaf=best_msl)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.10)
    dt.fit(new_xTrain, new_yTrain)
    yPred = dt.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['dt_90%', acc, auc])

    # remove 20%
    dt = DecisionTreeClassifier(max_depth=best_md, min_samples_leaf=best_msl)
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.20)
    dt.fit(new_xTrain, new_yTrain)
    yPred = dt.predict(xTest)
    acc = metrics.accuracy_score(y_true=yTest, y_pred=yPred)
    auc = metrics.roc_auc_score(y_true=yTest, y_score=yPred)

    result.append(['dt_80%', acc, auc])
    result = pd.DataFrame(result, columns=['model', 'Acc', 'AUC'])
    return result


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # find the optimal hyperparameters
    best_k = kfold_knn(xTrain, yTrain)
    best_md, best_msl = kfold_dt(xTrain, yTrain)

    print('Optimal Hyperparameters: ')
    print('For K-NN: ')
    print('Best k: ', best_k)
    print('\nFor Decision Tree: ')
    print('Best Max_Depth: ', best_md)
    print('Best Min_Sample_Leaf: ', best_msl)
    print('-----------------------------------------------------------------------')

    result = []
    result = pd.DataFrame(result, columns=['model', 'Acc', 'AUC'])
    result = pd.concat([result, opt_knn(xTrain, yTrain, xTest, yTest, best_k)])
    result = pd.concat([result, opt_dt(xTrain, yTrain, xTest, yTest, best_md, best_msl)])

    print('Sensitivity Analysis: ')
    print(result)
    print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    main()
