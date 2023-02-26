import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Node:
    feature_idx = None
    threshold = None
    left = None
    right = None
    label = None

    def __init__(self, feature_idx, threshold, left, right, label):
        """
        Tree node constructor

        Parameters
        ----------
        feature_idx: int
            index of the feature which current node represents
        threshold: float
            threshold that separates left and right node
        left: node
            left node
        right: node
            right node
        label: boolean
            if the node is a leaf node, label is the corresponding prediction result
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        return self

    def gini(self, y):
        """
        Calculate the Gini Index of given array of labels

        Parameters
        ----------
        y: 1d array
            Array of labels of which the Gini Index is calculated
        Returns
        -------
        result: float
            the Gini Index
        """
        values, counts = np.unique(y, return_counts = True)
        prob = counts / len(y)
        result = 1 - np.sum(np.square(prob))
        return result

    def entropy(self, y):
        """
        Calculate the entropy of given array of labels

        Parameters
        ----------
        y: 1d array
            Array of labels of which the entropy is calculated
        Returns
        -------
        result: float
            the entropy
        """
        values, counts = np.unique(y, return_counts = True)
        prob = counts / len(y)
        result = -np.sum(prob * np.log2(prob))
        return result

    def info_gain(self, xFeat, y, feat_idx, threshold):
        """
        Calculate the information gain

        Parameters
        ----------
        xFeat: nd-array with shape n x d
            Part of training data
        y: 1d array with shape n
            Array of labels associated with training data.
        feat_idx: int
            index of the feature upon which information gain is calculated
        threshold: float
            where to split the given feature
        Returns
        -------
        information gain
        """
        left_y = y[xFeat[:, feat_idx] < threshold]
        right_y = y[xFeat[:, feat_idx] >= threshold]

        left_n = len(left_y)
        right_n = len(right_y)
        total = len(y)

        left_ent = self.entropy(left_y)
        right_ent = self.entropy(right_y)
        ent = self.entropy(y)

        result = ent - left_ent * left_n / total - right_ent * right_n / total

        return result

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain", default="q4xTrain.csv",
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
