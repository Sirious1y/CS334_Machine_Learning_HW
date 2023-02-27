import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Node:
    feature = None
    threshold = None
    left = None
    right = None
    label = None

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        """
        Tree node constructor

        Parameters
        ----------
        feature: str
            the feature which current node represents
        threshold: float
            threshold that separates left and right node
        left: node
            left node
        right: node
            right node
        label: boolean
            if the node is a leaf node, label is the corresponding prediction result
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = None

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
        self.tree = None

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
        self.tree = self.build_tree(xFeat, y, 0)
        return self

    def build_tree(self, xFeat, y, depth):
        sample_n, feature_n = xFeat.shape
        label_n = len(np.unique(y))

        # Termination
        if depth == self.maxDepth or sample_n <= self.minLeafSample or label_n == 1:
            values, counts = np.unique(y, return_counts = True)
            return Node(label = values[counts.argmax()])

        # Find the feature and threshold
        split_feature, split_threshold = self.split(xFeat, y)

        left_idx = xFeat.loc[:, split_feature] < split_threshold
        right_idx = xFeat.loc[:, split_feature] >= split_threshold
        xFeat = xFeat.drop(columns=split_feature)
        left_tree = self.build_tree(xFeat[left_idx], y[left_idx], depth + 1)
        right_tree = self.build_tree(xFeat[right_idx], y[right_idx], depth + 1)

        return Node(feature=split_feature, threshold=split_threshold, left=left_tree, right=right_tree)

    def split(self, xFeat, y):
        """
        Find the best feature and its best threshold to split the tree
        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        feature_chosen: int
            the index of best feature
        threshold_chosen: float
            threshold at which the samples are split
        """
        sample_n, feature_n = xFeat.shape
        feature_chosen = None
        threshold_chosen = None

        if self.criterion == 'gini':
            gini_chosen = 2
            for feature in xFeat.columns:
                thresholds = np.unique(xFeat.loc[:, feature])
                for threshold in thresholds:
                    current_gini = self.gini_index(xFeat, y, feature, threshold)
                    if current_gini < gini_chosen:
                        feature_chosen = feature
                        threshold_chosen = threshold
                        gini_chosen = current_gini

        elif self.criterion == 'entropy':
            info_chosen = -1
            for feature in xFeat.columns:
                thresholds = np.unique(xFeat.loc[:, feature])
                for threshold in thresholds:
                    current_info = self.info_gain(xFeat, y, feature, threshold)
                    if current_info < info_chosen:
                        feature_chosen = feature
                        threshold_chosen = threshold
                        info_chosen = current_info

        return feature_chosen, threshold_chosen

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

    def gini_index(self, xFeat, y, feature, threshold):
        """
        Calculate the weighted average of gini index

        Parameters
        ----------
        xFeat: nd-array with shape n x d
            Part of training data
        y: 1d array with shape n
            Array of labels associated with training data.
        feature: str
            the feature upon which weighted average of gini index is calculated
        threshold: float
            where to split the given feature
        Returns
        -------
        result: float
            information gain
        """
        left_y = y[xFeat.loc[:, feature] < threshold]
        right_y = y[xFeat.loc[:, feature] >= threshold]
        left_n = len(left_y)
        right_n = len(right_y)
        total = len(y)

        left_gini = self.gini(left_y)
        right_gini = self.gini(right_y)
        result = left_n / total * left_gini + right_n / total * right_gini
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

    def info_gain(self, xFeat, y, feature, threshold):
        """
        Calculate the information gain

        Parameters
        ----------
        xFeat: nd-array with shape n x d
            Part of training data
        y: 1d array with shape n
            Array of labels associated with training data.
        feature: str
            the feature upon which information gain is calculated
        threshold: float
            where to split the given feature
        Returns
        -------
        result: float
            information gain
        """
        left_y = y[xFeat.loc[:, feature] < threshold]
        right_y = y[xFeat.loc[:, feature] >= threshold]

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
