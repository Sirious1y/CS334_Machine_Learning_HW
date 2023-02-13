import argparse
import numpy as np
import pandas as pd
import knn
import matplotlib.pyplot as plt


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    mean = xTrain.mean(axis=0)
    var = xTrain.std(axis=0)
    xTrain = (xTrain - mean) / var
    xTest = (xTest - mean) / var

    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    min = xTrain.min(axis=0)
    max = xTrain.max(axis=0)
    xTrain = (xTrain - min) / (max - min)
    xTest = (xTest - min) / (max - min)

    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    for i in range(2):
        add_feature = np.random.normal(loc=0, scale=1, size=xTrain.shape[0])
        add_feature = np.transpose(np.atleast_2d(add_feature))
        xTrain = np.append(xTrain, add_feature, axis=1)
        add_feature = np.random.normal(loc=0, scale=1, size=xTest.shape[0])
        add_feature = np.transpose(np.atleast_2d(add_feature))
        xTest = np.append(xTest, add_feature, axis=1)

    return xTrain, xTest


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
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
    model = knn.Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn.accuracy(yHatTest, yTest['label'])


def plot_acc(xTrain, yTrain, xTest, yTest):
    """
    Plot the testing accuracies of dataset with different preprocessing techniques with respect of different K values

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Attributes of training data
    yTrain : 1d-array with shape n
        Labels of training data
    xTest : nd-array with shape n x d
        The data to predict
    yTest : 1d-array with shape n
        Labels of testing data

    Returns
    -------
    None

    """
    results = [] # variable to store the accuracies of each k value
    for k in range(1,11):
        acc1 = knn_train_test(k, xTrain, yTrain, xTest, yTest)
        # print("Test Acc (no-preprocessing):", acc1)
        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = standard_scale(xTrain, xTest)
        acc2 = knn_train_test(k, xTrainStd, yTrain, xTestStd, yTest)
        # print("Test Acc (standard scale):", acc2)
        # preprocess the data using min max scaling
        xTrainMM, xTestMM = minmax_range(xTrain, xTest)
        acc3 = knn_train_test(k, xTrainMM, yTrain, xTestMM, yTest)
        # print("Test Acc (min max scale):", acc3)
        # add irrelevant features
        xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
        acc4 = knn_train_test(k, xTrainIrr, yTrain, yTrainIrr, yTest)
        # print("Test Acc (with irrelevant feature):", acc4)
        results.append([k, acc1, acc2, acc3, acc4])

    results = pd.DataFrame(data=results, columns=['k', 'NoPreprocessing Acc', 'Standardization Acc', 'Min Max Acc',
                                                  'Irrelevant Feature Acc'])

    # plot the accuracies with respect to k
    plot = results.plot.line(x='k')
    plt.title('Accuracies of Different Preprocessing Techniques with Respect to K')
    plt.xticks(np.arange(0, 11, 1.0))
    plot.set_xlabel('K')
    plot.set_ylabel('Acc')

    plt.show()

    return None


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

    plot_acc(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
