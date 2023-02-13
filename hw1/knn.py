import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Knn(object):
    k = 0    # number of neighbors to use
    trainX = np.empty(0) # attributes of training set
    trainy = np.empty(0) # labels of training set
    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

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
        # transform from dataframe to np array and store in the class
        if isinstance(xFeat, pd.DataFrame):
            self.trainX = xFeat.to_numpy()
        else:
            self.trainX = xFeat
        if isinstance(y, pd.DataFrame):
            self.trainy = y.to_numpy()
        else:
            self.trainy = y
        return self


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
        if isinstance(xFeat, pd.DataFrame):
            xFeat = xFeat.values
        for a in xFeat: # for each testing sample
            distances = [] # variable to store the distances with all training samples
            distances = np.linalg.norm(self.trainX - a, axis=1)
            k_index = distances.argsort()[:self.k]
            knearest = self.trainy[k_index]
            knearest = pd.DataFrame(knearest)
            yHat.append(knearest.value_counts().idxmax())

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = 0
    correct = 0
    all = len(yTrue)
    for i in range(all):
        if np.isclose(yHat[i], yTrue[i]):
            correct+=1

    acc = correct/all
    return acc

def plot_k(xTrain, yTrain, xTest, yTest):
    """
    Plot the training accuracy and testing accuracy with respect to k (0 <= k <= 10)
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
    for k in range(1, 11): # for each value of k, run the knn and store the accuracies
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
        # predict the training dataset
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        # predict the test dataset
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])

        results.append([k, trainAcc, testAcc])

    results = pd.DataFrame(data=results, columns=['k', 'trainAcc', 'testAcc'])

    # plot the accuracies with respect to k
    plot = results.plot.line(x='k')
    plt.title('Accuracy with Different K')
    plt.xticks(np.arange(0, 11, 1.0))
    plot.set_xlabel('K')
    plot.set_ylabel('Acc')

    plt.show()

    return None



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    plot_k(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
