import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        timeElapse = 0
        start = time.time()
        n = 0

        # Train
        xTrain = np.hstack((np.ones((xTrain.shape[0], 1)), xTrain))
        xTest = np.hstack((np.ones((xTest.shape[0], 1)), xTest))
        self.beta = np.zeros((xTrain.shape[1], 1))

        for epoch in range(self.mEpoch):
            idx = np.random.permutation(xTrain.shape[0])

            for batch_start in range(0, xTrain.shape[0], self.bs):
                batch_idx = idx[batch_start: batch_start+self.bs]

                gradient = (xTrain[batch_idx].T.dot(xTrain[batch_idx].dot(self.beta) - yTrain[batch_idx]) / self.bs)

                self.beta -= self.lr * gradient

                trainMSE = self.mse(xTrain, yTrain)
                testMSE = self.mse(xTest, yTest)
                end = time.time()
                timeElapse = end - start

                trainStats[n] = {
                        'time': timeElapse,
                        'train-mse': trainMSE,
                        'test-mse': testMSE
                }

                n += 1

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

