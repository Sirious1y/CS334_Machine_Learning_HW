import sgdLR
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lr import LinearRegression, file_to_numpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="new_xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="eng_yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="new_xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="eng_yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # randomly select 40% of the data
    new_xTrain, new_xTest, new_yTrain, new_yTest = train_test_split(xTrain, yTrain, test_size=0.6)

    # Q3 b
    for lr in (1, 0.1, 0.01, 0.001, 0.0001, 0.00001):
        model = sgdLR.SgdLR(lr, 1, 10)
        stats = model.train_predict(new_xTrain, new_yTrain, xTest, yTest)

        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = []
        for i in range(0, new_xTrain.shape[0] * 10, new_xTrain.shape[0]):
            y.append(stats.get(i).get('test-mse'))

        # print(y)

        plt.plot(x, y, label='lr='+str(lr))

    plt.title('MSE curve of different learning rates')
    plt.xlabel('Epoch')
    plt.ylabel('Test-MSE')
    ax = plt.gca()
    ax.set_ylim([0, 10])
    plt.legend()
    plt.show()

    # Q3 c
    # lr = 0.0001 is the optimal learning rate
    op_lr = 0.0001
    model = sgdLR.SgdLR(op_lr, 1, 10)
    op_stats = model.train_predict(xTrain, yTrain, xTest, yTest)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = []
    y2 = []
    for i in range(0, xTrain.shape[0] * 10, xTrain.shape[0]):
        y1.append(op_stats.get(i).get('train-mse'))
        y2.append(op_stats.get(i).get('test-mse'))

    plt.clf()
    plt.plot(x, y1, label='Train MSE')
    plt.plot(x, y2, label='Test MSE')
    ax = plt.gca()
    ax.set_ylim([0, 5])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
