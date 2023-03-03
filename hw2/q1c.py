import dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_c(xTrain, yTrain, xTest, yTest):
    """
    Draw the plots of different train and test accuracies
    with respect to Max_Depth and Min_Leaf_Sample separately

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

    Returns
    -------
    None
    """
    result1 = []  # store results for different Max_Depth
    result2 = []  # store results for different Min_Sample_Leaf
    for max_d in range(12):
        dtree = dt.DecisionTree('gini', max_d, 10)
        trainAcc, testAcc = dt.dt_train_test(dtree, xTrain, yTrain, xTest, yTest)
        result1.append([max_d, trainAcc, testAcc])

    for min_s in [0, 50, 100, 150, 200]:
        dtree = dt.DecisionTree('gini', 5, min_s)
        trainAcc, testAcc = dt.dt_train_test(dtree, xTrain, yTrain, xTest, yTest)
        result2.append([min_s, trainAcc, testAcc])

    result1 = pd.DataFrame(data=result1, columns=['Max_Depth', 'trainAcc', 'testAcc'])
    result2 = pd.DataFrame(data=result2, columns=['Min_Leaf_Sample', 'trainAcc', 'testAcc'])

    plot = result1.plot.line(x='Max_Depth', title='Accuracy with Different Max_Depth (Min_Leaf_Sample=10)')
    # plot.title('Accuracy with Different Max_Depth (Min_Leaf_Sample=10)')
    plot.set_xlabel('Max_Depth')
    plot.set_ylabel('Acc')

    plt.show()

    plot2 = result2.plot.line(x='Min_Leaf_Sample', title='Accuracy with Different Min_Leaf_Sample (Max_Depth=5)')
    # plot2.title('Accuracy with Different Min_Leaf_Sample (Max_Depth=5)')
    plot2.set_xlabel('Min_Leaf_Sample')
    plot2.set_ylabel('Acc')

    plt.show()


def main():
    # load dataset
    xTrain = pd.read_csv("q4xTrain.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTest = pd.read_csv("q4yTest.csv")

    plot_c(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
