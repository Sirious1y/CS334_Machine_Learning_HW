import numpy as np
import pandas as pd
import perceptron
from perceptron import file_to_numpy


def k_fold(xFeat, y, k):
    """
    Calculate the indices for training and testing using k-fold cross validation
    Parameters
    ----------
    xFeat: nd-array with shape m x d
            The data to predict.
    y: 1-d array or list with shape n
        The true label.
    k: int number of fold

    Returns
    -------
    fold_idx: a list contains indices for training and testing for each fold
    """
    n_samples = xFeat.shape[0]
    size = n_samples // k
    fold_idx = []
    for i in range(k):
        test_start = i * size
        test_end = (i + 1) * size
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
        fold_idx.append((train_idx, test_idx))
    return fold_idx


def perceptron_kfold(xFeat, y, k, epoch):
    """
    Train a perceptron using k fold validation and report average performance
    Parameters
    ----------
    xFeat: nd-array with shape m x d
            The data to predict.
    y: 1-d array or list with shape n
        The true label.
    k: int number of fold
    epoch: int number of epoch

    Returns
    -------
    mean number of mistakes
    """
    fold_idx = k_fold(xFeat, y, k)
    result = []
    for train_idx, test_idx in fold_idx:
        xTrain = xFeat[train_idx]
        xTest = xFeat[test_idx]
        yTrain = y[train_idx]
        yTest = y[test_idx]
        model = perceptron.Perceptron(epoch)
        model.train(xTrain, yTrain)
        yHat = model.predict(xTest)

        result.append(perceptron.calc_mistakes(yHat, yTest))

    return np.mean(result)


def main():
    np.random.seed(334)
    k = 5
    epoch_list = [5, 10, 20, 30, 50, 100, 200]
    results = {}
    # find optimal epochs
    # binary
    xTrain = file_to_numpy('xTrain_binary.csv')
    yTrain = file_to_numpy('yTrain_binary.csv')
    for epoch in epoch_list:
        result = perceptron_kfold(xTrain, yTrain, k, epoch)
        results.update({epoch: result})
    print('5-fold validation performance on binary dataset (epoch: mistakes):')
    print(results)
    print('------------------------------------------------------------------------------------------------')

    # count
    results = {}
    xTrain = file_to_numpy('xTrain_count.csv')
    yTrain = file_to_numpy('yTrain_count.csv')
    for epoch in epoch_list:
        result = perceptron_kfold(xTrain, yTrain, k, epoch)
        results.update({epoch: result})
    print('5-fold validation performance on count dataset (epoch: mistakes):')
    print(results)
    print('------------------------------------------------------------------------------------------------')

    vocab = np.array(pd.read_csv('xTrain_binary.csv', index_col=0).columns)
    # train with optimal epoch
    # binary
    results = {}
    xTrain = file_to_numpy('xTrain_binary.csv')
    yTrain = file_to_numpy('yTrain_binary.csv')
    xTest = file_to_numpy('xTest_binary.csv')
    yTest = file_to_numpy('yTest_binary.csv')
    model = perceptron.Perceptron(20)
    model.train(xTrain, yTrain)
    yHat = model.predict(xTrain)
    mistakes = perceptron.calc_mistakes(yHat, yTrain)
    results.update({'Train': mistakes})
    yHat = model.predict(xTest)
    mistakes = perceptron.calc_mistakes(yHat, yTest)
    results.update({'Test': mistakes})
    print('Results of training on binary dataset with 20 epochs: ')
    print(results)
    sort_idx = model.w.argsort()
    min = vocab[sort_idx[:15]]
    max = np.flip(vocab[sort_idx[-15:]])
    print('words with max 15 weights: ')
    print(max)
    print('words with min 15 weights: ')
    print(min)

    print('------------------------------------------------------------------------------------------------')
    # count
    results = {}
    xTrain = file_to_numpy('xTrain_count.csv')
    yTrain = file_to_numpy('yTrain_count.csv')
    xTest = file_to_numpy('xTest_count.csv')
    yTest = file_to_numpy('yTest_count.csv')
    model = perceptron.Perceptron(200)
    model.train(xTrain, yTrain)
    yHat = model.predict(xTrain)
    mistakes = perceptron.calc_mistakes(yHat, yTrain)
    results.update({'Train': mistakes})
    yHat = model.predict(xTest)
    mistakes = perceptron.calc_mistakes(yHat, yTest)
    results.update({'Test': mistakes})
    print('Results of training on count dataset with 200 epochs: ')
    print(results)
    sort_idx = model.w.argsort()
    min = vocab[sort_idx[:15]]
    max = np.flip(vocab[sort_idx[-15:]])
    print('words with max 15 weights: ')
    print(max)
    print('words with min 15 weights: ')
    print(min)


if __name__ == "__main__":
    main()
