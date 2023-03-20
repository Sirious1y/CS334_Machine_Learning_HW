import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column
    Namely, extract month and time as total minutes

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    df[['date', 'time']] = df['date'].str.split(pat=' ', n=1, expand=True)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['time'] = pd.to_timedelta(df['time']+':00')
    df['time'] = df['time'].dt.total_seconds() / 60

    df = df.drop(columns=['date'])
    df = df.reindex(columns=['month', 'day', 'time'] + list(df.columns[:-3]))

    return df


def pearson(xFeat, y):
    """
    Plot the pearson correlation as a heatmap
    And determine correlated features that need to be dropped
    Parameters
    ----------
    xFeat : nd-array with shape n x d
        Training data
    y : 1d array with shape n
        Array of labels associated with training data.

    Returns
    -------

    """
    df = pd.concat([xFeat, y], axis=1)
    pearson = df.corr(method='pearson')

    plt.figure(figsize=(24, 24))
    sns.heatmap(pearson, annot=True, fmt='.2f', center=0)
    plt.show()

    upper = pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(np.bool_))
    global drop  # store the column names that need to be dropped
    drop = [column for column in upper.columns if any(upper[column]>0.8)]

    return


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    df = df.drop(columns=drop)

    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # Normalize the data
    trainDF = pd.DataFrame(normalize(trainDF, axis=0), columns=trainDF.columns)
    testDF = pd.DataFrame(normalize(testDF, axis=0), columns=testDF.columns)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    parser.add_argument("--train_yFile",
                        default="eng_yTrain.csv",
                        help="filename of the training label")
    parser.add_argument("--test_yFile",
                        default="eng_yTest.csv",
                        help="filename of the test label")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    yTrain = pd.read_csv(args.train_yFile)
    yTest = pd.read_csv(args.test_yFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)

    pearson(xNewTrain, yTrain)

    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
