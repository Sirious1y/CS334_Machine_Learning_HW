import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    data = pd.read_csv(filename, header=None, names=['email'])
    data['label'] = data['email'].str[0]
    data['email'] = data['email'].str[1:]
    train, test = train_test_split(data, shuffle=True, random_state=0)
    return train, test


def build_vocab_map(train):
    """
    Build the vocab map on the given dataset
    Parameters
    ----------
    train: dataframe with shape m x 2
        the training data with a column being string contents
        and the other column being the label

    Returns
    -------
    vocab_map: a dictionary of words that appears in at least 30 e-mails in the training set
        and their numbers of appearances
    """
    vocab_map = {}
    vectorizer = CountVectorizer(binary=True)
    count_matrix = vectorizer.fit_transform(train['email'])
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())
    for word in vectorizer.get_feature_names_out():
        count = sum(df[word])
        if count >= 30:
            vocab_map[word] = count

    return vocab_map


def construct_binary(train, test, vocab_map):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    vectorizer = CountVectorizer(binary=True, vocabulary=list(vocab_map.keys()))
    xTrain = vectorizer.transform(train['email'])
    yTrain = train['label']
    xTest = vectorizer.transform((test['email']))
    yTest = test['label']
    return xTrain, yTrain, xTest, yTest


def construct_count(train, test, vocab_map):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    vectorizer = CountVectorizer(binary=False, vocabulary=list(vocab_map.keys()))
    xTrain = vectorizer.transform(train['email'])
    yTrain = train['label']
    xTest = vectorizer.transform((test['email']))
    yTest = test['label']
    return xTrain, yTrain, xTest, yTest


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    train, test = model_assessment(args.data)
    vocab_map = build_vocab_map(train)
    xTrain_bi, yTrain_bi, xTest_bi, yTest_bi = construct_binary(train, test, vocab_map)
    xTrain_n, yTrain_n, xTest_n, yTest_n = construct_count(train, test, vocab_map)

    xTrain_bi = pd.DataFrame(xTrain_bi.toarray(), columns=list(vocab_map.keys()))
    xTest_bi = pd.DataFrame(xTest_bi.toarray(), columns=list(vocab_map.keys()))
    yTrain_bi = yTrain_bi.to_frame()
    yTest_bi = yTest_bi.to_frame()
    xTrain_bi.to_csv('xTrain_binary.csv')
    xTest_bi.to_csv('xTest_binary.csv')
    yTrain_bi.to_csv('yTrain_binary.csv')
    yTest_bi.to_csv('yTest_binary.csv')

    xTrain_n = pd.DataFrame(xTrain_n.toarray(), columns=list(vocab_map.keys()))
    xTest_n = pd.DataFrame(xTest_n.toarray(), columns=list(vocab_map.keys()))
    yTrain_n = yTrain_n.to_frame()
    yTest_n = yTest_n.to_frame()
    xTrain_n.to_csv('xTrain_count.csv')
    xTest_n.to_csv('xTest_count.csv')
    yTrain_n.to_csv('yTrain_count.csv')
    yTest_n.to_csv('yTest_count.csv')


if __name__ == "__main__":
    main()
