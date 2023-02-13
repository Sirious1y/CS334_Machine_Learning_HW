from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    """
    Load the iris dataset from sklearn
    and convert to a dataframe

    Returns
    -------
    df: dataframe
        the iris data with the target value as the last column
    """
    data = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                      columns=data['feature_names'] + ['target'])
    df['target'].replace([0.0, 1.0, 2.0], data['target_names'], inplace=True) # replace the numerical target value with species name
    return df

def main():
    iris = load_dataset()
    print(iris)

    # Distribution of Sepal Length of each Species
    data1 = iris[['sepal length (cm)', 'target']]
    bp1 = data1.boxplot(column='sepal length (cm)', by='target')
    plt.title('Distribution of Sepal Length of each Species')
    plt.suptitle('')
    bp1.set_xlabel('Species')
    bp1.set_ylabel('Sepal length (cm)')
    plt.show()
    plt.clf()

    # Distribution of Sepal Width of each Species
    data2 = iris[['sepal width (cm)', 'target']]
    bp2 = data2.boxplot(column='sepal width (cm)', by = 'target')
    plt.title('Distribution of Sepal Width of each Species')
    plt.suptitle('')
    bp2.set_xlabel('Species')
    bp2.set_ylabel('Sepal width (cm)')
    plt.show()
    plt.clf()

    # Distribution of Petal Length of each Species
    data3 = iris[['petal length (cm)', 'target']]
    bp3 = data3.boxplot(column='petal length (cm)', by='target')
    plt.title('Distribution of Petal Length of each Species')
    plt.suptitle('')
    bp3.set_xlabel('Species')
    bp3.set_ylabel('Petal length (cm)')
    plt.show()
    plt.clf()

    # Distribution of Petal Width of each Species
    data4 = iris[['petal width (cm)', 'target']]
    bp4 = data4.boxplot(column='petal width (cm)', by='target')
    plt.title('Distribution of Petal Width of each Species')
    plt.suptitle('')
    bp4.set_xlabel('Species')
    bp4.set_ylabel('Petal width (cm)')
    plt.show()
    plt.clf()

    colors = {'setosa':'magenta', 'versicolor':'royalblue', 'virginica':'springgreen'}

    # Sepal Width V.S. Sepal Length
    sepal_data = iris[['sepal length (cm)', 'sepal width (cm)', 'target']]
    fig, ax = plt.subplots()
    grouped = sepal_data.groupby('target')
    for key, group in grouped:
       group.plot(ax=ax, kind='scatter', x='sepal length (cm)', y='sepal width (cm)', label=key, color=colors[key])
    plt.title('Sepal Width V.S. Sepal Length')
    plt.suptitle('')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    plt.show()
    plt.clf()

    # Petal Width V.S. Petal Length
    petal_data = iris[['petal length (cm)', 'petal width (cm)', 'target']]
    fig, ax = plt.subplots()
    grouped = petal_data.groupby('target')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='petal length (cm)', y='petal width (cm)', label=key, color=colors[key])
    plt.title('Petal Width V.S. Petal Length')
    plt.suptitle('')
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()