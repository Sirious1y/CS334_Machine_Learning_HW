import argparse
import matplotlib.pyplot as plt
import standardLR
import sgdLR
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

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad = 2.0)
    ax1.title.set_text('Train MSE')
    ax2.title.set_text('Test MSE')

    # Set batch size to 1, 5, 30, 390, 1290, 3354, n
    bs = [1, 30, 390, 1290, xTrain.shape[0]]
    n = 0
    SGDmodel = sgdLR.SgdLR(0.0001, 1, 10)
    stats = SGDmodel.train_predict(xTrain, yTrain, xTest, yTest)
    time = []
    train = []
    test = []
    for i in range(0, xTrain.shape[0] * 5 // int(bs[n]), int(xTrain.shape[0]) // int(bs[n])):
        time.append(stats.get(i).get('time'))
        train.append(stats.get(i).get('train-mse'))
        test.append(stats.get(i).get('test-mse'))
    ax1.plot(time, train, label='bs=' + str(bs[n]))
    ax2.plot(time, test, label='bs=' + str(bs[n]))
    n += 1


    SGDmodel = sgdLR.SgdLR(0.0001, 30, 100)
    stats = SGDmodel.train_predict(xTrain, yTrain, xTest, yTest)
    time = []
    train = []
    test = []
    for i in range(0, xTrain.shape[0] * 100 // int(bs[n]), int(xTrain.shape[0]) // int(bs[n])):
        time.append(stats.get(i).get('time'))
        train.append(stats.get(i).get('train-mse'))
        test.append(stats.get(i).get('test-mse'))
    ax1.plot(time, train, label='bs=' + str(bs[n]))
    ax2.plot(time, test, label='bs=' + str(bs[n]))
    n += 1

    SGDmodel = sgdLR.SgdLR(0.0001, 390, 500)
    stats = SGDmodel.train_predict(xTrain, yTrain, xTest, yTest)
    time = []
    train = []
    test = []
    for i in range(0, xTrain.shape[0] * 500 // int(bs[n]), int(xTrain.shape[0]) // int(bs[n])):
        time.append(stats.get(i).get('time'))
        train.append(stats.get(i).get('train-mse'))
        test.append(stats.get(i).get('test-mse'))
    ax1.plot(time, train, label='bs=' + str(bs[n]))
    ax2.plot(time, test, label='bs=' + str(bs[n]))
    n += 1

    SGDmodel = sgdLR.SgdLR(0.0001, 1290, 1200)
    stats = SGDmodel.train_predict(xTrain, yTrain, xTest, yTest)
    time = []
    train = []
    test = []
    for i in range(0, xTrain.shape[0] * 1200 // int(bs[n]), int(xTrain.shape[0]) // int(bs[n])):
        time.append(stats.get(i).get('time'))
        train.append(stats.get(i).get('train-mse'))
        test.append(stats.get(i).get('test-mse'))
    ax1.plot(time, train, label='bs=' + str(bs[n]))
    ax2.plot(time, test, label='bs=' + str(bs[n]))
    n += 1

    SGDmodel = sgdLR.SgdLR(0.0001, xTrain.shape[0], 4000)
    time = []
    train = []
    test = []
    for i in range(0, xTrain.shape[0] * 4000 // int(bs[n]), int(xTrain.shape[0]) // int(bs[n])):
        time.append(stats.get(i).get('time'))
        train.append(stats.get(i).get('train-mse'))
        test.append(stats.get(i).get('test-mse'))
    ax1.plot(time, train, label='bs=' + str(bs[n]))
    ax2.plot(time, test, label='bs=' + str(bs[n]))
    n += 1

    # for i, batch_size in zip(range(7), bs):
    #     ax1.plot(time[i], train[i], marker="o", label='bs='+str(batch_size))
    #     ax2.plot(time[i], test[i], marker="o", label='bs='+str(batch_size))

    # Closed form model
    stdmodel = standardLR.StandardLR()
    stats = stdmodel.train_predict(xTrain, yTrain, xTest, yTest)
    ax1.plot(list(stats.values())[-1].get('time'), list(stats.values())[-1].get('train-mse'), marker="o", label='Closed Form')
    ax2.plot(list(stats.values())[-1].get('time'), list(stats.values())[-1].get('test-mse'), marker="o", label='Closed Form')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
