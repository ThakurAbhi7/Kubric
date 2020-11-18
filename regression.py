import requests
import pandas
import scipy
import numpy
import sys
import matplotlib.pyplot as plt

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...
    import os, csv

    # os.system("wget "+TRAIN_DATA_URL)
    # os.system("wget "+TEST_DATA_URL)
    with open("linreg_train.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        train_X = next(reader)[1:]
        train_Y = next(reader)[1:]
    for i in range(len(train_X)):
        train_X[i] = float(train_X[i])**0.5
    train_X = [train_X, [1]*len(train_X)]
    train_X = numpy.array(train_X).astype('float64').T
    train_Y = numpy.array(train_Y).astype('float64').T
    I = numpy.array([[0.8, 0], [0, 0.00008]])
    XtX = numpy.linalg.inv(train_X.T.dot(train_X))  +10000000*I
    XtY = train_X.T.dot(train_Y)
    w, b = numpy.linalg.solve(XtX, XtY)
    for i in range(len(area)):
        area[i] = float(area[i])**0.5
    area = numpy.array(area).astype('float').T
    plt.plot(train_X[:,0], train_Y, 'bs')
    plt.plot([i for i in range(200)], [i*w+b for i in range(200)])
    plt.show()
    return area*w+b

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
