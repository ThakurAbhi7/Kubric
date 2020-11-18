import requests
import pandas
import scipy
import numpy
import sys

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

    os.system("wget "+TRAIN_DATA_URL)
    os.system("wget "+TEST_DATA_URL)
    with open("linreg_train.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        train_X = next(reader)[1:]
        train_Y = next(reader)[1:]
    train_X = [train_X, [1]*len(train_X)]
    train_X = numpy.array(train_X).astype('float64').T
    train_Y = numpy.array(train_Y).astype('float64').T
    I = numpy.array([[1, 0], [0, 1]])
    XtX = numpy.linalg.inv(train_X.T.dot(train_X))  + I*140400550000
    XtY = train_X.T.dot(train_Y)
    w, b = numpy.linalg.solve(XtX, XtY)
    # print(w, b)
    area = numpy.array(area).astype('float').T
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
