from data import load_data
from model import get_accuracy, make_predictions
from training import gradient_descent
from visualization import test_prediction


def main():
    X_train, Y_train, X_dev, Y_dev = load_data()
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

    for index in range(4):
        test_prediction(index, X_train, Y_train, W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    get_accuracy(dev_predictions, Y_dev)


if __name__ == "__main__":
    main()
