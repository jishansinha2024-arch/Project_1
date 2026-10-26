import os

import numpy as np
import pandas as pd


def load_data():
    train_path = "/kaggle/input/digit-recognizer/train.csv"
    if not os.path.exists(train_path):
        train_path = os.path.join(os.path.dirname(__file__), "train.csv")

    data = np.array(pd.read_csv(train_path))
    row_count, column_count = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:column_count] / 255.0

    data_train = data[1000:row_count].T
    Y_train = data_train[0]
    X_train = data_train[1:column_count] / 255.0

    return X_train, Y_train, X_dev, Y_dev
