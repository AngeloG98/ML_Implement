from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/linear_classification_data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    return X_train, X_test, Y_train, Y_test

class SVM:
    def __init__(self) -> None:
        pass
