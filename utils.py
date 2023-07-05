import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    #Get the weights of the model
    if model.fit_intercept:
        params = [model.coef_, 
                  model.intercept_
                  ]
    else:
        params = [model.coef_]
    return params

def set_model_params(model: LogisticRegression, params: LogRegParams)->LogisticRegression:
    #Set the weights of the model
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model
    
def set_initial_params(model: LogisticRegression):
    #Set the weights of the model
    n_classes = 2
    n_features = 186
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes,n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def load_data() -> Dataset:
    normal_df = pd.read_csv('datasets/ptbdb_normal.csv', header=None)
    abnormal_df = pd.read_csv('datasets/ptbdb_abnormal.csv', header=None)
    df = pd.concat([normal_df, abnormal_df], axis=0, ignore_index=True)
    print(df.shape)
    #Shuffle the data
    df = df.sample(df.shape[0], random_state=42)
    #X values should be until row 187
    X = df.iloc[:, :186].values
    #Y values should be from row 187
    y = df.iloc[:, 187].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trainset = (X_train,y_train)
    testset = (X_test, y_test)
    return trainset, testset

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

def partition_data(X: np.ndarray, y: np.ndarray, n_partitions: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    length = len(X)
    if length % n_partitions != 0:
        # Adjust n_partitions to achieve equal or close-to-equal divisions
        n_partitions = length // (length // n_partitions)

    X_split = np.array_split(X, n_partitions)
    y_split = np.array_split(y, n_partitions)
    return X_split, y_split

def train_val_divide_local_datasets(local_X: List[np.ndarray], local_y: List[np.ndarray], valid_fraction: float) -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Split each local dataset into train and validation."""
    X_trains = []
    y_trains = []
    X_valids = []
    y_valids = []
    partition_size = local_X[0].shape[0]
    for client_x, client_y in zip(local_X, local_y):
        train_end_idx = int((1 - valid_fraction) * partition_size)
        X_trains.append(client_x[:train_end_idx])
        y_trains.append(client_y[:train_end_idx])
        X_valids.append(client_x[train_end_idx:])
        y_valids.append(client_y[train_end_idx:])
    
    return (X_trains, y_trains), (X_valids, y_valids)

def load_datasets(n_partitions: int, valid_fraction: float):
    (X_train, y_train), (centralized_X, centralized_y) = load_data()
    X_split, y_split = partition_data(X_train, y_train, n_partitions)
    (X_trains, y_trains), (X_valids, y_valids) = train_val_divide_local_datasets(X_split, y_split, valid_fraction)
    return (X_trains, y_trains), (X_valids, y_valids), (centralized_X, centralized_y)