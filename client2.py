import argparse
import warnings
import flwr as fl
import pandas as pd
import numpy as np
import sklearn
import utils
import tensorflow as tf
import tensorflow.keras as keras
import torch
from flwr.common import (NDArrays,GetParametersRes, FitIns, EvaluateRes, EvaluateIns, FitRes)
import timeit
from typing import Tuple, List, Optional, Callable, Dict, Union

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
num_clients = 10
batch_size = 32
valid_fraction = 0.2
num_rounds = 5
XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def get_model_parameters(model):
    #Get the weights of the model
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
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
    n_features = 187
    model.classes_ = np.array([0,1])
    model.coef_ = np.zeros((n_classes,n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((2,))
    



class LogisticRegressionClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: str,
                 model: LogisticRegression,
                 X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 X_test: np.ndarray, 
                 y_test: np.ndarray,
                 )->None:
        self.cid=cid
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_weights = None
    
    def fit_model(self,parameters:LogRegParams,config):
        print("Client "+str(self.cid)+" is training")
        model = LogisticRegression(random_state=0, max_iter=1000)
        set_model_params(model=model, params = parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            print("Training finished for round"+str(config["current_round"]))
        return get_model_parameters(self.model)
    
    def get_parameters(self,config):
        print("Client "+str(self.cid)+" is getting parameters")
        return get_model_parameters(self.model)
    
   
    def evaluate_model(self,parameters,config):
        set_model_params(model=self.model, params = parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def load_data() -> Tuple[np.ndarray, np.ndarray]:
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

def main() -> None:
    """Load data, create and start client."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=str,required=True)
    parser.add_argument("--server_address", type=str)
    #parser.add_argument("--dataset", type=str)
    #parser.add_argument("--model", type=str, default= "LogisticRegression", help="Model to use for training")
    args = parser.parse_args()
    model = LogisticRegression(random_state=0, max_iter=1000)
    (X_trains, y_trains), (X_valids, y_valids), (centralized_X, centralized_y) = load_datasets(
    n_partitions=num_clients, 
    valid_fraction=valid_fraction)

    client = LogisticRegressionClient(args.cid, model, X_trains, y_trains, X_valids, y_valids)
    fl.client.start_numpy_client(server_address="172.23.249.207:8080", client=client)

if __name__ == "__main__":
    main()






