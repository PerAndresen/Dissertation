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

num_clients = 10
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
    '''
        for i, param in enumerate(model.state_dict().items()):
            param[1].data = weights[i]
        return model
    '''
def set_initial_params(model: LogisticRegression):
    #Set the weights of the model
    n_classes = 2
    n_features = 187
    model.classes_ = np.array([0,1])
    model.coef_ = np.zeros((2,187))
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
    
    def get_parameters(self,config):
        print("Client "+str(self.cid)+" is getting parameters")
        return get_model_parameters(self.model)
    
    def fit_model(self,parameters,config):
        set_model_params(model=self.model, params = parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            print("Training finished for round"+str(config["current_round"]))
        return get_model_parameters(self.model), len(self.X_train), {}
    
    def evaluate_model(self,parameters,config):
        set_model_params(model=self.model, params = parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}
    
def load_data():
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
    return X, y

def split_data(X,y, num_clients, client_id):
    client_id -= 1
    data_size = len(X)
    size_per_client = data_size // num_clients
    remainder = data_size % num_clients
    start = client_id * size_per_client
    end = (client_id + 1) * size_per_client if client_id < num_clients - 1 else data_size

    if client_id < remainder:
        start += client_id
        end += client_id + 1
    else: 
        start+= remainder
        end+=remainder

    return X[start:end], y[start:end]


def main() -> None:
    """Load data, create and start client."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=int,required=True)
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_clients", type=int, required=True)

    args = parser.parse_args()
    cid = args.cid


    X, y = load_data()
    X_partition, y_partition= split_data(X, y, num_clients, cid)
    X_train, X_test, y_train, y_test = train_test_split(X_partition, y_partition, test_size=0.2)
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train label counts:"+str(args.cid)+" ",dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test label counts:"+str(args.cid)+" ",dict(zip(unique, counts)))
    model = LogisticRegression(random_state=0, max_iter=1000)
    set_initial_params(model=model)
    client = LogisticRegressionClient(args.cid, model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()






