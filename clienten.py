import argparse
import warnings
import flwr as fl
import pandas as pd
import numpy as np
import sklearn
import utils
import tensorflow as tf
import torch
from flwr.common import (NDArrays,GetParametersRes, FitIns, EvaluateRes, EvaluateIns, FitRes, Scalar)
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

def split_data(X,y, num_clients, client_id):
    data_size = len(X)
    idxs = np.array(range(data_size))
    assert data_size % num_clients == 0
    idxs_splits = np.array_split(idxs, num_clients)
    client_idxs = idxs_splits[client_id]
    return X[client_idxs], y[client_idxs]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=str,required=True)
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_clients", type=int, required=True)
    args = parser.parse_args()
    cid = args.cid

    (X_train, y_train), (X_test, y_test) = utils.load_data()
    X_train, y_train = split_data(X_train, y_train, num_clients, cid)
    model = LogisticRegression(random_state=0, max_iter=1, penalty="l2", warm_start=True)
    utils.set_initial_params(model)




    class LogisticRegressionClient(fl.client.NumPyClient):

        def get_parameters(self,config):
            print("Client "+str(cid)+" is getting parameters")
            return utils.get_model_parameters(model)
    
        def fit(self,parameters,config):
            print("Client "+str(cid)+" is training")
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                print("Training finished for round"+str(config["rnd"]))
            return utils.get_model_parameters(model), len(X_train), {}
    
        def evaluate(self,parameters,config):
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
        
    client = LogisticRegressionClient()
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


    """Load data, create and start client."""
   
    #parser.add_argument("--dataset", type=str)
    #parser.add_argument("--model", type=str, default= "LogisticRegression", help="Model to use for training")



