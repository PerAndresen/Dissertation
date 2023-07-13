import argparse
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import utils
import tensorflow as tf
import torch
import torchvision
from typing import Callable, Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, log_loss
from flwr.common import Scalar, NDArrays
import socket

def get_server_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

server_ip = get_server_ip()
print("Server IP: "+str(server_ip))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('--rounds', type=int, default=20)
parser.add_argument('--clients', type=int, default=2)
parser.add_argument('--min_sample_size', type=int, default=2)
parser.add_argument('--sample_fraction', type=float, default=1.0)
parser.add_argument('--server_address', type=str, default="0.0.0.0:8080")
args = parser.parse_args()

def fit_round(rnd: int)-> Dict:
    return {"rnd": rnd}

def get_eval_fn(model: LogisticRegression):
    """Return evaluation function for server."""
    _, (X_test, y_test) = utils.load_data()
    def evaluate(server_round, parameters: NDArrays, config):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate

def main() -> None:
    """Create model and create and start server."""
    print(args)
    model = LogisticRegression(random_state=0, max_iter=1000)
    utils.set_initial_params(model)
    #params = get_model_parameters(model)
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
       # fraction_fit=args.sample_fraction,
       # min_fit_clients=args.min_sample_size,
        min_available_clients=2,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        #initial_parameters=fl.common.ndarrays_to_parameters(params)
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        #server_address=server_ip+":8080",
        server_address=args.server_address, 
        server=server,
        config=fl.server.ServerConfig(num_rounds=args.rounds))

if __name__ == "__main__":
    main()