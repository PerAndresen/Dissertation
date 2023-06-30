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
from client2 import load_data, set_model_params,set_initial_params, get_model_parameters
from sklearn.metrics import accuracy_score, log_loss
from flwr.common import Scalar, NDArrays


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('--rounds', type=int, default=1)
parser.add_argument('--clients', type=int, default=2)
parser.add_argument('--min_sample_size', type=int, default=2)
parser.add_argument('--sample_fraction', type=float, default=1.0)
args = parser.parse_args()

def main() -> None:
    """Create model and create and start server."""
    print(args)
    model = LogisticRegression(random_state=0, max_iter=1000)
    set_initial_params(model)
    params = get_model_parameters(model)
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
       # fraction_fit=args.sample_fraction,
       # min_fit_clients=args.min_sample_size,
        min_available_clients=2,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        initial_parameters=fl.common.ndarrays_to_parameters(params)
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        server_address="172.23.249.207:8080", 
        server=server,
        config=fl.server.ServerConfig(num_rounds=args.rounds))

def fit_round(rnd: int)-> Dict:
    return {"rnd": rnd}

'''
def fit_config(rnd: int) -> dict:
    """Return dict of fitting configuration."""
    config = {
        "epochs": 1,
        "batch_size": 32,
        "number_of_workers": 1,
        "pin_memory": False,
    }
    return config
    
def set_weights(model, weights):
    """Set model weights."""
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.from_numpy(weights[k])
    model.load_state_dict(state_dict)

'''

def get_eval_fn(model: LogisticRegression):
    """Return evaluation function for server."""
    _, (X_test, y_test) = load_data()
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        set_model_params(model,params=parameters)
        """Evaluate using given weights."""
        #model.fit(X_train, y_train)
        loss = log_loss(y_test, model.predict(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate

if __name__ == "__main__":
    main()