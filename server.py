import flwr as fl
import argparse
import utils
from sklearn.metrics import log_loss,accuracy_score
from sklearn.linear_model import LogisticRegression
from typing import Dict
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Callable, Dict, Union
import numpy as np
import pandas as pd


XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def set_model_params(model: LogisticRegression, params: LogRegParams)->LogisticRegression:
    #Set the weights of the model
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

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

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return  an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        model_with_parameters = set_model_params(model, parameters)
        
        unique_classes = np.unique(y)
        model_with_parameters.classes_ = unique_classes
        # Evaluate the updated model on the test data
        y_pred = model_with_parameters.predict(X_test)
        model_with_parameters.predict_proba(X_test)
        print("predicitions", y_pred)
        loss = log_loss(y_test, y_pred)
        sk_accuracy = accuracy_score(y_test, y_pred)
        accuracy = model.score(X_test, y_pred)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--server_address', type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )