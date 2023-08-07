import argparse
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import * 
import tensorflow as tf
import torch
import torchvision
from typing import Callable, Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, log_loss
from flwr.common import Scalar, NDArrays
import socket
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

class SecureAggregationStrategy(Strategy):

    def __init__(self, num_samples, threshold, num_dropouts):
        """
        @param num_samples: number of sampled clients per round
        @param threshold:  minimum number of surviving clients
        @param num_dropouts: number of dropouts
        """
        self.sample_num = num_samples
        self.threshold = threshold
        self.dropout_num = num_dropouts

        # runtime variables
        self.proxy2id = {}
        self.stage = 0
        self.surviving_clients = {}
        self.public_keys_dict = {}
        self.forward_packet_list_dict = {}
        self.masked_vector = []
        self.dropout_clients = {}

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters."""
        return set_initial_params(model=LogisticRegression(random_state=0, max_iter=1000))

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {
            'server_rnd': server_round,
            'stage': self.stage
        }
        if self.stage == 0:
            config['share_num'] = self.sample_num
            config['threshold'] = self.threshold
            self.proxy2id = {}
            self.surviving_clients = {}
            # Sample clients
            clients = client_manager.sample(
                num_clients=self.sample_num, min_num_clients=self.sample_num
            )
            ret = []
            for idx, client in enumerate(clients):
                self.proxy2id[client] = idx
                cfg = config.copy()
                cfg['id'] = idx
                cfg['drop_flag'] = idx < self.dropout_num
                ret += [(client, FitIns(empty_parameters(), cfg))]

            # Return client/config pairs
            return ret
        if self.stage == 1:
            save_content(self.public_keys_dict, config)
            fit_ins = FitIns(empty_parameters(), config)
            return [(client, fit_ins) for client in self.surviving_clients.values()]
        if self.stage == 2:
            # Fit Instructions here
            fit_ins_lst = [FitIns(empty_parameters(), {})] * self.sample_num

            ret_lst = []
            for idx, client in self.surviving_clients.items():
                assert idx == self.proxy2id[client]
                ret_lst.append((client, FitIns(
                    empty_parameters(),
                    save_content((
                        self.forward_packet_list_dict[idx],
                        fit_ins_lst[idx]
                    ), config.copy())
                )))
            return ret_lst
        if self.stage == 3:
            save_content(
                (list(self.surviving_clients.keys()), list(self.dropout_clients.keys())),
                config
            )
            fit_ins = FitIns(empty_parameters(), config)
            return [(client, fit_ins) for client in self.surviving_clients.values()]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if self.stage == 0:
            public_keys_dict: Dict[int, Tuple[bytes, bytes]] = {}
            ask_keys_results = results
            if len(ask_keys_results) < self.threshold:
                raise Exception("Not enough available clients after ask keys stage")
            share_keys_clients: Dict[int, ClientProxy] = {}

            # Build public keys dict
            # tmp_map = dict(ask_keys_results)
            for client, result in ask_keys_results:
                idx = self.proxy2id[client]
                public_keys_dict[idx] = load_content(result.metrics)
                share_keys_clients[idx] = client
            self.public_keys_dict = public_keys_dict
            self.surviving_clients = share_keys_clients
        elif self.stage == 1:
            # Build forward packet list dictionary
            total_packet_list: List[ShareKeysPacket] = []
            forward_packet_list_dict: Dict[int, List[ShareKeysPacket]] = {}  # destination -> list of packets
            ask_vectors_clients: Dict[int, ClientProxy] = {}
            # tmp_map = dict(share_keys_results)
            for client, fit_res in results:
                idx = self.proxy2id[client]
                result = load_content(fit_res.metrics)
                ask_vectors_clients[idx] = client
                packet_list = result
                total_packet_list += packet_list

            for idx in ask_vectors_clients.keys():
                forward_packet_list_dict[idx] = []
            # forward_packet_list_dict = dict(zip(ask_vectors_clients.keys(), [] * len(ask_vectors_clients.keys())))
            for packet in total_packet_list:
                destination = packet.destination
                if destination in ask_vectors_clients.keys():
                    forward_packet_list_dict[destination].append(packet)
            self.surviving_clients = ask_vectors_clients
            self.forward_packet_list_dict = forward_packet_list_dict
        elif self.stage == 2:
            if len(results) < self.threshold:
                raise Exception("Not enough available clients after ask vectors stage")
            # Get shape of vector sent by first client
            masked_vector = [np.array([0], dtype=int), np.zeros((1,186), dtype=int)]
            # Add all collected masked vectors and compuute available and dropout clients set
            unmask_vectors_clients: Dict[int, ClientProxy] = {}
            dropout_clients = self.surviving_clients.copy()
            for client, fit_res in results:
                idx = self.proxy2id[client]
                unmask_vectors_clients[idx] = client
                dropout_clients.pop(idx)
                client_parameters = fit_res.parameters

                #print("Shape of client_parameters:", parameters_to_ndarrays(client_parameters)[1].shape)
                masked_vector = weights_addition(masked_vector, parameters_to_ndarrays(client_parameters))
                #print("Shape of masked_vector:", masked_vector[1].shape)

            masked_vector = weights_mod(masked_vector, 1 << 24)
            self.masked_vector = masked_vector
            self.surviving_clients = unmask_vectors_clients
            self.dropout_clients = dropout_clients
        elif self.stage == 3:
            # Build collected shares dict
            collected_shares_dict: Dict[int, List[bytes]] = {}
            for idx in self.proxy2id.values():
                collected_shares_dict[idx] = []

            if len(results) < self.threshold:
                raise Exception("Not enough available clients after unmask vectors stage")
            for _, fit_res in results:
                share_dict = load_content(fit_res.metrics)
                for owner_id, share in share_dict.items():
                    collected_shares_dict[owner_id].append(share)
            masked_vector = self.masked_vector
            # Remove mask for every client who is available before ask vectors stage,
            # Divide vector by first element
            for client_id, share_list in collected_shares_dict.items():
                if len(share_list) < self.threshold:
                    raise Exception(
                        "Not enough shares to recover secret in unmask vectors stage")
                secret = combine_shares(share_list)
                if client_id in self.surviving_clients.keys():
                    # seed is an available client's b
                    private_mask = pseudo_rand_gen(
                        secret, 1 << 24, weights_shape(masked_vector))
                    masked_vector = weights_subtraction(masked_vector, private_mask)
                else:
                    # seed is a dropout client's sk1
                    neighbor_list = list(self.proxy2id.values())
                    neighbor_list.remove(client_id)

                    for neighbor_id in neighbor_list:
                        shared_key = generate_shared_key(
                            bytes_to_private_key(secret),
                            bytes_to_public_key(self.public_keys_dict[neighbor_id][0]))
                        pairwise_mask = pseudo_rand_gen(
                            shared_key, 1 << 24, weights_shape(masked_vector))
                        if client_id > neighbor_id:
                            masked_vector = weights_addition(
                                masked_vector, pairwise_mask)
                        else:
                            masked_vector = weights_subtraction(
                                masked_vector, pairwise_mask)
            masked_vector = weights_mod(masked_vector, 1 << 24)
            # Divide vector by number of clients who have given us their masked vector
            # i.e. those participating in final unmask vectors stage
            total_weights_factor, masked_vector = factor_weights_extract(masked_vector)
            masked_vector = weights_divide(masked_vector, total_weights_factor)
            aggregated_vector = reverse_quantize(
                masked_vector, 3, 1 << 16)
            #print(aggregated_vector[:4])
            aggregated_parameters = ndarrays_to_parameters(aggregated_vector)

            self.stage = 0
            return aggregated_parameters, {}

        self.stage = (self.stage + 1) % 4

        return None, {}
    def configure_evaluate(self, server_round, parameters, client_manager):
        '''
        clients = client_manager.sample(num_clients=10, min_num_clients=5)
        eval_ins = [fl.common.EvaluateIns(parameters, {"server_round":server_round})]*len(clients)
        return list(zip(clients, eval_ins))
        '''
        return []

    def aggregate_evaluate(self, server_round, eval_metrics, failures):
        '''print(eval_metrics)
        total_accuracy = 0
        total_examples = 0
        total_loss = 0
        for _, evaluate_res in eval_metrics:
            print("Num examples",evaluate_res.num_examples)
            accuracy = evaluate_res.metrics['accuracy']
            num_examples = evaluate_res.num_examples
            total_accuracy += accuracy *num_examples
            total_examples += num_examples
            total_loss += evaluate_res.loss * num_examples
        if total_examples == 0:
            return None
        aggregated_metric = total_accuracy / total_examples
        aggregated_loss = total_loss /total_examples
        return aggregated_loss,{'accuracy':aggregated_metric}
        """Not aggregating any evaluation.""" '''
        return None

    def evaluate(self, server_round, parameters):
        #eval_metrics = client_manager.evaluate(parameters)
        #aggregated_loss, aggregated_metric = self.aggregate_evaluate(eval_metrics=eval_metrics,failures=None)
        #return aggregated_loss, aggregated_metric
        '''
        results = client_manager.evaluate(parameters)
        
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model = LogisticRegression(random_state=0, max_iter=1000)
        model.fit(X_train, y_train)
        set_model_params(model, parameters)
    
        #model_with_params = get_model_parameters(model)
        print("Parameters on server evaluation:", parameters)

        model_with_params = get_model_parameters(model)
        #model_with_params = set_model_params(model, parameters_to_ndarrays(parameters))
        #model_with_params.fit(X_train, y_train)
        unique_classes = np.unique(y)
        model_with_params.classes_ = unique_classes
        y_pred = model_with_params.predict(X_test)
        loss = log_loss(y_test, y_pred)
        accuracy = model_with_params.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
        '''
        
        """Not running any centralized evaluation."""
        return None

def get_server_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

server_ip = get_server_ip()
print("Server IP: "+str(server_ip))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('--rounds', type=int, default=20)
parser.add_argument('--clients', type=int, default=10)
parser.add_argument('--min_sample_size', type=int, default=2)
parser.add_argument('--sample_fraction', type=float, default=1.0)
parser.add_argument('--server_address', type=str, default="0.0.0.0:8080")
args = parser.parse_args()

def fit_round(rnd: int)-> Dict:
    return {"rnd": rnd}

def get_eval_fn(model: LogisticRegression):
    """Return evaluation function for server."""
    _, (X_test, y_test) = load_data()
    def evaluate(server_round, parameters: NDArrays, config):
        set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate

parameters = {
    'num_clients_per_round': 10,
    'num_total_clients': 100,
    'min_num_surviving_clients': 5,
    'num_dropouts': 3,
    'num_rounds': 4,
    'data_dir': './client_data'
}


def main() -> None:
    """Create model and create and start server."""
    print(args)
    #model = LogisticRegression(random_state=0, max_iter=1000)
    #set_initial_params(model)
    #params = get_model_parameters(model)
    client_manager = fl.server.SimpleClientManager()
    sa_strategy = SecureAggregationStrategy(
        num_samples=args.clients,
        threshold=args.min_sample_size,
        num_dropouts=0,
    )
    '''
    strategy = fl.server.strategy.FedAvg(
        num_samples=args.clients,
        threshold=args.min_sample_size,
        num_dropouts=0,
       # fraction_fit=args.sample_fraction,
       # min_fit_clients=args.min_sample_size,
        min_available_clients=2,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        #initial_parameters=fl.common.ndarrays_to_parameters(params)
    )
    '''
    server = fl.server.Server(client_manager=client_manager, strategy=sa_strategy)
    fl.server.start_server(
        #num_samples=args.clients,
        #server_address=server_ip+":8080",
        server_address=args.server_address, 
        #server=server,
        client_manager = client_manager,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy= sa_strategy
        )

if __name__ == "__main__":
    main()