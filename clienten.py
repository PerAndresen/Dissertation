import argparse
import warnings
import flwr as fl
import pandas as pd
import numpy as np
import sklearn
from utils import *
import tensorflow as tf
import torch
from flwr.common import (NDArrays,GetParametersRes, FitIns, EvaluateRes, EvaluateIns, FitRes, Scalar)
import timeit
from typing import Tuple, List, Optional, Callable, Dict, Union
from logging import INFO, ERROR, WARNING
from flwr.common.logger import log
from pathlib import Path



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
weight = 10

def split_data(X,y, num_clients, client_id):
    data_size = len(X)
    size_per_client = data_size // num_clients
    remainder = data_size % num_clients
    start = client_id * size_per_client
    end = start + size_per_client

    if client_id < remainder:
        end += 1
    
    return X[start:end], y[start:end]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=int,required=True)
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_clients", type=int, required=True)
    args = parser.parse_args()
    cid = args.cid

    (X_train, y_train), (X_test, y_test) = load_data()
    X_train, y_train = split_data(X_train, y_train, num_clients, cid)
    model = LogisticRegression(random_state=0, max_iter=1, penalty="l2", warm_start=True)
    set_initial_params(model)




    class LogisticRegressionClient(fl.client.NumPyClient):
        def __init__(self,cid):
            self.cid = cid
            self.cache_pth = Path('cache'+str(cid)+'.pth')


        def get_parameters(self,config):
            print("Client "+str(cid)+" is getting parameters")
            return get_model_parameters(model)
    
        def fit(self,parameters,config):
            self.reload()
            stage = config.pop('stage')
            ret = 0
            ndarrays = []
            print("Client "+str(cid)+" is training")
            if stage == 0:
                ret = setup_param(self,config)
                #set_model_params(model, parameters)
            elif stage == 1:
                ret = share_keys(self, load_content(config))
            elif stage == 2:
                packet_lst, fit_ins = load_content(config)
                ndarrays = ask_vectors(self, packet_lst, fit_ins)
            elif stage == 3:
                available_clients, dropout_clients = load_content(config)
                ret = unmask_vectors(self, available_clients, dropout_clients)
            '''with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                print("Training finished for round"+str(config["rnd"]))'''
            self.cache()
            return ndarrays, 0, save_content(ret,{})
            #return get_model_parameters(model), len(X_train), {}
    
        def evaluate(self,parameters,config):
            set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
        """Helper functions for encryption and arithmetics in Secure Aggregation"""
        """Provided by Heng Pan working for Flower """
        """Demo code for Secure Aggregation"""
        def get_vars(self):
            return vars(self)
        
        def cache(self):
            with open(self.cache_pth, "wb") as f:
                pickle.dump(self.get_vars, f)
        
        def reload(self):
            if self.cache_pth.exists():
                log(INFO, f'CID {self.cid} reloading from {str(self.cache_pth)}')
                with open(self.cache_pth, 'rb') as f:
                    self.__dict__.update(pickle.load(f))


"""Helper functions for encryption and arithmetics in Secure Aggregation"""
"""Provided by Heng Pan working for Flower """
"""Demo code for Secure Aggregation"""

def setup_param(client, setup_param_dict: Dict[str, Scalar]) -> Tuple[bytes, bytes]:
    # Assigning parameter values to object fields
    sec_agg_param_dict = setup_param_dict
    client.sample_num = sec_agg_param_dict['share_num']
    client.sec_id = sec_agg_param_dict['id']
    client.sec_agg_id = sec_agg_param_dict['id']
    log(INFO, f'Client {client.sec_agg_id}: starting stage 0...')

    client.share_num = sec_agg_param_dict['share_num']
    client.threshold = sec_agg_param_dict['threshold']
    client.drop_flag = sec_agg_param_dict['drop_flag']
    client.clipping_range = 3
    client.target_range = 1 << 16
    client.mod_range = 1 << 24

    # key is the sec_agg_id of another client (int)
    # value is the secret share we possess that contributes to the client's secret (bytes)
    client.b_share_dict = {}
    client.sk1_share_dict = {}
    client.shared_key_2_dict = {}
    return ask_keys(client)


def ask_keys(client) -> Tuple[bytes, bytes]:
    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    client.sk1, client.pk1 = generate_key_pairs()
    client.sk2, client.pk2 = generate_key_pairs()

    client.sk1, client.pk1 = private_key_to_bytes(client.sk1), public_key_to_bytes(client.pk1)
    client.sk2, client.pk2 = private_key_to_bytes(client.sk2), public_key_to_bytes(client.pk2)
    log(INFO, f'Client {client.sec_agg_id}: stage 0 completes. uploading public keys...')
    return client.pk1, client.pk2


def share_keys(client, share_keys_dict: Dict[int, Tuple[bytes, bytes]]) -> List[ShareKeysPacket]:
    log(INFO, f'Client {client.sec_agg_id}: starting stage 1...')
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    client.public_keys_dict = share_keys_dict
    # check size is larger than threshold
    if len(client.public_keys_dict) < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # check if all public keys received are unique
    pk_list: List[bytes] = []
    for i in client.public_keys_dict.values():
        pk_list.append(i[0])
        pk_list.append(i[1])
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # sanity check that own public keys are correct in dict
    if client.public_keys_dict[client.sec_agg_id][0] != client.pk1 or \
       client.public_keys_dict[client.sec_agg_id][1] != client.pk2:
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!")

    # Generate private mask seed
    client.b = rand_bytes(32)

    # Create shares
    b_shares = create_shares(
        client.b, client.threshold, client.share_num
    )
    sk1_shares = create_shares(
        client.sk1, client.threshold, client.share_num
    )

    share_keys_res_list = []

    for idx, p in enumerate(client.public_keys_dict.items()):
        client_sec_agg_id, client_public_keys = p
        if client_sec_agg_id == client.sec_agg_id:
            client.b_share_dict[client.sec_agg_id] = b_shares[idx]
            client.sk1_share_dict[client.sec_agg_id] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                bytes_to_private_key(client.sk2), bytes_to_public_key(client_public_keys[1]))
            client.shared_key_2_dict[client_sec_agg_id] = shared_key
            plaintext = share_keys_plaintext_concat(
                client.sec_agg_id, client_sec_agg_id, b_shares[idx], sk1_shares[idx])
            ciphertext = encrypt(shared_key, plaintext)
            share_keys_packet = ShareKeysPacket(
                source=client.sec_agg_id, destination=client_sec_agg_id, ciphertext=ciphertext)
            share_keys_res_list.append(share_keys_packet)

    log(INFO, f'Client {client.sec_agg_id}: stage 1 completes. uploading key shares...')
    return share_keys_res_list


def ask_vectors(client, packet_list, fit_ins) -> Parameters:
    log(INFO, f'Client {client.sec_agg_id}: starting stage 2...')
    # Receive shares and fit model
    available_clients: List[int] = []

    if len(packet_list)+1 < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # decode all packets and verify all packets are valid. Save shares received
    for packet in packet_list:
        source = packet.source
        available_clients.append(source)
        destination = packet.destination
        ciphertext = packet.ciphertext
        if destination != client.sec_agg_id:
            raise Exception(
                "Received packet meant for another user. Not supposed to happen")
        shared_key = client.shared_key_2_dict[source]
        plaintext = decrypt(shared_key, ciphertext)
        try:
            plaintext_source, plaintext_destination, plaintext_b_share, plaintext_sk1_share = \
                share_keys_plaintext_separate(plaintext)
        except:
            raise Exception(
                "Decryption of ciphertext failed. Not supposed to happen")
        if plaintext_source != source:
            raise Exception(
                "Received packet source is different from intended source. Not supposed to happen")
        if plaintext_destination != destination:
            raise Exception(
                "Received packet destination is different from intended destination. Not supposed to happen")
        client.b_share_dict[source] = plaintext_b_share
        client.sk1_share_dict[source] = plaintext_sk1_share

    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    
    with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                print("Training finished for round"+str(fit_ins["rnd"]))
    
    parameters = get_model_parameters(model)
    fit_res = FitRes(parameters, len(X_train), {})
    weights_factor = fit_res.num_examples
    

    if client.drop_flag:
        # log(ERROR, "Force dropout due to testing!!")
        raise Exception("Force dropout due to testing")
    weights = get_model_parameters(model)
    weights_factor = weight

    # Quantize weight update vector
    quantized_weights = quantize(weights, client.clipping_range, client.target_range)

    quantized_weights = weights_multiply(quantized_weights, weights_factor)
    quantized_weights = factor_weights_combine(weights_factor, quantized_weights)

    dimensions_list: List[Tuple] = [a.shape for a in quantized_weights]

    # add private mask
    private_mask = pseudo_rand_gen(client.b, client.mod_range, dimensions_list)
    quantized_weights = weights_addition(quantized_weights, private_mask)

    for client_id in available_clients:
        # add pairwise mask
        shared_key = generate_shared_key(bytes_to_private_key(client.sk1),
                                         bytes_to_public_key(client.public_keys_dict[client_id][0]))
        # print('shared key length: %d' % len(shared_key))
        pairwise_mask = pseudo_rand_gen(shared_key, client.mod_range, dimensions_list)
        if client.sec_agg_id > client_id:
            quantized_weights = weights_addition(quantized_weights, pairwise_mask)
        else:
            quantized_weights = weights_subtraction(quantized_weights, pairwise_mask)

    # Take mod of final weight update vector and return to server
    quantized_weights = weights_mod(quantized_weights, client.mod_range)
    # return ndarrays_to_parameters(quantized_weights)
    log(INFO, f'Client {client.sec_agg_id}: stage 2 completes. uploading masked weights...')
    return quantized_weights


def unmask_vectors(client, available_clients, dropout_clients) -> Dict[int, bytes]:
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(available_clients) < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")
    share_dict: Dict[int, bytes] = {}
    for idx in available_clients:
        share_dict[idx] = client.b_share_dict[idx]
    for idx in dropout_clients:
        share_dict[idx] = client.sk1_share_dict[idx]
    return share_dict


client = LogisticRegressionClient(args.cid)
fl.client.start_numpy_client(server_address=args.server_address, client=client)
