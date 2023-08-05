from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from typing import Any, Tuple, Union, List
import json
import base64

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    #Get the weights of the model
    if not hasattr(model, "coef_"):
        raise ValueError("Cannot get parameters of untrained model")
    if model.fit_intercept:
        params = [model.coef_, 
                  model.intercept_
                  ]
    else:
        params = [model.coef_]
    return params

def set_model_params(model: LogisticRegression, params: LogRegParams)->LogisticRegression:
    #Set the weights of the model
    #print("coef",coef)
    #print(type(coef))
    #print("inter",inter)
    #print(type(inter))
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    #print(params[2])
    #print(type(params[2]))
    #print(params[2].shape)
    #model.classes_ = params[2]
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
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #trainset = (X_train,y_train)
    #testset = (X_test, y_test)
    return X, y

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


"""Helper functions for encryption and arithmetics in Secure Aggregation"""
"""Provided by Heng Pan working for Flower """
"""Demo code for Secure Aggregation"""
import base64
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.SecretSharing import Shamir
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from dataclasses import dataclass

from flwr.common import FitIns, Parameters, ndarrays_to_parameters, Scalar
import numpy as np
import os
import pickle
from typing import Optional, Dict, List, Tuple

"""
    These functions are for demonstration purposes only.

    In practice, one should define and create new message types to facilitate communications in SA,
    since adversaries can exploit pickle.loads to conduct attacks.
"""


def empty_parameters():
    return ndarrays_to_parameters([])


def save_content(content, d: Dict[str, Scalar]) -> Dict[str, Scalar]:
    #print("save content dict: "+ str(d.values()))
    d['content'] = pickle.dumps(content)
    return d


def load_content(d: Dict[str, Scalar]):
    #print("load content dict: "+ str(d.values()))
    if 'content' in d:
        return pickle.loads(d.pop('content'))
    else:
        raise KeyError("No content in the dictionary")

'''
def save_content(content, d: Dict[str, Scalar]) -> Dict[str, Scalar]:
    print("Type of content: ", type(content))
    print("Content: ", content)
    if isinstance(content, bytes):
        content = base64.b64encode(content).decode('utf-8')
        d['content'] = json.dumps(content)
    else:
        serialized_keys = []
        for key in content:
            if isinstance(key, int):
                # Convert the integer to bytes and then base64 encode
                serialized_key = base64.b64encode(key.to_bytes(32, byteorder='big')).decode()
            else:
                # Assuming key is a bytes-like object
                serialized_key = base64.b64encode(key).decode()
            serialized_keys.append(serialized_key)
        d['content'] = json.dumps(serialized_keys)
    return d

'''



'''
def load_content(d: Dict[str, Scalar]) -> Dict[str, Any]:
    print("dict: "+ str(d.values()))
    content_base64 = d.pop('content')
    #return content_base64
    print("Type of content_base64: ", type(content_base64))
    print("Content_base64: ", content_base64)
    content_json = base64.b64decode(content_base64).decode('utf-8')
    print("Type of content_json: ", type(content_json))
    print("Content_json: ", content_json)
    #content = json.loads(content_json)
    content = base64.b64decode(content_base64).decode('utf-8') if isinstance(content_base64, str) else content_base64
    print("content after json.loads: ", content)
    d['content'] = content
    return d

'''

def build_fit_ins(content, stage: int, server_round: int, parameters: Optional[Parameters] = None) -> FitIns:
    cfg = save_content(content, {
        'server_rnd': server_round,
        'stage': stage
    })
    return FitIns(
        parameters=parameters if parameters is not None else empty_parameters(),
        config=cfg
    )


"""
    Encryption
"""


# Key Generation  ====================================================================

# Generate private and public key pairs with Cryptography
def generate_key_pairs() -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    sk = ec.generate_private_key(ec.SECP384R1())
    pk = sk.public_key()
    return sk, pk


# Serialize private key
def private_key_to_bytes(sk: ec.EllipticCurvePrivateKey) -> bytes:
    return sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


# Deserialize private key
def bytes_to_private_key(b: bytes) -> ec.EllipticCurvePrivateKey:
    return serialization.load_pem_private_key(data=b, password=None)


# Serialize public key
def public_key_to_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
    return pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


# Deserialize public key
def bytes_to_public_key(b: bytes) -> ec.EllipticCurvePublicKey:
    return serialization.load_pem_public_key(data=b)


# Generate shared key by exchange function and key derivation function
# Key derivation function is needed to obtain final shared key of exactly 32 bytes
def generate_shared_key(
        sk: ec.EllipticCurvePrivateKey, pk: ec.EllipticCurvePublicKey
) -> bytes:
    # Generate a 32 byte urlsafe(for fernet) shared key from own private key and another public key
    sharedk = sk.exchange(ec.ECDH(), pk)
    derivedk = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
    ).derive(sharedk)
    return base64.urlsafe_b64encode(derivedk)


# Authenticated Encryption ================================================================

# Encrypt plaintext with Fernet. Key must be 32 bytes.
def encrypt(key: bytes, plaintext: bytes) -> bytes:
    # key must be url safe
    f = Fernet(key)
    return f.encrypt(plaintext)


# Decrypt ciphertext with Fernet. Key must be 32 bytes.
def decrypt(key: bytes, token: bytes):
    # key must be url safe
    f = Fernet(key)
    return f.decrypt(token)


# Random Bytes Generator =============================================================

# Generate random bytes with os. Usually 32 bytes for Fernet
def rand_bytes(num: int = 32) -> bytes:
    return os.urandom(num)


"""
    Arithmetics
"""


# Combine factor with weights
def factor_weights_combine(weights_factor: int, weights: List[np.ndarray]) -> List[np.ndarray]:
    return [np.array([weights_factor])]+weights


# Extract factor from weights
def factor_weights_extract(weights: List[np.ndarray]) -> Tuple[int, List[np.ndarray]]:
    return weights[0][0], weights[1:]


# Create dimensions list of each element in weights
def weights_shape(weights: List[np.ndarray]) -> List[Tuple]:
    return [arr.shape for arr in weights]


# Generate zero weights based on dimensions list
def weights_zero_generate(dimensions_list: List[Tuple], dtype=np.int64) -> List[np.ndarray]:
    return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]


# Add two weights together
def weights_addition(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [a[idx]+b[idx] for idx in range(len(a))]


# Subtract one weight from the other
def weights_subtraction(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [a[idx]-b[idx] for idx in range(len(a))]


# Take mod of a weights with an integer
def weights_mod(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    if bin(b).count("1") == 1:
        msk = b - 1
        return [a[idx] & msk for idx in range(len(a))]
    return [a[idx] % b for idx in range(len(a))]


# Multiply weight by an integer
def weights_multiply(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    return [a[idx] * b for idx in range(len(a))]


# Divide weight by an integer
def weights_divide(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    return [a[idx] / b for idx in range(len(a))]


"""
    Quantization
"""


def stochastic_round(arr: np.ndarray):
    ret = np.ceil(arr).astype(np.int32)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1
    return ret


def quantize(weight: List[np.ndarray], clipping_range: float, target_range: int) -> List[np.ndarray]:
    quantized_list = []
    quantizer = target_range / (2 * clipping_range)
    for arr in weight:
        # stochastic quantization
        quantized = (np.clip(arr, -clipping_range, clipping_range) + clipping_range) * quantizer
        quantized = stochastic_round(quantized)
        quantized_list.append(quantized)
    return quantized_list


# Transform weight vector to range [-clipping_range, clipping_range]
# Convert to float
def reverse_quantize(weight: List[np.ndarray], clipping_range: float, target_range: int) -> List[np.ndarray]:
    reverse_quantized_list = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range
    for arr in weight:
        arr = arr.view(np.ndarray).astype(float) * quantizer + shift
        reverse_quantized_list.append(arr)
    return reverse_quantized_list


"""
    Shamir's secret sharing
"""


# Create shares with PyCryptodome. Each share must be processed to be a byte string with pickle for RPC
def create_shares(
        secret: bytes, threshold: int, num: int
) -> List[bytes]:
    # return list of list for each user. Each sublist contains a share for a 16 byte chunk of the secret.
    # The int part of the tuple represents the index of the share, not the index of the chunk it is representing.
    secret_padded = pad(secret, 16)
    secret_padded_chunk = [
        (threshold, num, secret_padded[i: i + 16])
        for i in range(0, len(secret_padded), 16)
    ]
    share_list = []
    for i in range(num):
        share_list.append([])

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk_shares in executor.map(
                lambda arg: shamir_split(*arg), secret_padded_chunk
        ):
            for idx, share in chunk_shares:
                # idx start with 1
                share_list[idx - 1].append((idx, share))

    for idx, shares in enumerate(share_list):
        share_list[idx] = pickle.dumps(shares)
    # print("send", [len(i) for i in share_list])

    return share_list


def shamir_split(threshold: int, num: int, chunk: bytes) -> List[Tuple[int, bytes]]:
    return Shamir.split(threshold, num, chunk)


# Reconstructing secret with PyCryptodome
def combine_shares(share_list: List[bytes]) -> bytes:
    # print("receive", [len(i) for i in share_list])
    for idx, share in enumerate(share_list):
        share_list[idx] = pickle.loads(share)

    chunk_num = len(share_list[0])
    secret_padded = bytearray(0)
    chunk_shares_list = []
    for i in range(chunk_num):
        chunk_shares = []
        for j in range(len(share_list)):
            chunk_shares.append(share_list[j][i])
        chunk_shares_list.append(chunk_shares)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in executor.map(shamir_combine, chunk_shares_list):
            secret_padded += chunk

    secret = unpad(secret_padded, 16)
    return bytes(secret)


def shamir_combine(shares: List[Tuple[int, bytes]]) -> bytes:
    return Shamir.combine(shares)


"""
    Miscellaneous
"""


# Unambiguous string concatenation of source, destination, and two secret shares.
# We assume they do not contain the 'abcdef' string
def share_keys_plaintext_concat(source: int, destination: int, b_share: bytes, sk_share: bytes) -> bytes:
    source, destination = int.to_bytes(source, 4, 'little'), int.to_bytes(destination, 4, 'little')
    return b''.join([source, destination, int.to_bytes(len(b_share), 4, 'little'), b_share, sk_share])


# Unambiguous string splitting to obtain source, destination and two secret shares.


def share_keys_plaintext_separate(plaintext: bytes) -> Tuple[int, int, bytes, bytes]:
    src, dst, mark = int.from_bytes(plaintext[:4], 'little'), int.from_bytes(plaintext[4:8], 'little'), \
                     int.from_bytes(plaintext[8:12], 'little')
    ret = [src, dst, plaintext[12:12 + mark], plaintext[12 + mark:]]
    return ret


# Pseudo Bytes Generator ==============================================================

# Pseudo random generator for creating masks.
# the one use numpy PRG
def pseudo_rand_gen(seed: bytes, num_range: int, dimensions_list: List[Tuple]) -> List[np.ndarray]:
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i:i + 4], 'little')
    np.random.seed(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(np.random.randint(0, num_range - 1), dtype=int)
        else:
            arr = np.random.randint(0, num_range - 1, dimension)
        output.append(arr)
    return output


@dataclass
class ShareKeysPacket:
    source: int
    destination: int
    ciphertext: bytes
