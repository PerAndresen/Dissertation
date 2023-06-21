import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Load dataset

df = pd.read_csv('datasets/mitbih_test.csv', header=None, delimiter=',',nrows=10000) 
print(df.shape)
print(df.head())


