# import modules
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
rnd_seed = 1
tf.random.set_seed(rnd_seed)
np.random.seed(rnd_seed)

hyperbandit = True  # Bayesian optimization if false

dir_opt = 'ANN Structure Design'
if not os.path.exists(dir_opt): os.makedirs(dir_opt)
dir_prj = 'Hyperband Optimization' if hyperbandit else 'Bayesian Optimization'
output_folder = 'Metamodel Hyperband' if hyperbandit else 'Metamodel Bayesian'
output_dir = os.path.join(dir_opt,dir_prj,output_folder)
if not os.path.exists(output_dir): os.makedirs(output_dir)
log_dir = os.path.join(output_dir,'ANN_info.txt')
#sys.stdout = open(log_dir, "w")

# load input data
data_folder = 'Datasets'
file_train = 'Nu_train_1.csv'
file_test = 'Nu_test.csv'

dir_train = os.path.join(data_folder, file_train)
dir_test = os.path.join(data_folder, file_test)

dataframe_train = pd.read_csv(dir_train, delim_whitespace=True, header=None)
dataframe_test = pd.read_csv(dir_test, delim_whitespace=True, header=None)

data_train = dataframe_train.values
data_test = dataframe_test.values

# because the relationship between Nusselt and input variables follow a power function, we use the log transform to help the ANN training to be more smoothly done:
data_train = np.log10(data_train)
data_test = np.log10(data_test)

# split the dataset into train and validation
data_train, data_val = train_test_split(data_train, test_size=0.15, random_state=rnd_seed)

# separate the input and output variables:
X_train = data_train[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
X_val = data_val[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
Y_train = data_train[:,2].reshape(-1, 1)   # Nusselt
Y_val = data_val[:,2].reshape(-1, 1)   # Nusselt
X_test = data_test[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
Y_test = data_test[:,2].reshape(-1, 1)   # Nusselt

# Compile model
def build_model(hp):
    model = Sequential()
    model.add(Dense(4, input_dim=2, kernel_initializer='normal', activation='tanh'))
    for i in range(hp.Int('hidden_layers_count', 1, 5, default=2)):
        model.add(Dense(hp.Choice('hidden_size', values=[4, 8, 16, 32], default=16), kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=5e-4, max_value=1e-3, sampling='LOG', default=5e-4)),        
        loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# tune model
tuner_1 = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=1000,
    num_initial_points=2,
    directory=dir_opt,
    project_name=dir_prj)

tuner_2 = kt.Hyperband(
    build_model,
    objective='val_loss',
    factor=2,
    max_epochs=1000,
    hyperband_iterations=10,
    directory=dir_opt,
    project_name=dir_prj)

tuner = tuner_2 if hyperbandit else tuner_1
tuner.search(X_train, Y_train, validation_data = (X_val, Y_val), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)], verbose=0)

sys.stdout = open(log_dir, "w")
best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_model.summary()
best_model.save(output_dir)
result = best_model.evaluate(X_test, Y_test, verbose=0)
print('--- test loss:' + str(result))
tuner.results_summary()
