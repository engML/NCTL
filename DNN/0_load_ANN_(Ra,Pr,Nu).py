"""
Loading a metamodel for the free convection problem in a square enclosure.
The input variables are Rayleigh and Prandtl numbers.
The output variable is the Nusselt number at the vertical walls, which in this case are the same.
"""
# import modules
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import winsound
import time
start_time = time.time()
tf.get_logger().setLevel('ERROR')
rnd_seed = 1
tf.random.set_seed(rnd_seed)
np.random.seed(rnd_seed)

revised = False # True if new train data is applied

tl_1 = '(Ra,Nu)(Pr,Nu)'
tl_2 = '(Ra,Pr,Nu)'
tl = tl_2

# load input data
datasets_folder = 'Datasets'
data_folder = tl
file_validation = 'Validation' + tl + '.csv'
file_test = 'Test' + tl + '.csv'
if revised: tl += '_revised'
file_train = 'Train' + tl + '.csv'

dir_train = os.path.join(datasets_folder, data_folder, file_train)
dir_val = os.path.join(datasets_folder, data_folder, file_validation)
dir_test = os.path.join(datasets_folder, data_folder, file_test)

dataframe_train = pd.read_csv(dir_train, delim_whitespace=True, header=None)
dataframe_test = pd.read_csv(dir_test, delim_whitespace=True, header=None)
dataframe_val = pd.read_csv(dir_val, delim_whitespace=True, header=None)

data_train = dataframe_train.values
data_val_1 = dataframe_val.values
data_test = dataframe_test.values

# because the relationship between Nusselt and input variables follow a power function, we use the log transform to help the ANN training to be more smoothly done:
data_train = np.log10(data_train)
data_val_1 = np.log10(data_val_1)
data_test = np.log10(data_test)

# split the dataset into train and validation (to consider more validation data points)
data_train_all = data_train
data_train, data_val_2 = train_test_split(data_train, test_size=0.15, random_state=rnd_seed)
data_val = np.concatenate((data_val_1,data_val_2))

# separate the input and output variables: 
Ra_train = data_train[:,0].reshape(-1, 1) # Rayleigh
Pr_train = data_train[:,1].reshape(-1, 1) # Prandtl
Ra_val = data_val[:,0].reshape(-1, 1) # Rayleigh
Pr_val = data_val[:,1].reshape(-1, 1) # Prandtl
Ra_test = data_test[:,0].reshape(-1, 1) # Rayleigh
Pr_test = data_test[:,1].reshape(-1, 1) # Prandtl
Y_train = data_train[:,2].reshape(-1, 1)   # Nusselt
Y_val = data_val[:,2].reshape(-1, 1)   # Nusselt
Y_test = data_test[:,2].reshape(-1, 1)   # Nusselt

# separate the input and output variables:
X_train = [Ra_train, Pr_train]
X_val = [Ra_val, Pr_val]
X_test = [Ra_test, Pr_test]

# evaluation of the model
def evaluation(X_test, Y_test, data):
    result = model.evaluate(X_test, Y_test, verbose=0)
    print('--- the {} loss is {}' .format(str(data), str(result)))
    Y_ANN = model.predict(X_test, verbose=0)
    rel_diff = ((10**Y_ANN) / (10**Y_test) - 1.)*100.
    mean = np.mean(np.abs(rel_diff))
    std_error = np.std(np.abs(rel_diff))
    print('--- mean relative error: {:.2f}%, standard deviation of relative error: {:.2f}%' .format(mean, std_error))
    return Y_ANN

# plot the results
def scatterer(Y_test, Y_ANN, output_folder, data):
    test_Y = 10**(Y_test)
    ANN_Y = 10**(Y_ANN)
    a = plt.axes(aspect='equal')
    plt.scatter(test_Y, ANN_Y, 1, c="b", alpha=0.5, marker=r'$\clubsuit$')
    plt.xlabel('Simulation Results')
    plt.ylabel('ANN Predictions')
    plt.title('ANN Test '+str(data))
    l_max = np.amax(test_Y)
    lims = [0.9, l_max]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims, color='black', linewidth=0.5)
    plot_file = str(data) + '_scatter.pdf'
    plot_dir = os.path.join(output_folder,plot_file)
    plt.savefig(plot_dir)
    plt.close()
    scatter_file = str(data) + '_scatter.txt'
    scatter_dir = os.path.join(output_folder, scatter_file)
    np.savetxt(scatter_dir, np.c_[test_Y,ANN_Y], delimiter=" ")
    return

# recording the history of the training
def history_recorder(history, output_folder):
    train_loss = np.array(history.history['mean_absolute_error'])
    test_loss = np.array(history.history['mean_absolute_error'])
    plt.plot(history.history['mean_absolute_error'], color='black', linestyle='solid', label='train')
    plt.plot(history.history['mean_absolute_error'], color='black', linestyle='dashed', label='test')
    plt.legend()
    l_max = np.amax(test_loss)
    plt.ylim([0,l_max])
    plt.xlabel('Epoch number')
    plt.ylabel('Mean absolute error')
    ploth_file = 'loss_history.pdf'
    ploth_dir = os.path.join(output_folder,ploth_file)
    plt.savefig(ploth_dir)
    plt.close()
    hist_file = 'loss_history.txt'
    hist_dir = os.path.join(output_folder,hist_file)
    np.savetxt(hist_dir, np.c_[train_loss,test_loss], delimiter=" ")
    return

# directories
opt_folder = 'Brute-Force Optimization'
if not os.path.exists(opt_folder): os.makedirs(opt_folder)
modeling_folder = tl
modeling_dir = os.path.join(opt_folder,modeling_folder)
if not os.path.exists(modeling_dir): os.makedirs(modeling_dir)
result_folder = 'Results'
if not os.path.exists(result_folder): os.makedirs(result_folder)
metamodels_folder = 'Metamodels'
metamodels_dir = os.path.join(result_folder,metamodels_folder)
if not os.path.exists(metamodels_dir): os.makedirs(metamodels_dir)
ANN_folder = tl
ANN_dir = os.path.join(metamodels_dir,ANN_folder)
if not os.path.exists(ANN_dir): os.makedirs(ANN_dir)
model_name = 'ANN_model.h5'
best_model_dir = os.path.join(ANN_dir,model_name)
log_name = 'ANN_info.txt'
last_log_dir = os.path.join(ANN_dir,log_name)

# here we have the best model that we trained:
sys.stdout = open(last_log_dir, "w")
model = load_model(best_model_dir)
model.summary()
schm = os.path.join(ANN_dir,'model_plot.png')
plot_model(model, to_file=schm, show_shapes=True, show_layer_names=True, expand_nested=False, rankdir='TB', dpi=300)

Y_ANN = evaluation(X_test, Y_test, 'Test')
scatterer(Y_test, Y_ANN, ANN_dir, 'Test')    
print('\n Checking the train dataset through the final model:')
# we check our training dataset to see how much accurate are our multi-model data:
# (one can then remove parts of the data from the multi-model dataset that have high errors to see if the training can be improved.)
Y_ANN_check = evaluation(X_train, Y_train, 'Train')
scatterer(Y_train, Y_ANN_check, ANN_dir, 'Train')
Y_ANN_check = evaluation(X_val, Y_val, 'Validation')
scatterer(Y_val, Y_ANN_check, ANN_dir, 'Validation')
Ra_train_all = data_train_all[:,0].reshape(-1, 1) # (Rayleigh, Prandtl)
Pr_train_all = data_train_all[:,1].reshape(-1, 1) # (Rayleigh, Prandtl)
Y_train_all = data_train_all[:,2].reshape(-1, 1)   # Nusselt
Y_ANN_check = evaluation([Ra_train_all,Pr_train_all], Y_train_all, 'Train_all')
scatterer(Y_train_all, Y_ANN_check, ANN_dir, 'Train_all')
print('The total runtime was {:.2f} minutes.' .format((time.time() - start_time)/60.))
winsound.Beep(2500, 1000)
