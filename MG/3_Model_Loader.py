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
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
rnd_seed = 1
tf.random.set_seed(rnd_seed)
np.random.seed(rnd_seed)

revised = True # True if loading the revised metamodel

# load input data
data_folder = 'Datasets'
file_train = 'Nu_train_2.csv' if revised else 'Nu_train_1.csv'
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
data_train_all = data_train
data_train, data_val = train_test_split(data_train, test_size=0.15, random_state=rnd_seed)

# separate the input and output variables:
X_train = data_train[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
X_val = data_val[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
X_test = data_test[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
Y_train = data_train[:,2].reshape(-1, 1)   # Nusselt
Y_val = data_val[:,2].reshape(-1, 1)   # Nusselt
Y_test = data_test[:,2].reshape(-1, 1)   # Nusselt

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
    test_Y = 10**Y_test
    ANN_Y = 10**Y_ANN
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
    test_loss = np.array(history.history['val_mean_absolute_error'])
    plt.plot(history.history['mean_absolute_error'], color='black', linestyle='solid', label='train')
    plt.plot(history.history['val_mean_absolute_error'], color='black', linestyle='dashed', label='test')
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
opt_folder = 'ANN Structure Design'
if not os.path.exists(opt_folder): os.makedirs(opt_folder)
modeling_folder = 'Brute-Force Optimization'
modeling_folder += ' 2' if revised else ' 1'
modeling_dir = os.path.join(opt_folder,modeling_folder)
if not os.path.exists(modeling_dir): os.makedirs(modeling_dir)
result_folder = 'Results'
if not os.path.exists(result_folder): os.makedirs(result_folder)
ANN_folder = 'Metamodel'
ANN_folder += ' 2' if revised else ' 1'
ANN_dir = os.path.join(result_folder,ANN_folder)
if not os.path.exists(ANN_dir): os.makedirs(ANN_dir)
model_name = 'ANN_model.h5'
best_model_dir = os.path.join(ANN_dir,model_name)
log_name = 'ANN_info.txt'
last_log_dir = os.path.join(ANN_dir,log_name)

sys.stdout = open(last_log_dir, "w")
# load the model
model = load_model(best_model_dir)
model.summary()
Y_ANN = evaluation(X_test, Y_test, 'Test')
scatterer(Y_test, Y_ANN, ANN_dir, 'Test')    
print('\n Checking the train dataset through the final model:')
# we check our training dataset to see how much accurate are our multi-model data:
# (one can then remove parts of the data from the multi-model dataset that have high errors to see if the training can be improved.)
Y_ANN_check = evaluation(X_train, Y_train, 'Train')
scatterer(Y_train, Y_ANN_check, ANN_dir, 'Train')
split_train_dir = os.path.join(ANN_dir,'train.txt')
np.savetxt(split_train_dir, np.c_[10**X_train, 10**Y_train], delimiter=" ")
Y_ANN_check = evaluation(X_val, Y_val, 'Validation')
scatterer(Y_val, Y_ANN_check, ANN_dir, 'Validation')
split_val_dir = os.path.join(ANN_dir,'validation.txt')
np.savetxt(split_val_dir, np.c_[10**X_val, 10**Y_val], delimiter=" ")

X_train_all = data_train_all[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
Y_train_all = data_train_all[:,2].reshape(-1, 1)   # Nusselt
Y_ANN_check = evaluation(X_train_all, Y_train_all, 'Train_all')
scatterer(Y_train_all, Y_ANN_check, ANN_dir, 'Train_all')
