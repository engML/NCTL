"""
We aimed to extract a metamodel out of a physical model that numerically predicts the natural convection
characteristics in a square enclosure, filled with a Newtonian fluid.
This problem is governed by two parameters: Ra and Pr (see Appendix A for details about the mathematical analysis).
We consider Ra of up to 10^8 and Pr of greater than 0.05 (0<Ra≤10^8 and 0.05≤Pr<∞); however,
lower Pr were also considered provided that the ratio of Ra⁄Pr is at most less than 10^8.
A 400×400 grid system was shown to provide precise results for the average Nu even for the most stringent cases.
Nonetheless, lower grid systems can provide accurate numerical solutions for limited ranges of Ra.
For example, a 200×200 grid system can reliably be used for Ra of up to 10^7 with errors of less than 0.5%.
Therefore, we consider a multi-grid simulation that also uses lower grid systems,
wherever possible, to decrease the simulation cost in training our AI model.

We carried out a mesh refinement study on the input space to find the appropriate grid size to perform accurate simulations.
Although simulations using coarse grid systems may provide precise results, they only do so for a limited range of input parameters.
For example, the higher the Ra the finer the grid size required.
As such, we utilized a multi-grid dataset to train our ANN in order to reduce simulation time.
Any irregularity in the training loss could be an indication of inconsistency in the dataset due to grid error.
We effectively denoised the dataset, and retrained the ANN,
based on abnormalities observed in the training losses.

While our highest fidelity model uses 400x400 grids to simulate the problem, we also use 25x25, 50x50,
and 200x200 grid systems to generate more data anywhere the result is valid.
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
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt
import winsound
import time
start_time = time.time()
tf.get_logger().setLevel('ERROR')
rnd_seed = 1
tf.random.set_seed(rnd_seed)
np.random.seed(rnd_seed)

revised = True # True if new train data is applied

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

# Compile ANN model
def build_model():
    model = Sequential()
    model.add(Dense(4, input_dim=2, kernel_initializer='normal', activation='tanh'))
    for i in range(2):
        model.add(Dense(16, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_h),
        loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

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
ANN_folder = 'Metamodel 2' if revised else 'Metamodel 1'
ANN_dir = os.path.join(result_folder,ANN_folder)
if not os.path.exists(ANN_dir): os.makedirs(ANN_dir)
model_name = 'ANN_model.h5'
best_model_dir = os.path.join(ANN_dir,model_name)
ANN_1_folder = 'Metamodel 1 2nodes'
first_model_dir = os.path.join(result_folder,ANN_1_folder,model_name)
log_name = 'ANN_info.txt'
last_log_dir = os.path.join(ANN_dir,log_name)

# learning hyperparameters
epochs = 10000
LR_h = 0.0005
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

# build/load the model
model = load_model(first_model_dir) if revised else build_model()

cycle = 1
pat = 0
best_score = 1.
# other than decreasing the learning rate, we set a fixed limit for the learning rate considering that we use Adam optimizer.
# still, we loop through our inner process over and over again untill we see no improvement for two consequtive loops.
while pat < 3:
    loss_val = 1.
    # folders to save the results at the end of each complete loop
    output_folder = 'Model ' + str(cycle)
    output_dir = os.path.join(modeling_dir,output_folder)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    log_dir = os.path.join(output_dir,log_name)
    sys.stdout = open(log_dir, "w")
    model_dir = os.path.join(output_dir,model_name)
    # if we want to use variable limit for the learning rate:
    #K.set_value(model.optimizer.lr, LR_h)
    # we train our ANNs over multiple batch-sizes. To do this, we start from a low value and redo the training up to a batch at the size of dataset.
    # we do not reset the trained weight at the start of a training using a new batch-size (we continue the training using a new batch-size).
    # Increasing the batch-size sequentially, we achieve a better performace without worrying about the choice of the batch-size.
    data_count = len(Y_train)
    n = 0 if data_count<200 else 1
    for i in range(1-n+int(np.log2(data_count))):
        batch_size = 2**(i+1+n)
        if batch_size>data_count: batch_size = data_count
        print('\n>>> training using {} data for a batch size of {} in progress...' .format(data_count,str(batch_size)))
        history = model.fit(x=X_train,y=Y_train, batch_size = batch_size, epochs=epochs, verbose=0, 
                            validation_data = (X_val,Y_val), shuffle=True, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
        result_t = model.evaluate(X_train, Y_train, verbose=0)
        print('\n--- the loss for the training data is {}' .format(str(result_t)))
        result_v = model.evaluate(X_val, Y_val, verbose=0)
        print('--- the loss for the validation data is {}' .format(str(result_v)))   
        if result_v[0]<loss_val:
            loss_val = result_v[0]
            batch_train = batch_size
            lr_train = LR_h
            model.save(model_dir)
            history_recorder(history, output_dir)
    print('a batch size of {} for the fixed learning rate of {} resulted in the best validation loss of {}' .format(batch_train, lr_train, loss_val))
    # load the best model that we got after training using one full loop
    model = load_model(model_dir)
    model.summary()
    Y_ANN = evaluation(X_test, Y_test, 'Test')
    scatterer(Y_test, Y_ANN, output_dir, 'Test')
    # still we repeat the multi-batch-size training and wait untill we have no improvement
    if loss_val<best_score:
        best_score = loss_val
        model.save(best_model_dir)
        history_recorder(history, ANN_dir)
        pat = 0
    else:
        pat += 1
    # if we want to use variable learning rate limit
    #LR_h /= 1.25
    cycle += 1
# here we have the best model that we trained:
sys.stdout = open(last_log_dir, "w")
model = load_model(best_model_dir)
model.summary()
Y_ANN = evaluation(X_test, Y_test, 'Test')
scatterer(Y_test, Y_ANN, ANN_dir, 'Test')    
print('the best validation loss is {}' .format(best_score))
print('\n Checking the train dataset through the final model:')
# we check our training dataset to see how much accurate are our multi-model data:
# (one can then remove parts of the data from the multi-model dataset that have high errors to see if the training can be improved.)
Y_ANN_check = evaluation(X_train, Y_train, 'Train')
scatterer(Y_train, Y_ANN_check, ANN_dir, 'Train')
Y_ANN_check = evaluation(X_val, Y_val, 'Validation')
scatterer(Y_val, Y_ANN_check, ANN_dir, 'Validation')
X_train_all = data_train_all[:,0:2].reshape(-1, 2) # (Rayleigh, Prandtl)
Y_train_all = data_train_all[:,2].reshape(-1, 1)   # Nusselt
Y_ANN_check = evaluation(X_train_all, Y_train_all, 'Train_all')
scatterer(Y_train_all, Y_ANN_check, ANN_dir, 'Train_all')
print('The total runtime was {:.2f} minutes.' .format((time.time() - start_time)/60.))
winsound.Beep(2500, 1000)
