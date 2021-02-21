"""
We generate a metamodel for the free convection problem in a square enclosure.
The training of our feedforward neural network is performed using multi-models datasets.
While our highest fidelity model uses 400x400 grids to simulate the problem, we also use 25x25 and 50x50 grids to generate more data where the result is valid.
We feed our ANN sequentially by the lowest fidelity data to the highest fidelity data.
The goal is to achieve the highest accuracy by keeping the simulation cost low.
The final metamodel can be transferred into other similar problems.
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

revised = False # True if new train data is applied
tl = '(Ra,Pr,d,Nu)'

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

data_train_all = dataframe_train.values
data_val_1 = dataframe_val.values
data_test = dataframe_test.values

# split the dataset into train and validation (to consider more validation data points)
data_train, data_val_2 = train_test_split(data_train_all, test_size=0.1, random_state=rnd_seed)
data_val = np.concatenate((data_val_1,data_val_2))


# separate the input and output variables:
X_train = data_train[:,0:3].reshape(-1, 3) # (Rayleigh, Prandtl)
X_val = data_val[:,0:3].reshape(-1, 3) # (Rayleigh, Prandtl)
X_test = data_test[:,0:3].reshape(-1, 3) # (Rayleigh, Prandtl)
Y_train = data_train[:,3].reshape(-1, 1)   # Nusselt
Y_val = data_val[:,3].reshape(-1, 1)   # Nusselt
Y_test = data_test[:,3].reshape(-1, 1)   # Nusselt

# because the relationship between Nusselt and input variables follow a power function, we use the log transform to help the ANN training to be more smoothly done:
X_train[:,0:2] = np.log10(X_train[:,0:2])
X_val[:,0:2] = np.log10(X_val[:,0:2])
X_test[:,0:2] = np.log10(X_test[:,0:2])

eps = 1e-3
Y_train[Y_train<eps]=eps
Y_val[Y_val<eps]=eps
Y_test[Y_test<eps]=eps

Y_train = np.log10(Y_train)
Y_val = np.log10(Y_val)
Y_test = np.log10(Y_test)

# Compile ANN model
def build_model():
    model = Sequential()
    model.add(Dense(8, input_dim=3, kernel_initializer='normal', activation='tanh'))
    for i in range(3):
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
opt_folder = 'Brute-Force Optimization'
if not os.path.exists(opt_folder): os.makedirs(opt_folder)
modeling_folder = tl + ' no TL'
modeling_dir = os.path.join(opt_folder,modeling_folder)
if not os.path.exists(modeling_dir): os.makedirs(modeling_dir)
result_folder = 'Results'
if not os.path.exists(result_folder): os.makedirs(result_folder)
metamodels_folder = 'Metamodels'
metamodels_dir = os.path.join(result_folder,metamodels_folder)
if not os.path.exists(metamodels_dir): os.makedirs(metamodels_dir)
ANN_folder = tl + ' no TL'
ANN_dir = os.path.join(metamodels_dir,ANN_folder)
if not os.path.exists(ANN_dir): os.makedirs(ANN_dir)
model_name = 'ANN_model.h5'
best_model_dir = os.path.join(ANN_dir,model_name)
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
best_score = 1e22
# other than decreasing the learning rate, we set a fixed limit for the learning rate considering that we use Adam optimizer.
# still, we loop through our inner process over and over again untill we see no improvement for two consequtive loops.
while pat < 3:
    loss_val = 1e22
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
X_train_all = data_train_all[:,0:3].reshape(-1, 3) # (Rayleigh, Prandtl, d)
Y_train_all = data_train_all[:,3].reshape(-1, 1)   # Nusselt
X_train_all[:,0:2] = np.log10(X_train_all[:,0:2])
Y_train_all = np.log10(Y_train_all)
Y_ANN_check = evaluation(X_train_all, Y_train_all, 'Train_all')
scatterer(Y_train_all, Y_ANN_check, ANN_dir, 'Train_all')
print('The total runtime was {:.2f} minutes.' .format((time.time() - start_time)/60.))
winsound.Beep(2500, 1000)
