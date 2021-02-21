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

While buoyancy is what drives natural convection, the performance of a natural convection system can be affected by 
other physical phenomena and different factors such as geometry, boundary conditions, and the behavior of a fluid. 
Data-driven metamodels governing real-world natural convection systems require datasets that cover the entire feature space. 
Moreover, some unforeseen features may become active under a different process, or after a system is redesigned. 
Among other limiting factors, generating a new full set of simulations or experiments is time-consuming. 
Our methodology using TL with DNN can flexibly adapt to the expansion of the feature space when a natural convection system
becomes more complicated and needs to be described more precisely.

We adopted a TL approach using DNN. We demonstrated the capability of this approach to incorporate any additional input features.
We built a metamodel to predict Nu as a function of Ra (the only input variable), for an air-filled enclosure.
We then applied a DNN and transferred the learning of an air-filled enclosure to an enclosure with arbitrary fluid (i.e., Pr was added as a new input).
This metamodel can be retrained to predict Nu in different natural convection problems.
This TL strategy is versatile and can handle straightforward metamodeling for different engineering systems.
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
from sklearn.model_selection import KFold
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

tl = '(Ra,Pr,d,Nu)'
tl_1 = '(Ra,Pr,Nu)_revised'
if revised: tl_1 = tl

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
kf = KFold(n_splits=5, random_state = rnd_seed, shuffle = True)

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
    plt.title('ANN '+str(data))
    l_min = 0.9*np.amin(test_Y)
    l_max = 1.1*np.amax(test_Y)
    lims = [l_min, l_max]
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
modeling_folder = tl + ' kfold'
modeling_dir = os.path.join(opt_folder,modeling_folder)
if not os.path.exists(modeling_dir): os.makedirs(modeling_dir)
result_folder = 'Results'
if not os.path.exists(result_folder): os.makedirs(result_folder)
metamodels_folder = 'Metamodels'
metamodels_dir = os.path.join(result_folder,metamodels_folder)
if not os.path.exists(metamodels_dir): os.makedirs(metamodels_dir)
ANN_folder = tl + ' kfold'
ANN_dir = os.path.join(metamodels_dir,ANN_folder)
if not os.path.exists(ANN_dir): os.makedirs(ANN_dir)
model_name = 'ANN_model.h5'
best_model_dir = os.path.join(ANN_dir,model_name)
transfer_folder = tl_1
transfer_model_dir = os.path.join(metamodels_dir,transfer_folder,model_name)
log_name = 'ANN_info.txt'
last_log_dir = os.path.join(ANN_dir,log_name)

# learning hyperparameters
epochs = 10000
LR_h = 0.0005
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

# build the model
model_1 = load_model(transfer_model_dir)

if revised:
    model = model_1
    for layer in model_1.layers:
        layer.trainable = False
    model_1.layers[-1].trainable = True
    model_1.layers[-2].trainable = True
    model_1.layers[-3].trainable = True
else:
    model_1._layers.pop()
    model_1._layers.pop()
    model_1._layers.pop()
    Y_1 = model_1.layers[-1].input[0]
    Y_2 = model_1.layers[-1].input[1]
    input1 = model_1.input[0]
    input2 = model_1.input[1]

    input3 = keras.Input(shape=(1,), name='Hollow_width')
    dense_7 = layers.Dense(4, activation='tanh', name='4_nodes_hidden_layer_7')(input3)
    dense_8 = layers.Dense(16, activation='tanh', name='16_nodes_hidden_layer_8')(dense_7)
    dense_9 = layers.Dense(16, activation='tanh', name='16_nodes_hidden_layer_9')(dense_8)
    output3 = layers.Dense(1, name='Nu_d_')(dense_9)
    model_2 = keras.Model(inputs=input3, outputs=output3)
    
    input3 = model_2.input
    Y_3 = model_2.output

    Ym = layers.Concatenate(name = 'Nu_Ra_Nu_Pr_Nu_d_')([Y_1, Y_2, Y_3])
    dense_10 = layers.Dense(16, activation='tanh', name='16_nodes_hidden_layer_10')(Ym)
    dense_11 = layers.Dense(16, activation='tanh', name='16_nodes_hidden_layer_11')(dense_10)
    outputs = layers.Dense(1, name='output_Nu_number')(dense_11)

    model = keras.Model(inputs=[input1,input2,input3], outputs=outputs, name="Block_IV")

model.summary()
schm = os.path.join(ANN_dir,'model_plot.png')
plot_model(model, to_file=schm, show_shapes=True, show_layer_names=True, expand_nested=False, rankdir='TB', dpi=300)

cycle = 1
pat = 0
best_score = 1e6
# other than decreasing the learning rate, we set a fixed limit for the learning rate considering that we use Adam optimizer.
# still, we loop through our inner process over and over again untill we see no improvement for two consequtive loops.
for train_index, val_index in kf.split(data_train_all):
    data_train = data_train_all[train_index]
    data_val_2 = data_train_all[val_index]
    data_val = np.concatenate((data_val_1,data_val_2))
    # separate the input and output variables: 
    Ra_train = data_train[:,0].reshape(-1, 1) # Rayleigh
    Pr_train = data_train[:,1].reshape(-1, 1) # Prandtl
    d_train = data_train[:,2].reshape(-1, 1) # hollow diameter
    Ra_val = data_val[:,0].reshape(-1, 1) # Rayleigh
    Pr_val = data_val[:,1].reshape(-1, 1) # Prandtl
    d_val = data_val[:,2].reshape(-1, 1) # hollow diameter
    Ra_test = data_test[:,0].reshape(-1, 1) # Rayleigh
    Pr_test = data_test[:,1].reshape(-1, 1) # Prandtl
    d_test = data_test[:,2].reshape(-1, 1) # hollow diameter
    Y_train = data_train[:,3].reshape(-1, 1)   # Nusselt
    Y_val = data_val[:,3].reshape(-1, 1)   # Nusselt
    Y_test = data_test[:,3].reshape(-1, 1)   # Nusselt
    # because the relationship between Nusselt and input variables follow a power function, we use the log transform to help the ANN training to be more smoothly done:
    Ra_train = np.log10(Ra_train)
    Ra_val = np.log10(Ra_val)
    Ra_test = np.log10(Ra_test)
    Pr_train = np.log10(Pr_train)
    Pr_val = np.log10(Pr_val)
    Pr_test = np.log10(Pr_test)
    # in case Nu is very small
    eps = 1e-3
    Y_train[Y_train<eps]=eps
    Y_val[Y_val<eps]=eps
    Y_test[Y_test<eps]=eps
    Y_train = np.log10(Y_train)
    Y_val = np.log10(Y_val)
    Y_test = np.log10(Y_test)
    # separate the input and output variables:
    X_train = [Ra_train, Pr_train, d_train]
    X_val = [Ra_val, Pr_val, d_val]
    X_test = [Ra_test, Pr_test, d_test]
    
    loss_val = 1e6
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
    if data_count>500: n = 2

    # Compile ANN model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_h),
        loss='mean_squared_error', metrics=['mean_absolute_error'])

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
print('the best train loss is {}' .format(best_score))
print('\n Checking the train dataset through the final model:')
# we check our training dataset to see how much accurate are our multi-model data:
# (one can then remove parts of the data from the multi-model dataset that have high errors to see if the training can be improved.)
Y_ANN_check = evaluation(X_train, Y_train, 'Train')
scatterer(Y_train, Y_ANN_check, ANN_dir, 'Train')
Y_ANN_check = evaluation(X_val, Y_val, 'Validation')
scatterer(Y_val, Y_ANN_check, ANN_dir, 'Validation')
Ra_train_all = data_train_all[:,0].reshape(-1, 1) # (Rayleigh, Prandtl)
Pr_train_all = data_train_all[:,1].reshape(-1, 1) # (Rayleigh, Prandtl)
d_train_all = data_train_all[:,2].reshape(-1, 1) # hollow diameter
Y_train_all = data_train_all[:,3].reshape(-1, 1)   # Nusselt
Ra_train_all = np.log10(Ra_train_all)
Pr_train_all = np.log10(Pr_train_all)
Y_train_all[Y_train_all<eps]=eps
Y_train_all = np.log10(Y_train_all)
Y_ANN_check = evaluation([Ra_train_all,Pr_train_all, d_train_all], Y_train_all, 'Train_all')
scatterer(Y_train_all, Y_ANN_check, ANN_dir, 'Train_all')
print('The total runtime was {:.2f} minutes.' .format((time.time() - start_time)/60.))
winsound.Beep(2500, 1000)
