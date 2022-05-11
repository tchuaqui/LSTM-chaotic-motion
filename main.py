import double_pendulum
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import IO_functions
import models
import loss_metrics

"""
===================================
    Define problem inputs
===================================
"""
# Number of simulation time steps
time_steps = 500
# duration of simulation (s)
t_final = 10
# Masses 1 and 2 of double pendulum (kg)
M = [2.0, 5.0]
# Length 1 and 2 of double pendulum (m)
L = [1.4, 1.0]
# Gravitational acceleration (m/s**2)
g = 9.8

# Generate new data for LSTM training/validation
generate_new_data = False
# Use existing (trained) LSTM model
use_trained = True
# Number of time steps used in each sequence for LSTM model
sequence_length = 10

"""
    Number of simulation files loaded at each batch
    Note that this is not the actual batch_size used in training, which is
    batch_size = (time_steps-train_interval)*batch_n_files
"""
batch_n_files = 100
# Train/test split ratio
ratio_train = 0.8
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-5
num_layers_lstm = 1
hidden_size = 50
"""
    alpha hyperparameter (between 0 and 1) 
    Defines contribution of energy conservation penalty term in loss function.
    alpha = 0 -> loss uses MSE of positions and velocities only.
    alpha = 1 -> loss uses MSE of internal energy only.
"""
alpha = 0

"""
===================================
         Main pipeline
===================================
"""

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

def pipeline():

    # Directory for tensorboard outputs
    dir_tb = dir_path + '/runs/'
    # Generate data from initial positions
    if generate_new_data:
        IO_functions.generate_data(angle_increments=50, dir_path=dir_path, time_steps=time_steps,
                                   t_final=t_final, M=M, L=L, g=g)

    # check CUDA and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        use_cuda = True
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        use_cuda = False

    # Create model
    model = models.LSTM_Model(hidden_size=hidden_size, num_layers=num_layers_lstm)
    if use_cuda:
        model = model.cuda()

    # Create loss function and optimizer
    loss_func = loss_metrics.energy_conservation_loss(M, L, g, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # If not using previously trained model -> train new model
    if not use_trained:
        # Create dataset for DataLoader wrapper
        dataset = IO_functions.load_dataset(dir_path, train_interval=sequence_length)
        # Create loaders with train/test split
        loaders = IO_functions.loaders_train_test_split(dataset, ratio_train, batch_n_files, dir_path)
        # Train model
        models.train_model(epochs, model, optimizer, loss_func, loaders, dir_path, dir_tb, use_cuda)

    return model, optimizer

# Run pipeline
model, optimizer = pipeline()


"""
===============================================================
Compare results - LSTM vs physical model on validation examples
===============================================================
"""
# Load validation example
train_test_files_idx = pd.read_csv(dir_path + '/saved_models/train_test_files.csv', index_col=0)
test_idx = train_test_files_idx['test'].dropna().astype(int).tolist()
validation_example = test_idx[3]

# Load trained LSTM model and scaler
model, optimizer, start_epoch, valid_accuracy_min = IO_functions.load_ckp(dir_path + '/saved_models/best_model.pt', model, optimizer)
scaler = IO_functions.StandardScaler_Torch()
scaler.load(dir_path + '/saved_models/scaler.pt')

# Use LSTM to forecast pendulum motion
y_hat = models.make_prediction(model, scaler, validation_example, sequence_length, time_steps, dir_path, forecast_all=False)
pendulum_LSTM = double_pendulum.Pendulum(t_final=t_final, N_t=time_steps, M=M, L=L, g=g, solution=y_hat)

# Load physical model solution
pendulum_physical = double_pendulum.Pendulum(t_final=t_final, N_t=time_steps, M=M, L=L, g=g,
                                             import_from_csv=dir_path + f'/data/df_{str(validation_example)}.csv')

# Animations
animation1 = pendulum_physical.animation_gen(title='Physical model')
animation2 = pendulum_LSTM.animation_gen(title='LSTM model')
plt.show()

# writervideo = animation.FFMpegWriter(fps=60)
# animation2.save('video.mp4', writer=writervideo)


