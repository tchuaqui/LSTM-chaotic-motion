import double_pendulum
import numpy as np
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import re

def generate_data(angle_increments, dir_path, time_steps, t_final, M, L, g):
    """
    Function to generate data to train LSTM model.
    angle_increments determines the variation in positional angles of both masses 1 and 2.
    A matrix of initial_conditions (t = 0 s) is created using all combinations of angle_increments
    and the physical model is used to find the system's response for a given number of time_steps,
    between t = 0 and t_final.

    The inital conditions are saved to a .csv file.
    Solutions of the physical model are saved to df_i.csv files in dir_path/data/
    Note that each df_i.csv file contains the entire sequence of the double pendulum's motion
    for a given set of initial_conditions.
    """
    # Variation in angle
    positions = np.linspace(1*np.pi/180, 2*np.pi, num=angle_increments, endpoint=False)
    initial_conditions = [[theta1, 0, theta2, 0] for theta1 in positions for theta2 in positions]
    df = pd.DataFrame(initial_conditions)
    df.to_csv(dir_path + '/saved_models/initial_conditions.csv')

    # Solve ODEs
    for i in range(len(initial_conditions)):
        pendulum = double_pendulum.Pendulum(t_final=t_final, N_t=time_steps, M=M, L=L, g=g,
                                            initial_conditions=initial_conditions[i])
        pendulum.solve_physical_model()
        pendulum.save_to_csv(dir_path + f'/data/df_{i}.csv')


def moving_sequences(data_array, train_interval):
    """
    Splits data_array into inputs sequence_X and outputs (labels) sequence_y.
    sequence_X is a sequence of length train_interval, from i:i+train_interval.
    sequence_y is a single instant in time at i+train_interval+1.
    """
    sequence_X, sequence_y = [], []
    for i in range(len(data_array) - train_interval):
        X = data_array[i:i+train_interval, :]
        y = data_array[i+train_interval:i+train_interval+1, :]
        sequence_X.append(X.tolist())
        sequence_y.append(y.tolist())
    return sequence_X, sequence_y

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_accuracy_min from checkpoint to valid_accuracy_min
    valid_accuracy_min = checkpoint['valid_accuracy_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_accuracy_min.item()

class load_dataset(Dataset):
    """
    Create dataset for LSTM from pre-existing data/df_i.csv files
    """
    def __init__(self, dir_path, train_interval, transform=None):
        self.transform = transform
        self.dir_path = dir_path
        self.train_interval = train_interval
        # List of all files in dir_path
        all_files = os.listdir(dir_path + '/data/')
        # Get .csv files only
        self.files = list(filter(lambda f: f.endswith('.csv'), all_files))
        # Sort files by number
        self.files = sorted(self.files, key=lambda x: float(re.search(r'\d+', x).group(0)))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.dir_path + '/data/' + self.files[idx]
        df = pd.read_csv(file_path, index_col=0)
        X, y = moving_sequences(df.to_numpy(), self.train_interval)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return torch.FloatTensor(X), torch.FloatTensor(y)

def loaders_train_test_split(dataset, ratio_train, batch_n_files, dir_path):
    """
    Split dataset into train/test sets and return data loaders.
    The files used for train/test are saved in dir_path/saved_models/train_test_files.csv.
    """
    # Split train/test sets
    num_trains = int(len(dataset) * ratio_train)
    num_tests = len(dataset) - num_trains
    train_set, test_set = torch.utils.data.random_split(dataset, [num_trains, num_tests])
    # Create loaders
    loaders = {'train': DataLoader(dataset=train_set, batch_size=batch_n_files, shuffle=True),
               'test': DataLoader(dataset=test_set, batch_size=batch_n_files, shuffle=True)}

    # Save train/test files names to csv
    d = {'train': loaders['train'].dataset.indices, 'test': loaders['test'].dataset.indices}
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    df.to_csv(dir_path + '/saved_models/train_test_files.csv')

    return loaders

class StandardScaler_Torch:
    """
    Standard scaler (akin to sklearn.preprocessing.StandardScaler) for torch tensors.
    Includes methods for fitting, transforming and inverse transforming,
    along with saving the scaler object to and loading from an existing .csv file.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.dir_path = None

    def load(self, dir_path):
        self.dir_path = dir_path
        self.mean = torch.load(self.dir_path)['scaler_mean']
        self.std = torch.load(self.dir_path)['scaler_std']

    def fit(self, x):
        self.mean = x.view(-1, x.size(-1)).mean(0, keepdim=True)[:,None,:]
        self.std = x.view(-1, x.size(-1)).std(0, unbiased=False, keepdim=True)[:,None,:]

    def transform(self, x):
        x -= self.mean
        # Added 1e-8 to avoid division by zero
        x /= (self.std + 1e-8)
        return x

    def fit_transform(self, x):
        self.mean = x.view(-1, x.size(-1)).mean(0, keepdim=True)[:,None,:]
        self.std = x.view(-1, x.size(-1)).std(0, unbiased=False, keepdim=True)[:,None,:]
        x -= self.mean
        x /= (self.std + 1e-8)
        return x

    def inverse_transform(self, x):
        x *= (self.std + 1e-8)
        x += self.mean
        return x

    def save(self, dir_path):
        self.dir_path = dir_path
        torch.save({"scaler_mean": self.mean, "scaler_std": self.std}, self.dir_path)
