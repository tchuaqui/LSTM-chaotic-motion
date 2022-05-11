from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import IO_functions
import pandas as pd
import loss_metrics

class LSTM_Model(nn.Module):
    """
    Sequence to vector LSTM model. Model uses num_layers LSTM layers with hidden_size units.
    A fully connected layer is used at the end, returning tensors with 4 features [theta1, omega1, theta2, omega2]
    at the new time instant.
    """
    def __init__(self, hidden_size=100, num_layers=1):
        super(LSTM_Model, self).__init__()
        # Number of recurrent layers
        self.num_layers = num_layers
        # Number of features in hidden state
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=4, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # Fully connected layer
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=4)

    def forward(self, x):
        # x in (batch_length x sequence_length x features) format
        # Initial hidden state for each element in input sequence
        h_n = torch.zeros(self.num_layers, len(x), self.hidden_size)
        # Initial cell state for each element in input sequence
        c_n = torch.zeros(self.num_layers, len(x), self.hidden_size)
        # LSTM layer(s)
        h_t, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        # Fully connected layer taking last h_t as input
        out = self.linear(h_t[:,-1,:])
        # Add dummy tensor dimension for desired format
        return out[:,None,:]


def train_model(epochs, model, optimizer, loss_func, loaders, dir_path, dir_tb, use_cuda):
    """
    Function to train LSTM model. The best model and fitted standard scaler are saved in dir_path/saved_models/
    Training and validation loss/metrics are written to tensorboard logs.
    """

    init_epoch = 0
    # initialize minimum validation accuracy
    valid_accuracy_min = np.Inf
    # write to tensorboard visualisation
    if dir_tb is not None:
        # if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        train_step, test_step = 0, 0

    for epoch in range(init_epoch, epochs):
        # initialize variables to monitor training and validation loss & accuracy metrics
        train_loss = 0.0
        valid_loss = 0.0
        train_accuracy = 0.0
        valid_accuracy = 0.0

        # Train net
        for batch_idx, (X, y) in enumerate(tqdm(loaders['train']), 0):
            # Collapse X and y to 3-dimensions to format (batch_size x sequence_length x features)
            X = X.reshape((-1,) + X.shape[2:])
            y = y.reshape((-1,) + y.shape[2:])

            # Standardise batch training data
            scaler = IO_functions.StandardScaler_Torch()
            # Fit scaler and transform data
            scaler.fit_transform(X)
            scaler.transform(y)

            if use_cuda:
                # Move to GPU
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

            # Forward pass
            y_pred = model(X)
            # Record accuracy metric (computed before loss because it uses standardised y_pred, y)
            accuracy = loss_metrics.accuracy_metric(y_pred, y)
            # Calculate batch loss (note that y_pred, y become unstandardised as a result of loss computation)
            loss = loss_func.loss(y_pred, y, scaler)
            # Clear gradient
            optimizer.zero_grad()
            # Backward pass: compute gradient of loss with respect to model parameters
            loss.backward()
            # Perform single optimisation step
            optimizer.step()
            # Record average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # Record average accuracy metric
            train_accuracy = train_accuracy + ((1 / (batch_idx + 1)) * (accuracy.data - train_accuracy))

            # Tensorboard output
            if dir_tb is not None:
                writer.add_scalar('Training loss vs batch_idx', loss, global_step=train_step)
                writer.add_scalar('Training accuracy vs batch_idx', accuracy, global_step=train_step)
                train_step += 1
        if dir_tb is not None:
            writer.add_scalar('Training loss vs epoch', train_loss, global_step=epoch)
            writer.add_scalar('Training accuracy vs epoch', train_accuracy, global_step=epoch)

        # Validate net
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(tqdm(loaders['test']), 0):
                # Collapse X and y to 3-dimensions
                X = X.reshape((-1,) + X.shape[2:])
                y = y.reshape((-1,) + y.shape[2:])

                # Standardise batch validation data (transform only)
                scaler.transform(X)
                scaler.transform(y)

                if use_cuda:
                    # Move to GPU
                    X = Variable(X).cuda()
                    y = Variable(y).cuda()
                else:
                    X = Variable(X)
                    y = Variable(y)

                # Forward pass
                y_pred = model(X)
                # Record accuracy metric (computed before loss because it uses standardised y_pred, y)
                accuracy = loss_metrics.accuracy_metric(y_pred, y)
                # Calculate batch loss (note that y_pred, y become unstandardised as a result of loss computation)
                loss = loss_func.loss(y_pred, y, scaler)
                # Record average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                # Record average accuracy metric
                valid_accuracy = valid_accuracy + ((1 / (batch_idx + 1)) * (accuracy.data - valid_accuracy))

                # Tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss vs batch_idx', loss, global_step=test_step)
                    writer.add_scalar('Validation accuracy vs batch_idx', accuracy, global_step=test_step)
                    test_step += 1
        if dir_tb is not None:
            writer.add_scalar('Validation loss vs epoch', valid_loss, global_step=epoch)
            writer.add_scalar('Validation accuracy vs epoch', valid_accuracy, global_step=epoch)

        # print training/validation metrics
        print('Epoch: {} \tTrain Loss: {:.6f} \tValid Loss: {:.6f} \tTrain accuracy: {:.6f} \tValid accuracy: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss,
            train_accuracy,
            valid_accuracy
        ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_accuracy_min': valid_accuracy_min,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save the model if validation accuracy has decreased
        if valid_accuracy <= valid_accuracy_min:
            print('Validation accuracy decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_accuracy_min, valid_accuracy))
            # save checkpoint as best model
            torch.save(checkpoint, dir_path + '/saved_models/best_model.pt')
            # save scaler
            scaler.save(dir_path + '/saved_models/scaler.pt')
            valid_accuracy_min = valid_accuracy

def make_prediction(model, scaler, validation_example, input_sequence_length, time_steps, dir_path, forecast_all=False):
    """
    Function to make predictions using a trained model.
    """

    # Get initial sequence from data to start LSTM forecast
    x = pd.read_csv(dir_path + f'/data/df_{str(validation_example)}.csv', index_col=0).to_numpy()
    y_hat = x[:input_sequence_length, :]
    with torch.no_grad():
        if forecast_all:
            x = torch.FloatTensor(y_hat[None,:,:])
            for i in range(time_steps - input_sequence_length):
                # Predict at sequence_length+1
                y_ = model(scaler.transform(x.detach().clone()))
                scaler.inverse_transform(y_)
                # Append to input sequence for next prediction
                x = torch.cat((x[:,1:,:], y_), dim=1)
                # Reshape prediction and append to array of predictions y_hat
                y_ = y_.reshape((-1,) + y_.shape[2:]).numpy()
                y_hat = np.append(y_hat, y_, axis=0)
        else:
            x = torch.FloatTensor(x[None,:,:])
            for i in range(time_steps - input_sequence_length):
                # Predict at sequence_length+1
                y_ = model(scaler.transform(x[:, i:input_sequence_length+i, :].detach().clone()))
                scaler.inverse_transform(y_)
                # Reshape prediction and append to array of predictions y_hat
                y_ = y_.reshape((-1,) + y_.shape[2:]).numpy()
                y_hat = np.append(y_hat, y_, axis=0)
    return y_hat
