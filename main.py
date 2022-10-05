#!/usr/bin/env python3

"""Source code for a PCA, encoder and autoencoder, as well as training scripts

This script creates code for basic PCA code as well as source code for an LSTM
encoder and autoencoder. This also contains scripts for training said models
using input data.

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.data_processing import DataStore

import argparse
import pandas as pd
import torch as T

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


def do_pca(train_data):
    #----------------#
    # Plot PCA #
    #----------------#
    pca = PCA(2)
    x_pca = pca.fit_transform(train_data)
    x_pca = pd.DataFrame(x_pca)
    x_pca.columns=['PC1','PC2']
    print(x_pca.head())

    # Plot
    import matplotlib.pyplot as plt
    plt.scatter(x_pca['PC1'], x_pca['PC2'], alpha=0.8)
    plt.title('Scatter plot')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def init_hidden(input: T.Tensor, hidden_size: int):
    """ Initialize hidden layer weights

    Args:
        input: input tensor
        hidden_size: the size of the hidden layer
    """
    return Variable(T.zeros(1, input.size(0), hidden_size)).to(device)


class Encoder(T.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int):
        """Model initialization.

        Args:
            input_size: Size if the input
            hidden_size: Size of the hidden layer
            seq_len: Length of the training sequence windows
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = T.nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input_data: T.Tensor):
        """Run forwarding computation

        Args:
            input_data: input data as tensor
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        input_encoded = T.autograd.Variable(T.zeros(input_data.size(0), self.seq_len, self.hidden_size))

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:,t,:].unsqueeze(0), (h_t,c_t))
            input_encoded[:,t,:] = h_t
        return _, input_encoded


class AutoEncoder(T.nn.Module):
    def __init__(self, input_size: int):
        super(AutoEncoder, self).__init__()
        self.layer1 = T.nn.Linear(input_size, 4)
        self.layer2 = T.nn.Linear(4, 3)
        self.layer3 = T.nn.Linear(3, 4)
        self.layer4 = T.nn.Linear(4, input_size)
        self.double()

    def encode(self, x): # input_size-5-3
        z = T.tanh(self.layer1(x))
        z = T.tanh(self.layer2(z))
        return z

    def decode(self, x): # 3-5-input_size
        z = T.tanh(self.layer3(x))
        z = T.sigmoid(self.layer4(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z

    def train(self, data_set, batch_size: int, max_epochs: int, learning_rate: int, log_every: int = 10):
        data_ldr = T.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
        loss_fn = T.nn.MSELoss()
        opt = T.optim.SGD(self.parameters(), lr=learning_rate)
        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0
            for (batch_idx, batch) in enumerate(data_ldr):
                #print(batch_idx, batch)
                X = batch # input
                Y = batch # target

                opt.zero_grad()                     # prepare gradient
                out_target = self.forward(X)        # compute output target
                loss_val = loss_fn(out_target, Y)   # compute reconstruction loss
                epoch_loss += loss_val.item()       # accumulate error
                loss_val.backward()                 # compute gradients
                opt.step()                          # update weights

            if epoch % log_every == 0:
                print("epoch = %4d loss = %0.4f" % (epoch, epoch_loss))
        print("Done!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument_group('--output_path', type=str, default='./model.dat', help='Output path for the model.')
    parser.parse_args()

    #-----------#
    # Read Data #
    #-----------#
    print("Reading data")

    from yaml import full_load

    with open(r'./data_store.yml') as file:
        data_store = full_load(file)

    recordingmeta_data = pd.read_csv(data_store['recordingMeta_file']).squeeze().to_dict()
    tracksmeta_data = pd.read_csv(data_store['tracksMeta_file']).squeeze().to_dict()
    tracks_data = pd.read_csv(data_store['data_file'])


    #----------------------------#
    # Format, Prune & Scale Data #
    #----------------------------#
    print("Data formatting magic")
    downsample_rate = recordingmeta_data['frameRate'] / 5 # downsample by a factor 5

    parsed_markings = recordingmeta_data['upperLaneMarkings'].split(';')
    lowest_marking = max(float(marking_height) for marking_height in parsed_markings)

    train_data = tracks_data[(tracks_data['y']<=lowest_marking)].iloc[::int(downsample_rate),:] # downsample and select data from upper lane only
    train_data = pd.DataFrame(train_data[['frame', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]) # select only relevant features

    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scaled = scaler.transform(train_data)

    #-----#
    # PCA #
    #-----#
    #do_pca(train_data_scaled)

    #-------------------------#
    # Basic AutoEncoder Model #
    #-------------------------#
    print(train_data_scaled)
    input_size = train_data_scaled.shape
    print("Creating an autoencoder model with input size %d" % len(train_data_scaled[0]))
    clf = AutoEncoder(len(train_data_scaled[0])).to(device)

    clf.train(train_data_scaled, batch_size=16, max_epochs=200, learning_rate=0.1, log_every=5)

    T.save(clf.state_dict(), output_path)

    return


if __name__ == '__main__':
    main()

