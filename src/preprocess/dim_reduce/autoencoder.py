"""
Creates and trains a determinisitic autoencoder model for dimension reduction of 4-lead ECG signals
Saves the encoded and reconstructed signals to the working data directory
"""

import numpy as np
import os
import threading
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Dropout
from tensorflow.keras.models import Sequential, Model
from src.preprocess.dim_reduce.patient_split import *


def read_in(file_index, normalized, train, ratio):
    """
    Reads in a file and can toggle between normalized and original files
    :param file_index: patient number as string
    :param normalized: binary that determines whether the files should be normalized or not
    :param train: binary that determines whether or not we are reading in data to train the model or for encoding
    :param ratio: ratio to split the files into train and test
    :return: returns npy array of patient data across 4 leads
    """
    filepath = os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + file_index + ".npy")
    if normalized == 1:
        if train == 1:
            normal_train, normal_test, abnormal = patient_split_train(filepath, ratio)
            return normal_train, normal_test
        else:
            train, test = patient_split_all(filepath, ratio)
            return train, test
    else:
        data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + file_index + ".npy"))
        return data


def build_autoencoder(sig_shape, encode_size):
    """
    Builds a deterministic autoencoder model, returning both the encoder and decoder models
    :param sig_shape: shape of input signal
    :param encode_size: dimension that we want to reduce to
    :return: encoder, decoder models
    """
    # Encoder
    encoder = Sequential()
    encoder.add(InputLayer(sig_shape))
    encoder.add(Flatten())
    encoder.add(Dense(200, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dense(125, activation='relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(100, activation = 'relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(50, activation='relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(25, activation = 'relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(encode_size))

    # Decoder
    decoder = Sequential()
    decoder.add(InputLayer((encode_size,)))
    decoder.add(Dense(25, activation = 'relu',kernel_initializer='glorot_normal'))
    decoder.add(Dense(50, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(100, activation = 'relu',kernel_initializer='glorot_normal'))
    decoder.add(Dense(125, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(200, activation = 'tanh',kernel_initializer='glorot_normal'))
    decoder.add(Dense(np.prod(sig_shape), activation = 'linear'))
    decoder.add(Reshape(sig_shape))

    return encoder, decoder


def training_ae(num_epochs, reduced_dim, file_index):
    """
    Training function for deterministic autoencoder model, saves the encoded and reconstructed arrays
    :param num_epochs: number of epochs to use
    :param reduced_dim: goal dimension
    :param file_index: patient number
    :return: None
    """
    normal, abnormal = read_in(file_index, 1, 0, 0.3)
    signal_shape = normal.shape[1:]
    encoder, decoder = build_autoencoder(signal_shape, reduced_dim)

    inp = Input(signal_shape)
    encode = encoder(inp)
    reconstruction = decoder(encode)

    autoencoder = Model(inp, reconstruction)
    autoencoder.compile(optimizer='Adam', loss='mse')

    autoencoder.fit(x=normal, y=normal, epochs=num_epochs)

    encoded = encoder.predict(abnormal)
    reconstruction = decoder.predict(encoded)

    reconstruction_save = os.path.join("Working_Data", "reconstructed_ae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    encoded_save = os.path.join("Working_Data", "reduced_ae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    # input_save = os.path.join("Working_Data", "original_data_test" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    np.save(reconstruction_save, reconstruction)
    np.save(encoded_save,encoded)
    # np.save(input_save, normal_test)


def run_over(num_epochs, encoded_dim):
    """
    Run training autoencoder over all dims in list
    :param num_epochs: number of epochs to train for
    :param encoded_dim: dimension to run on
    :return None, saves arrays for reconstructed and dim reduced arrays
    """
    indices = ['1', '4', '5', '6', '7', '8', '10', '11', '12', '14', '16', '17', '18',
               '19', '20', '21', '22', '25', '27', '28','30', '31', '32', '33', '34',
               '35', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48',
               '49', '50', '52', '53', '54', '55', '56']

    for patient_ in indices:
        print("Starting on index: " + str(patient_))
        training_ae(num_epochs, encoded_dim, patient_)
        print("Completed " + patient_ + " reconstruction and encoding, saved test data to assess performance")



#################### Training to be done for 100 epochs for all dimensions ############################################
# run_over(100,10)