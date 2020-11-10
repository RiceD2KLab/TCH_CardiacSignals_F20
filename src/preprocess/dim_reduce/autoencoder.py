"""
Creates and trains a determinisitic autoencoder model for dimension reduction of 4-lead ECG signals
Saves the encoded and reconstructed signals to the working data directory
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv1D, MaxPooling1D, BatchNormalization, UpSampling1D
from tensorflow.keras.models import Sequential, Model
from src.preprocess.dim_reduce.patient_split import *
from src.preprocess.heartbeat_split import heartbeat_split


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
            train, test, full = patient_split_all(filepath, ratio)
            return train, test, full
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


def build_convolutional_autoencoder(sig_shape, encode_size):
    """

    :param sig_shape:
    :param encode_size:
    :return:
    """
    latent_dim = 10
    # Build the encoder
    encoder_inputs = keras.Input(shape=(100, 4))
    encoder = Sequential()
    encoder.add(InputLayer((100, 4)))

    encoder.add(Conv1D(32, 3, activation="linear", padding="same"))
    encoder.add(MaxPooling1D(2, padding="same"))

    encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
    encoder.add(MaxPooling1D(1, padding="same"))

    encoder.add(BatchNormalization())

    encoder.add(Conv1D(128, 3, activation="relu", padding="same"))
    encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
    encoder.add(MaxPooling1D(5, padding="same"))

    encoder.add(BatchNormalization())
    encoder.add(Conv1D(32, 3, activation="relu", padding="same"))
    encoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    encoder.add(Conv1D(1, 3, activation="relu", padding="same"))

    encoder.summary()
    ####################################################################################################################
    # Build the decoder
    # %%

    decoder = Sequential()
    decoder.add(InputLayer((latent_dim, 1)))
    # decoder.add(UpSampling1D(2))
    decoder.add(Conv1D(1, 7, activation="relu", padding="same"))
    decoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    decoder.add(UpSampling1D(1))

    decoder.add(Conv1D(32, 7, activation="relu", padding="same"))
    decoder.add(Conv1D(64, 9, activation="relu", padding="same"))
    decoder.add(UpSampling1D(2))

    decoder.add(Conv1D(32, 5, activation="relu", padding="same"))
    decoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    decoder.add(UpSampling1D(5))

    decoder.add(Conv1D(4, 9, activation="relu", padding="same"))
    # decoder.add(UpSampling1D(2))
    # decoder.add(Conv1D(4, 7, activation="tanh", padding="same"))

    # decoder.add(UpSampling1D(2))
    # decoder.add(Conv1D(4, 3, activation="tanh", padding="same"))

    decoder.summary()
    return encoder, decoder


def training_ae(num_epochs, reduced_dim, file_index):
    """
    Training function for deterministic autoencoder model, saves the encoded and reconstructed arrays
    :param num_epochs: number of epochs to use
    :param reduced_dim: goal dimension
    :param file_index: patient number
    :return: None
    """
    normal, abnormal, full = read_in(file_index, 1, 0, 0.3)
    signal_shape = normal.shape[1:]
    encoder, decoder = build_autoencoder(signal_shape, reduced_dim)
    # print(encoder.summary())
    # print(decoder.summary())
    inp = Input(signal_shape)
    encode = encoder(inp)
    reconstruction = decoder(encode)

    autoencoder = Model(inp, reconstruction)
    autoencoder.compile(optimizer='Adam', loss='mse')

    early_stopping = EarlyStopping(patience=20, min_delta=0.001, mode='min')
    autoencoder.fit(x=normal, y=normal, epochs=num_epochs, validation_split=0.25, callbacks=early_stopping)

    # save out the model
    filename = 'ae_patient_' + str(file_index) + '_dim' + str(reduced_dim) + '_model'
    autoencoder.save(filename + '.h5')
    print('Model saved for ' + 'patient ' + str(file_index))

    # using AE to encode other data
    encoded = encoder.predict(full)
    reconstruction = decoder.predict(encoded)

    reconstruction_save = os.path.join("Working_Data", "reconstructed_ae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    encoded_save = os.path.join("Working_Data", "reduced_ae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    # input_save = os.path.join("Working_Data", "original_data_test" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    np.save(reconstruction_save, reconstruction)
    np.save(encoded_save,encoded)
    # np.save(input_save, normal_test)


def run(num_epochs, encoded_dim):
    """
    Run training autoencoder over all dims in list
    :param num_epochs: number of epochs to train for
    :param encoded_dim: dimension to run on
    :return None, saves arrays for reconstructed and dim reduced arrays
    """
    # indicies = ["33"]
    # indicies = ['30', '31', '32',
    #             '33', '34', '35', '37', '38', '39']

    for patient_ in heartbeat_split.indicies:
        print("Starting on index: " + str(patient_))
        training_ae(num_epochs, encoded_dim, patient_)
        print("Completed " + patient_ + " reconstruction and encoding, saved test data to assess performance")



#################### Training to be done for 100 epochs for all dimensions ############################################
# for i in range(5, 20):
#     run_over(100, i)
# run(100, 10)