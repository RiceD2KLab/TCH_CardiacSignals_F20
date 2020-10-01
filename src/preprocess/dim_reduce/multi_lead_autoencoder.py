"""
Creates a normal sequential autoencoder reducing dimensionality of data and saving the output as numpy arrays
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Dropout
from tensorflow.keras.models import Sequential, Model


def read_in(file_index, normalized):
    """
    :param file_index: patient number as string
    :param normalized: boolean that determines whether the files should be normalized or not
    :return: returns npy array of patient data across 4 leads
    """
    if normalized == 1:
        data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + file_index + ".npy"))
    else:
        data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + file_index + ".npy"))

    return data


def build_autoencoder(sig_shape, encode_size):
    """
    Builds a sequential autoencoder
    :param sig_shape: shape of input signal
    :param encode_size: dimension that we want to reduce to
    :return: encoder, decoder models
    """
    # Encoder
    encoder = Sequential()
    encoder.add(InputLayer(sig_shape))
    encoder.add(Flatten())
    # encoder.add(Dense(350, activation = 'tanh'))
    encoder.add(Dense(200, activation = 'tanh', kernel_initializer='normal'))
    encoder.add(Dense(100, activation = 'tanh', kernel_initializer='normal'))
    encoder.add(Dense(25, activation = 'tanh', kernel_initializer='normal'))
    encoder.add(Dense(encode_size))

    # Decoder
    decoder = Sequential()
    decoder.add(InputLayer((encode_size,)))
    decoder.add(Dense(25, activation = 'tanh',kernel_initializer='normal'))
    decoder.add(Dense(100, activation = 'tanh',kernel_initializer='normal'))
    decoder.add(Dense(200, activation = 'tanh',kernel_initializer='normal'))
    # decoder.add(Dense(350, activation = 'tanh'))
    decoder.add(Dense(np.prod(sig_shape), activation = 'linear'))
    decoder.add(Reshape(sig_shape))

    return encoder, decoder


def training_ae(num_epochs, reduced_dim, file_index):
    """
    Training function for autoencoder
    :param num_epochs: number of epochs to use
    :param reduced_dim: goal dimension
    :param file_index: patient number
    :return: saved numpy arrays of the encoded and reconstructed signals
    """
    data = read_in(file_index,1)
    signal_shape = data.shape[1:]
    encoder, decoder = build_autoencoder(signal_shape, reduced_dim)

    inp = Input(signal_shape)
    encode = encoder(inp)
    reconstruction = decoder(encode)

    autoencoder = Model(inp, reconstruction)
    autoencoder.compile(optimizer='adam', loss='mse')

    mod = autoencoder.fit(x=data, y=data, epochs=num_epochs)

    encoded = encoder.predict(data)
    reconstruction = decoder.predict(encoded)

    reconstruction_save = os.path.join("Working_Data", "reconstructed_ae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    encoded_save = os.path.join("Working_Data", "reduced_vae_" + str(reduced_dim) + "d_Idx" + str(file_index) + ".npy")
    np.save(reconstruction_save, reconstruction)
    np.save(encoded_save,encoded)


def run_over(num_epochs, encoded_dim):
    """
    Run training autoencoder over all dims in list
    :param dims: dimension to run on
    :return None, saves arrays for reconstructed and dim reduced arrays
    """
    indices = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
            '33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']

    for patient_ in indices:
        print("Starting on index: " + str(patient_))
        training_ae(num_epochs, encoded_dim, patient_)
        print("Completed " + patient_ + " reconstruction and encoding")

if __name__ == "__main__":
    run_over(40,1)