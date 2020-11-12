"""
Convolutional Autoencoder Model Architecture
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv1D, MaxPooling1D, BatchNormalization, UpSampling1D, Conv1DTranspose
from tensorflow.keras.models import Sequential, Model


def build_convolutional_autoencoder(data, encode_size):
    """

    :param sig_shape:
    :param encode_size:
    :return:
    """
    # latent_dim = 10
    # # Build the encoder
    # encoder_inputs = keras.Input(shape=(100, 4))
    # encoder = Sequential()
    # encoder.add(InputLayer((100, 4)))
    #
    # encoder.add(Conv1D(32, 3, activation="linear", padding="same"))
    # encoder.add(MaxPooling1D(2, padding="same"))
    #
    # encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
    # encoder.add(MaxPooling1D(1, padding="same"))
    #
    # encoder.add(BatchNormalization())
    #
    # encoder.add(Conv1D(128, 3, activation="relu", padding="same"))
    # encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
    # encoder.add(MaxPooling1D(5, padding="same"))
    #
    # encoder.add(BatchNormalization())
    # encoder.add(Conv1D(32, 3, activation="relu", padding="same"))
    # encoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    # encoder.add(Conv1D(1, 3, activation="relu", padding="same"))
    #
    # encoder.summary()
    # ####################################################################################################################
    # # Build the decoder
    # # %%
    #
    # decoder = Sequential()
    # decoder.add(InputLayer((latent_dim, 1)))
    # # decoder.add(UpSampling1D(2))
    # decoder.add(Conv1D(1, 7, activation="relu", padding="same"))
    # decoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    # decoder.add(UpSampling1D(1))
    #
    # decoder.add(Conv1D(32, 7, activation="relu", padding="same"))
    # decoder.add(Conv1D(64, 9, activation="relu", padding="same"))
    # decoder.add(UpSampling1D(2))
    #
    # decoder.add(Conv1D(32, 5, activation="relu", padding="same"))
    # decoder.add(Conv1D(16, 3, activation="relu", padding="same"))
    # decoder.add(UpSampling1D(5))
    #
    # decoder.add(Conv1D(4, 9, activation="relu", padding="same"))
    # # decoder.add(UpSampling1D(2))
    # # decoder.add(Conv1D(4, 7, activation="tanh", padding="same"))
    #
    # # decoder.add(UpSampling1D(2))
    # # decoder.add(Conv1D(4, 3, activation="tanh", padding="same"))

    latent_dim = 100
    # Build the encoder
    # encoder = Sequential()
    # encoder.add(InputLayer((1000, 4)))
    # encoder.add(Conv1D(10, 7, activation="linear", padding="same"))
    #
    # encoder.add(Flatten())
    # encoder.add(Dense(750, activation='tanh', kernel_initializer='glorot_normal'))
    # encoder.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    # encoder.add(Dense(400, activation='relu', kernel_initializer='glorot_normal'))
    # encoder.add(Dense(300, activation='relu', kernel_initializer='glorot_normal'))
    # encoder.add(Dense(200, activation='relu', kernel_initializer='glorot_normal'))
    # encoder.add(Dense(latent_dim))

    encoder = Sequential()
    encoder.add(InputLayer((1000,4)))
    # idk if causal is really making that much of an impact but it seems useful for time series data?
    encoder.add(Conv1D(10, 11, activation="linear", padding="causal"))
    encoder.add(Conv1D(10, 5, activation="relu", padding="causal"))
    # encoder.add(Conv1D(10, 3, activation="relu", padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(750, activation = 'tanh', kernel_initializer='glorot_normal')) #tanh
    encoder.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(400, activation = 'relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(300, activation='relu', kernel_initializer='glorot_normal'))
    encoder.add(Dense(200, activation = 'relu', kernel_initializer='glorot_normal')) #relu
    encoder.add(Dense(latent_dim))
    # encoder.summary()
    ####################################################################################################################
    # Build the decoder

    # decoder = Sequential()
    # decoder.add(InputLayer((latent_dim,)))
    # decoder.add(Dense(200, activation='tanh', kernel_initializer='glorot_normal'))
    # decoder.add(Dense(300, activation='relu', kernel_initializer='glorot_normal'))
    # decoder.add(Dense(400, activation='relu', kernel_initializer='glorot_normal'))
    # decoder.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    # decoder.add(Dense(750, activation='relu', kernel_initializer='glorot_normal'))
    # decoder.add(Dense(10000, activation='relu', kernel_initializer='glorot_normal'))
    # decoder.add(Reshape((1000, 10)))
    # decoder.add(Conv1DTranspose(4, 7, activation="relu", padding="same"))

    decoder = Sequential()
    decoder.add(InputLayer((latent_dim,)))
    decoder.add(Dense(200, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dense(300, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(400, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(750, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Dense(10000, activation='relu', kernel_initializer='glorot_normal'))
    decoder.add(Reshape((1000, 10)))
    # decoder.add(Conv1DTranspose(8, 3, activation="relu", padding="same"))
    decoder.add(Conv1DTranspose(8, 5, activation="relu", padding="same"))
    decoder.add(Conv1DTranspose(4, 11, activation="linear", padding="same"))

    return encoder, decoder

    # decoder.summary()
    # return encoder, decoder