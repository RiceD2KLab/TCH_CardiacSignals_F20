import os
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv1D, MaxPooling1D, Conv1DTranspose
from tensorflow.keras.models import Sequential, Model
from src.models.patient_split import *
from sklearn.model_selection import train_test_split
from src.models.conv_denoising_ae import *


def build_model(encode_size):
    """
    Builds a convolutional autoencoder model, returning both the encoder and decoder models
    :param encode_size: [int] dimension that we want to reduce to
    :return: encoder, decoder models
    """
    # Build the encoder
    encoder = Sequential()
    encoder.add(InputLayer((1000,4)))
    encoder.add(Conv1D(5, 11, activation="tanh", padding="same"))
    encoder.add(Conv1D(7, 7, activation="relu", padding="same"))
    encoder.add(MaxPooling1D(2))
    encoder.add(Conv1D(11, 5, activation="tanh", padding="same"))
    encoder.add(Conv1D(11, 3, activation="tanh", padding="same"))
    encoder.add(MaxPooling1D(2))
    encoder.add(Flatten())
    encoder.add(Dense(750, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dense(400, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dense(200, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dense(encode_size))

    # Build the decoder
    decoder = Sequential()
    decoder.add(InputLayer((encode_size,)))
    decoder.add(Dense(200, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dense(400, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dense(750, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dense(10000, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Reshape((1000, 10)))
    decoder.add(Conv1DTranspose(8, 11, activation="relu", padding="same"))
    decoder.add(Conv1DTranspose(4, 5, activation="linear", padding="same"))

    return encoder, decoder


def training_ae(num_epochs, reduced_dim, save_model, fit_data, predict_data, file_index, iteration, lr):
    """
    Training function for convolutional autoencoder model, saves encoded hbs, reconstructed hbs, and model files
    :param num_epochs: [int] number of epochs to use
    :param reduced_dim:  [int] encoded dimension that model will compress to
    :param file_index: [int] patient id to run on
    :param save_model: [boolean] if true saves model
    :return: None
    """
    if iteration == 0:
        signal_shape = fit_data.shape[1:]
        batch_size = round(len(fit_data) * 0.15)

        encoder, decoder = build_model(reduced_dim)

        inp = Input(signal_shape)
        encode = encoder(inp)
        reconstruction = decoder(encode)

        autoencoder = Model(inp, reconstruction)
        opt = keras.optimizers.Adam(learning_rate=lr)
        autoencoder.compile(optimizer=opt, loss='mse')

        autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format = "tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction and encoded files
        reconstruction_save = "Working_Data/reconstructed_10hb_cae_" + str(file_index) + "iter" + str(iteration) + ".npy"
        # encoded_save = "Working_Data/encoded_10hb_cae_" + str(file_index) + ".npy"
        np.save(reconstruction_save, reconstruction)
        print("Reconstructed hbs saved for patient:" + str(file_index) + " iteration: " + str(iteration))
        # np.save(encoded_save, encoded)
    else:
        signal_shape = fit_data.shape[1:]
        batch_size = round(len(fit_data) * 0.15)

        encoder, decoder = build_model(reduced_dim)

        inp = Input(signal_shape)
        encode = encoder(inp)
        reconstruction = decoder(encode)

        autoencoder = Model(inp, reconstruction)
        autoencoder.load_weights('Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration-1) + '_model')
        opt = keras.optimizers.Adam(learning_rate=lr)
        autoencoder.compile(optimizer=opt, loss='mse')

        autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format="tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction and encoded files
        reconstruction_save = "Working_Data/reconstructed_10hb_cae_" + str(file_index) + "iter" + str(
            iteration) + ".npy"
        # encoded_save = "Working_Data/encoded_10hb_cae_" + str(file_index) + ".npy"
        np.save(reconstruction_save, reconstruction)
        print("Reconstructed hbs saved for patient:" + str(file_index) + " iteration: " + str(iteration))
        # np.save(encoded_save, encoded)


def noise(data):
    noise_factor = 0.5
    noise_train = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noise_train2 = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    train_ = np.concatenate((data, noise_train, noise_train2))
    return train_


# train a model, save reconstruction and then move to next time chunk training and reconstruction
if __name__ == "__main__":
    patient_set = [ "C106", "C11", "C214", "C109"] # "4", "1", "5",
    for patient_index in patient_set:
        file_index = patient_index
        filepath = "Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"
        split_ratio = 0.3
        train_, remaining = patient_split_adaptive(filepath, split_ratio)
        train_ = noise(train_)
        three, four, five, six = split(remaining, 4)
        first_predict = np.concatenate((three, four))
        second_train = noise(three)
        third_train = noise(four)
        training_ae(110, 10, True, train_, first_predict, patient_index, 0, 0.001)
        training_ae(30, 10, True, second_train, five, patient_index, 1, 0.01)
        training_ae(30,10,True, third_train, six, patient_index, 2, 0.01)
        # training_ae(30,10,True, train_3, five)

