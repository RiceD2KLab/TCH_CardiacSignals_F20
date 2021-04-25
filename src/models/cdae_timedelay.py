"""
Transfer Learning (Time Delayed) version of the Convolutional Denoising Autoencoder
Contains functions to read in preprocessed data, split according to training parameters,
train models, and save model outputs
"""


import os
import logging
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv1D, MaxPooling1D, Conv1DTranspose, Dropout
from tensorflow.keras.models import Sequential, Model
from src.models.patient_split import *
from sklearn.model_selection import train_test_split
from src.models.conv_denoising_ae import *
from src.utils.file_indexer import get_patient_ids


def noise(data):
    """
    Pass through the data, adds two noised versions of the data to the original version and returns
    :param data: preprocessed heartbeat data
    :return: noise version of data
    """
    noise_factor = 0.5
    noise_train = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noise_train2 = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    train_ = np.concatenate((data, noise_train, noise_train2))
    return train_


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
    encoder.add(Dropout(0.2))
    encoder.add(Dense(400, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dropout(0.2))
    encoder.add(Dense(200, activation = 'tanh', kernel_initializer='glorot_normal'))
    encoder.add(Dense(encode_size))

    # Build the decoder
    decoder = Sequential()
    decoder.add(InputLayer((encode_size,)))
    decoder.add(Dense(200, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dense(400, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dropout(0.2))
    decoder.add(Dense(750, activation='tanh', kernel_initializer='glorot_normal'))
    decoder.add(Dropout(0.2))
    # decoder.add(Dense(5000, activation='tanh', kernel_initializer='glorot_normal'))
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
    :param save_model: [boolean] save the model file (required for time delay)
    :param fit_data: [array] data for the model to be trained on
    :param predict_data: [array] data for the model to predict
    :param file_index: [int] patient id to run on
    :param iteration: [int] version of time delay to run (0: first run on patient, 1: second run on patient, 2: last run on patient)
    :param lr: [float] learning rate to use for a particular run of the time delay model
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
        regularizer = tensorflow.keras.regularizers.l1_l2(l2 = 10e-3)
        for layer in autoencoder.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
        autoencoder.compile(optimizer=opt, loss='mse')

        model = autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format = "tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction files only
        reconstruction_save = "Working_Data/reconstructed_10hb_CDAE_" + str(file_index) + "_iter" + str(iteration) + ".npy"

        np.save(reconstruction_save, reconstruction)
        print("Reconstructed hbs saved for patient:" + str(file_index) + " iteration: " + str(iteration))

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

        model = autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format="tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction files only
        reconstruction_save = "Working_Data/reconstructed_10hb_CDAE_" + str(file_index) + "_iter" + str(
            iteration) + ".npy"

        np.save(reconstruction_save, reconstruction)
        print("Reconstructed hbs saved for patient:" + str(file_index) + " iteration: " + str(iteration))


def train_model(patient_index):
    """
    Run the transfer learning training procedure for an individual patient
    :param patient_index: patient index
    :return: 
    """
    try:
        # if int(patient_index) < 22:
        #     continue
        file_index = patient_index
        print("Starting training on patient ", patient_index)
        filepath = "Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"
        split_ratio = 0.3
        train_, remaining = patient_split_adaptive(filepath, split_ratio)
        # train_noise = noise(train_)
        three, four, five, six = split(remaining, 4)
        first_predict = np.concatenate((train_, three, four))
        second_train = noise(three)
        third_train = noise(four)

        training_ae(110, 10, True, train_, first_predict, patient_index, 0, 0.001)
        training_ae(30, 10, True, second_train, five, patient_index, 1, 0.001)
        training_ae(30, 10, True, third_train, six, patient_index, 2, 0.001)
    except Exception as e:
        logging.critical(f"COULD NOT COMPLETE TRAINING FOR PATIENT {patient_index}")
        logging.info(e)


# train a model, save reconstruction and then move to next time chunk training and reconstruction
if __name__ == "__main__":
    # setup logging basic configuration for logging to a file
    logging.basicConfig(filename="transfer.log")
    all_patients = get_patient_ids(False) + get_patient_ids(True)
    for idx in all_patients:
        train_model(idx)
