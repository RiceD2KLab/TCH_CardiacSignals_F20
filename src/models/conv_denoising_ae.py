"""
Convolutional Denoising Autoencoder
Contains functions to read in preprocessed data, split according to training parameters,
train models, and save model outputs
"""
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
from src.utils.file_indexer import get_patient_ids
from src.utils.plotting_utils import *
set_font_size()


def read_in(file_index, normalized, train, split_ratio):
    """
    Reads in a file and can toggle between normalized and original files
    :param file_index: [int]  patient number as string
    :param normalized: [boolean] that determines whether the files should be normalized or not
    :param train: [int] 0 for full data for training, 1 for tuning model, 2 for full noisy data for training
    :param ratio: [float] ratio to split the files into train and test
    :return: returns npy array of patient data across 4 leads
    """
    filepath = "Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"
    if normalized:
        if train == 0:
            # returns data without modification for training models
            training, test, full = patient_split_all(filepath, split_ratio)
            return training, test, full
        elif train == 1:
            # returns normal data split into a train and test, and abnormal data
            normal_train, normal_test, abnormal = patient_split_train(filepath, split_ratio)
            return normal_train, normal_test, abnormal
        elif train == 2:
            # 3x the data, adding gaussian noise to the 2 duplicated train arrays
            train_, test, full = patient_split_all(filepath, split_ratio)
            noise_factor = 0.5
            noise_train = train_ + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_.shape)
            noise_train2 = train_ + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_.shape)
            train_ = np.concatenate((train_, noise_train, noise_train2))
            return train_, test, full
    else:
        # returns the full array
        data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + file_index + ".npy"))
        return data


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


def tuning_ae(num_epochs, encode_size, file_index, plot_loss, save_files):
    """
    Assist in tuning a model parameters and checking for overfit / underfit
    :param num_epochs: [int] number of epochs to use for training
    :param encode_size: [int] encoded dimension that model will compress to
    :param file_index: [int] patient id to run on
    :param plot_loss: [boolean] if true will plot the loss curve for the model
    :param save_files: [boolean] if true will save the .npy arrays for encoded and reconstructed heartbeats
    :return: None
    """
    normal, abnormal, all = read_in(file_index, True, 2, 0.3)
    normal_train, normal_valid = train_test_split(normal, train_size=0.85, random_state=1)

    signal_shape = normal.shape[1:]
    batch_size = round(len(normal) * 0.15)

    encoder, decoder = build_model(encode_size)

    inp = Input(signal_shape)
    encode = encoder(inp)
    reconstruction = decoder(encode)

    autoencoder = Model(inp, reconstruction)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='min')
    model = autoencoder.fit(x=normal_train, y=normal_train, epochs=num_epochs, batch_size=batch_size,
                            validation_data=(normal_valid, normal_valid), callbacks=early_stopping)
    if plot_loss:
        plt.figure()
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('Model Training Loss Patient: ' + str(file_index))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig("images/CDAE_" + file_index + "_loss.png")
        plt.show()

    if save_files:
        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(all)
        reconstruction = decoder.predict(encoded)

        # save reconstruction and encoded files
        reconstruction_save = "Working_Data/reconstructed_tuning_10hb_cae_" + str(file_index) + ".npy"
        encoded_save = "Working_Data/encoded_tuning_10hb_cae_" + str(file_index) + ".npy"
        np.save(reconstruction_save, reconstruction)
        np.save(encoded_save, encoded)


def training_ae(num_epochs, reduced_dim, file_index, save_model):
    """
    Training function for convolutional autoencoder model, saves encoded hbs, reconstructed hbs, and model files
    :param num_epochs: [int] number of epochs to use
    :param reduced_dim:  [int] encoded dimension that model will compress to
    :param file_index: [int] patient id to run on
    :param save_model: [boolean] if true saves model
    :return: None
    """
    normal, abnormal, all = read_in(file_index, 1, 2, 0.3)
    signal_shape = normal.shape[1:]
    batch_size = round(len(normal) * 0.15)

    encoder, decoder = build_model(signal_shape, reduced_dim)

    inp = Input(signal_shape)
    encode = encoder(inp)
    reconstruction = decoder(encode)

    autoencoder = Model(inp, reconstruction)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')

    autoencoder.fit(x=normal, y=normal, epochs=num_epochs, batch_size=batch_size)

    if save_model:
        # save out the model
        filename = 'Working_Data/CDAE_patient_' + str(file_index) + '_dim' + str(reduced_dim) + '_model.h5'
        autoencoder.save(filename)
        print('Model saved for patient: ' + str(file_index))

    # using autoencoder to encode all of the patient data
    encoded = encoder.predict(all)
    reconstruction = decoder.predict(encoded)

    # save reconstruction and encoded files
    reconstruction_save = "Working_Data/reconstructed_10hb_cae_" + str(file_index) + ".npy"
    encoded_save = "Working_Data/encoded_10hb_cae_" + str(file_index) + ".npy"
    np.save(reconstruction_save, reconstruction)
    np.save(encoded_save, encoded)


def run(num_epochs, encoded_dim):
    """
    Run training autoencoder over all dims in list
    :param num_epochs: number of epochs to train for
    :param encoded_dim: dimension to run on
    :return None, saves arrays for reconstructed and dim reduced arrays
    """
    for patient_ in get_patient_ids():
        print("Starting on index: " + str(patient_))
        training_ae(num_epochs, encoded_dim, patient_, True)
        print("Completed " + str(patient_) + " reconstruction and encoding, saved test data to assess performance")


# trains and saves a model for each patient from get_patient_ids
# if __name__ == "__main__":
#     run(110, 100)




