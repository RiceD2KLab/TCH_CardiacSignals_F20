import os
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
        regularizer = tensorflow.keras.regularizers.l1_l2(l2 = 10e-3)
        for layer in autoencoder.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
        autoencoder.compile(optimizer=opt, loss='mse')
        normal_train, normal_valid = train_test_split(fit_data, train_size=0.85, random_state=1)
        model = autoencoder.fit(x=normal_train, y=normal_train, epochs=num_epochs, batch_size=batch_size,
                                validation_data=(normal_valid, normal_valid))

        # autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        # if plot_loss:
        SMALLER_SIZE = 10
        MED_SIZE = 12
        BIG_SIZE = 18
        plt.figure()
        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MED_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MED_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MED_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        # plt.title('Example of Training and Validation Loss')
        plt.ylabel('Mean Squared Error')
        plt.title("T/V: " + file_index)
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.ylim(0,1)
        # plt.savefig("images/CDAE_" + file_index + "_loss.png", dpi=500)

        plt.show()

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format = "tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction and encoded files
        reconstruction_save = "Working_Data/reconstructed_10hb_CDAE_" + str(file_index) + "_iter" + str(iteration) + ".npy"
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

        normal_train, normal_valid = train_test_split(fit_data, train_size=0.85, random_state=1)
        model = autoencoder.fit(x=normal_train, y=normal_train, epochs=num_epochs, batch_size=batch_size,
                                validation_data=(normal_valid, normal_valid))

        # autoencoder.fit(x=fit_data, y=fit_data, epochs=num_epochs, batch_size=batch_size)

        # if plot_loss:
        SMALLER_SIZE = 10
        MED_SIZE = 12
        BIG_SIZE = 18
        plt.figure()
        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MED_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MED_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MED_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        # plt.title('Example of Training and Validation Loss')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epochs')
        plt.title("T/V: " + file_index)
        plt.legend(['Train', 'Validation'], loc='upper right')
        # plt.savefig("images/CDAE_" + file_index + "_loss.png", dpi=500)
        plt.ylim(0, 1)
        plt.show()

        if save_model:
            # save out the model
            filename = 'Working_Data/CDAE_weights_' + str(file_index) + '_train' + str(iteration) + '_model'
            autoencoder.save_weights(filename, save_format="tf")
            print('Model weights saved for patient: ' + str(file_index))

        # using autoencoder to encode all of the patient data
        encoded = encoder.predict(predict_data)
        reconstruction = decoder.predict(encoded)

        # save reconstruction and encoded files
        reconstruction_save = "Working_Data/reconstructed_10hb_CDAE_" + str(file_index) + "_iter" + str(
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
    #  ['C172', 'C174', 'C176', 'C181',
    #                           'C186', 'C203', 'C205', 'C206', 'C207', 'C209', 'C213', 'C214', 'C218', 'C219', 'C221',
    #                           'C222', 'C225', 'C234', 'C238', 'C241', 'C248', 'C249', 'C251', 'C252']
    # patient_set = ["11"] # "4", "1", "5","C106", "C11", "C214", "C109"
    for patient_index in ["C106", "1"]:  #,"C106", "C11", "C214", "4", "1", "11""C11", "C214"
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
        # second_train = noise(three)
        # third_train = noise(four)
        training_ae(110, 10, True, train_, first_predict, patient_index, 0, 0.001)
        training_ae(30, 10, True, three, five, patient_index, 1, 0.001)
        training_ae(30,10,True, four, six, patient_index, 2, 0.001)
        # training_ae(30,10,True, train_3, five)

        ## add regularizers and also potentially try retrinaing only once
        ## model is overfitting the data if I had scaled the axes correctly
        ## try with the same amount of data every time
        ## make the y-axis the same everytime
        ##  can we keep updating the regularizer as we retrain to prevent overfitting
