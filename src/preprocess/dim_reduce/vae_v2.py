import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict
from keras.utils.vis_utils import plot_model

from src.preprocess.dim_reduce.reduction_error import *
from src.preprocess.heartbeat_split import heartbeat_split
from src.preprocess.dim_reduce.patient_split import *
from sklearn.model_selection import train_test_split

from numpy.random import seed
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
    # filepath = os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + file_index + ".npy")
    # filepath = os.path.join("Working_Data", "1000d", "Normalized_Fixed_Dim_HBs_Idx35.npy")
    filepath = "Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"

    if normalized == 1:
        if train == 1:
            # normal_test,
            normal_train, normal_test, abnormal = patient_split_train(filepath, ratio)
            # noise_factor = 0.5
            # noise_train = normal_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=normal_train.shape)
            return normal_train, normal_test # noise_train  # normal_test,
        elif train == 0:
            training, test, full = patient_split_all(filepath, ratio)
            noise_factor = 0.5
            noise_train = training + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training.shape)
            abnormal_valid = test[:len(training),:]
            return training, noise_train, abnormal_valid, test, full
    else:
        data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + file_index + ".npy"))
        return data



class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    VAE needs this random sampling -> the latent space contains parameters which represent probability distributions.
    We will sample from the probability distributions parameterized here
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0., stddev=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_loss_weight, reduced_dim, validation_data, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = kl_loss_weight
        self.reduced_dim = reduced_dim
        self.validation_data = validation_data

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            mse = tf.keras.losses.MeanSquaredError()

            reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))

            reconstruction_loss *= 1
            kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) * self.alpha
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

            #######################
            z_mean_val, z_log_var_val, z_val = self.encoder(self.validation_data)
            reconstruction_val = self.decoder(z_val)

            val_loss = tf.reduce_mean(mse(self.validation_data, reconstruction_val))
            #######################

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "val_loss": val_loss,
        }


def train_vae(data, normal_valid, latent_dim, alpha, learning_rate, num_epoch):
    """
    Train VAE for given parameters
    :param data: input data, a numpy array of many heartbeats
    :param latent_dim: desired latent dimension (integer)
    :param alpha: weighting for the KL Loss (between 0 and 1)
    :param learning_rate: Learning rate for the model (usually 0.001)
    :param num_epoch: Number of epochs of model (about 500 is usually good).
    :return: Trained model vae and the object vaefit, which is a dictionary with info about the training process.
    """
    # Build the encoder
    encoder_inputs = keras.Input(shape=(200, 4))

    x = layers.Flatten()(encoder_inputs)
    # print(np.shape(x))

    x = layers.Dense(200, activation="linear", name="encode_layer_1")(x)
    x = layers.Dense(100, activation="relu", name="encode_layer_2")(x)
    x = layers.Dense(50, activation="relu", name="encode_layer_3")(x)
    x = layers.Dense(25, activation="relu", name="encode_layer_4")(x)
    x = layers.Dense(10, activation="relu", name="encode_layer_5")(x)

    z_mean = layers.Dense(latent_dim, activation="relu", name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, activation="relu", name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # Build the decoder
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(10, activation="relu", name="decode_layer_1")(latent_inputs)
    x = layers.Dense(25, activation="relu", name="decode_layer_2")(x)
    x = layers.Dense(50, activation="relu", name="decode_layer_3")(x)
    x = layers.Dense(100, activation="relu", name="decode_layer_4")(x)
    x = layers.Dense(200, activation="relu", name="decode_layer_5")(x)
    x = layers.Dense(800, activation="linear", name="decode_layer_6")(x)

    decoder_outputs = layers.Reshape((200, 4))(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder, alpha, latent_dim, normal_valid)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    #vae.fit()
    vaefit = vae.fit(data, epochs=num_epoch, batch_size=len(data))



    return vae, vaefit


if __name__ == "__main__":
    for patient_ in [1]: #heartbeat_split.indicies:
        ################################################################################################################
        print("Starting on index: " + str(patient_))

        ################################################################################################################
        normal, noise, abnormal_valid, abnormal, all = read_in(patient_, 1, 0, 0.3)

        train, valid = train_test_split(normal, train_size=0.85, random_state=1)
        normal_train = normal[:round(len(normal) * .85), :]
        normal_valid = normal[round(len(normal) * .85):, :]
        abnormal_valid = abnormal_valid[:round(len(abnormal_valid) * 0.10), :]

        ################################################################################################################
        vae, vaefit = train_vae(normal_train, normal_valid, 10, 1, 0.002, 100)

        ################################################################################################################
        vae.encoder.save_weights('Working_Data\encoder_weights_Idx'+str(patient_)+'_d.h5')
        vae.decoder.save_weights('Working_Data\decoder_weights_Idx'+str(patient_)+'_d.h5')


        ###############################################################################################################
        # PUT TRAIN DATA IN
        # save the z parameter? save the z-mean or z-variance? --> YES
        all = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(patient_) + ".npy"))
        z = vae.encoder.predict(all)
        # z_savename = os.path.join("Working_Data",
        #                           "reduced_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + ".npy")
        # np.save(z_savename, z)


        reconstruction = vae.decoder.predict(z)
        reconstruction_savename = os.path.join("Working_Data",
                                               "reconstructed_vaeAlpha{}_{}d_Idx{}.npy".format(1, 10, patient_))
        np.save(reconstruction_savename, reconstruction)
        ################################################################################################################
        # print(np.shape(z))
        # plt.figure()
        # data_stack = z[0]
        # plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=401))
        # plt.title("Latent Variable Means - Test Data")
        # plt.legend('feature 1', 'feature 2')
        # plt.show()
        #
        # plt.figure()
        # data_stack = z[1]
        # plt.hist(data_stack, bins=np.linspace(-1.2, -1.2, num=401))
        # plt.title("Latent Variable log(Variance) - Test Data")
        # plt.show()
        #
        # plt.figure()
        # data_stack = z[2]
        # plt.hist(data_stack, bins=np.linspace(-5, 5, num=401))
        # plt.title("Sampled Latent Variable - Test Data")
        # plt.show()

        ################################################################################################################

        means = np.mean(z[0], 0)
        variances = np.mean(np.exp(z[0]), 0)
        print(means)
        print('')
        for lead in range(0,4):
            for iter in range(0,1000):
                X = np.sqrt(variances) * np.random.randn(1, 10) + means
                fake_heartbeat = vae.decoder.predict(X)

                #print(np.shape(fake_heartbeat[0][:, 0]))
                plt.plot(fake_heartbeat[0][:, lead], 'b')

            plt.title("1000 artificial heartbeats, patient "+str(patient_)+', lead '+str(lead), fontsize=18)
            plt.xlabel("Sample Index", fontsize=12)
            plt.savefig('images//frank_fake_hb_patient1_lead'+str(lead)+'.png')
            plt.show()

        # print(np.shape(all))
        # print(np.shape(all[0][:, 0]))

        # plt.plot(all[0][:, 0], 'b')
        # plt.show()

        # plt.plot(vaefit.history['loss'])
        # plt.plot(vaefit.history['reconstruction_loss'])
        # plt.plot(vaefit.history['kl_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['loss', 'reconstruction loss', 'KL loss'], loc='upper left')
        # plt.show()



        # plt.plot(vaefit.history['reconstruction_loss'])
        # plt.plot(vaefit.history['val_loss'])
        # plt.title('model loss: patient '+str(patient_))
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['reconstruction loss', 'validation loss'], loc='upper left')
        # plt.show()
        #model = ...  # Get model (Sequential, Functional Model, or Model subclass)


