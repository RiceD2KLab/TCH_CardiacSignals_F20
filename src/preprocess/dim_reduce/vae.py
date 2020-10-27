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

"""
Created: 9/22/2020, by Frank Yang
Last edited: 10/11/2020, by andypandy737 and Frank Yang
"""


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
    def __init__(self, encoder, decoder, kl_loss_weight, reduced_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = kl_loss_weight
        self.reduced_dim = reduced_dim

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            mse = tf.keras.losses.MeanSquaredError()

            reconstruction_loss = tf.reduce_mean(
                mse(data, reconstruction)
            )

            reconstruction_loss *= 1
            kl_loss = (self.reduced_dim + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) * self.alpha
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def train_vae(data, latent_dim, alpha, learning_rate, num_epoch):
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
    # x = layers.Dense(200, activation="linear", name="encode_layer_1")(x)
    # x = layers.Dense(100, activation="tanh", name="encode_layer_2")(x)
    # x = layers.Dense(50, activation="tanh", name="encode_layer_3")(x)
    # x = layers.Dense(25, activation="tanh", name="encode_layer_4")(x)
    # x = layers.Dense(10, activation="tanh", name="encode_layer_5")(x)
    #
    # z_mean = layers.Dense(latent_dim, activation="tanh", name="z_mean")(x)
    # z_log_var = layers.Dense(latent_dim, activation="tanh", name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    #
    # encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    #
    # # Build the decoder
    # latent_inputs = keras.Input(shape=(latent_dim,))
    #
    # x = layers.Dense(10, activation="tanh", name="decode_layer_1")(latent_inputs)
    # x = layers.Dense(25, activation="tanh", name="decode_layer_2")(x)
    # x = layers.Dense(50, activation="tanh", name="decode_layer_3")(x)
    # x = layers.Dense(100, activation="tanh", name="decode_layer_4")(x)
    # x = layers.Dense(200, activation="tanh", name="decode_layer_5")(x)
    # x = layers.Dense(400, activation="linear", name="decode_layer_6")(x)

    # x = layers.Dense(200, activation="linear", name="encode_layer_1")(x)
    # x = layers.Dense(100, activation="tanh", name="encode_layer_2")(x)
    # x = layers.Dense(50, activation="tanh", name="encode_layer_3")(x)
    # x = layers.Dense(25, activation="tanh", name="encode_layer_4")(x)
    # x = layers.Dense(10, activation="tanh", name="encode_layer_5")(x)
    #
    # z_mean = layers.Dense(latent_dim, activation="tanh", name="z_mean")(x)
    # z_log_var = layers.Dense(latent_dim, activation="tanh", name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    #
    # encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    #
    # # Build the decoder
    # latent_inputs = keras.Input(shape=(latent_dim,))
    #
    # x = layers.Dense(10, activation="tanh", name="decode_layer_1")(latent_inputs)
    # x = layers.Dense(25, activation="tanh", name="decode_layer_2")(x)
    # x = layers.Dense(50, activation="tanh", name="decode_layer_3")(x)
    # x = layers.Dense(100, activation="tanh", name="decode_layer_4")(x)
    # x = layers.Dense(200, activation="tanh", name="decode_layer_5")(x)
    # x = layers.Dense(400, activation="linear", name="decode_layer_6")(x)

    # x = layers.Conv1D(100, 4, activation="tanh", input_shape = [None, 400], name="encode_layer_test")(x)
    #x = layers.Conv1D(200, 3, activation="relu", padding="same")(encoder_inputs)
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
    # plot_model(encoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    vae = VAE(encoder, decoder, alpha, latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    vaefit = vae.fit(data, epochs=num_epoch, batch_size=len(data))
    return vae, vaefit


def vae_alpha_dim_sweep(file_index, dim_rng, alpha_rng, learning_rate, num_epoch, save_results=False):
    """
    Performs a sweep across the alpha(KL-weight) and the dimensions range, and optionally calls a callback
    function to execute optional code such as data splitting or plotting
    :param file_index: the patient to perform the sweep over
    :param dim_rng: range of dimension of latent space (1-15 is appropriate).
    :param alpha_rng: range of alpha (between 0 and 1)
    :param learning_rate: Learning rate of model (0.001 is usually good).
    :param num_epoch: Number of epochs of model (about 500 is usually good).
    :param save_results: True if I want to save results
    :return: Return the MSEs vs. alpha.
    """

    # Load heartbeat data
    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))
    alpha_mses = {}
    for alpha in alpha_rng:
        dim_mses = {}
        for latent_dim in dim_rng:
            print("Training vae for alpha {} and latent dimensions {} for patient{}".format(alpha, latent_dim,
                                                                                            file_index))
            vae, vaefit = train_vae(data, latent_dim, alpha, learning_rate, num_epoch)

            ###############################################################################################################
            # PUT TRAIN DATA IN
            # save the z parameter? save the z-mean or z-variance? --> YES
            z = vae.encoder.predict(data)
            reconstruction = vae.decoder.predict(z)

            if save_results:
                reconstruction_savename = os.path.join("Working_Data",
                                                       "reconstructed_vaeAlpha{}_{}d_Idx{}.npy".format(alpha,
                                                                                                       latent_dim,
                                                                                                       file_index))
                z_savename = os.path.join("Working_Data",
                                          "reduced_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + ".npy")
                np.save(reconstruction_savename, reconstruction)
                np.save(z_savename, z)

            # compute the mse between the original signal and the reconstruction
            mse = np.zeros(np.shape(data)[0])
            for i in range(np.shape(data)[0]):
                mse[i] = (np.linalg.norm(data[i, :, :] - reconstruction[i, :, :]) ** 2) / (
                        np.linalg.norm(data[i, :, :]) ** 2)

            dim_mses[latent_dim] = mse.mean()
        alpha_mses[alpha] = dim_mses

    print(alpha_mses)
    return alpha_mses


def process_vae_sweep():
    """
    Plots the computed MSEs for the VAE sweep across the alpha and dimension range (because they had to be computed on AWS)
    :return: plots of computed MSEs vs. alpha
    """

    # transform the dictionary from patient -> {alpha -> dimensions} to alpha -> {dimensions} by taking the mean
    patient_mses = pickle.load(open("Working_Data/vae_sweep_mses.pkl", "rb"))
    print(patient_mses)

    alpha_mses = defaultdict(lambda: defaultdict(list))
    for patient, alpha_map in patient_mses.items():
        for alpha, dim_map in alpha_map.items():
            for dim, err in dim_map.items():
                alpha_mses[alpha][dim].append(err)

    # now, take the mean of each alpha list
    for alpha in alpha_mses.keys():
        for dim in dim_map.keys():
            alpha_mses[alpha][dim] = np.mean(np.array(alpha_mses[alpha][dim]))

    # plot the figures
    print(alpha_mses)
    for alpha, dim_map in alpha_mses.items():
        plt.figure()
        plt.plot(dim_map.keys(), dim_map.values())
        plt.title("VAE Reconstruction Loss MSE for alpha {}".format(alpha))
        plt.xlabel("Latent Dimension")
        plt.ylabel("MSE")
        plt.show()


def plot_data_splitting(file_index, data_split_ratio, dim_range, alpha_range, learning_rate, num_epoch):
    """
    Input the file indices and data splitting ratio. Sweep the desired latent space dimensionality and range of alpha.
    Train the VAE on the first split. Evaluate the VAE on the second split. Plot the latent space means, variances, and
    sampled data.
    :param file_index: List of integers representing file indices.
    :param data_split_ratio: float between 0 and 1. It represents the way we split the heartbeats.
    :param dim_range: List of integers representing latent space dimensions.
    :param alpha_range: List of float between 0 and 1 representing the weighting of the KL loss.
    :param learning_rate: Learning rate of model (0.001 is usually good).
    :param num_epoch: Number of epochs of model (about 500 is usually good).
    :return: Plots of latent space means, variances, and samples for the first dimension on train and test data.
    """
    # data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))

    filestring = os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy")

    for alpha in alpha_range:
        for latent_dim in dim_range:
            # splitting_idx = round(len(data) * data_split_ratio)
            # data_train = data[0:splitting_idx]
            # data_test = data[(splitting_idx + 1):]
            data_train, data_test, dummy_remainder = patient_split(filestring, data_split_ratio)

            # only train on the first 5 hours
            vae, vaefit = train_vae(data_train, latent_dim, alpha, learning_rate, num_epoch)
            z = vae.encoder.predict(data_test)

            # visualize the loss convergence as we iterate
            plt.plot(vaefit.history['loss'])
            plt.plot(vaefit.history['reconstruction_loss'])
            plt.plot(vaefit.history['kl_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'reconstruction loss', 'KL loss'], loc='upper left')
            plt.show()

            plt.figure()
            data_stack = z[0]
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=401))
            plt.title("Latent Variable Means - Train Data")
            plt.show()

            plt.figure()
            data_stack = z[1]
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=401))
            plt.title("Latent Variable log(Variance) - Train Data")
            plt.show()

            plt.figure()
            data_stack = z[2]
            plt.hist(data_stack, bins=np.linspace(-5, 5, num=201))
            plt.title("Sampled Latent Variable - Train Data")
            plt.show()

            #######################################################################################################
            # PUT TEST DATA IN
            # save the z parameter? save the z-mean or z-variance? --> YES
            z = vae.encoder.predict(data_test)
            reconstruction = vae.decoder.predict(z)
            print('TESTING SHTI')
            print(np.shape(z))

            # visualize the loss convergence as we iterate
            plt.figure()
            data_stack = z[0]
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=201))
            plt.title("Latent Variable Means - Test Data")
            plt.show()

            plt.figure()
            data_stack = z[1]
            plt.hist(data_stack, bins=np.linspace(-1.2, -1.2, num=401))
            plt.title("Latent Variable log(Variance) - Test Data")
            plt.show()

            plt.figure()
            data_stack = z[2]
            plt.hist(data_stack, bins=np.linspace(-5, 5, num=401))
            plt.title("Sampled Latent Variable - Test Data")
            plt.show()

            data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))

<<<<<<< HEAD
            # print(np.shape(data[200][:]))
            #
            # z_100 = vae.encoder.predict(data[100][:])
            # print('heartbeat 100')
            # print(z_100)
            # print(' ')
            # z_200 = vae.encoder.predict(data[200][:])
            # print('heartbeat 200')
            # print(z_200)

            z_all = vae.encoder.predict(data)
            reconstruct_all = vae.decoder.predict(z_all)

            for heartbeat_num in [100, 200]:
                print(str(heartbeat_num))
                print(z[0][heartbeat_num][0])
                print(z[1][heartbeat_num][0])
                print(z[2][heartbeat_num][0])
                print(' ')
                for lead_num in range(4):
                    plt.plot(data[heartbeat_num, :, lead_num])
                    plt.plot(reconstruct_all[heartbeat_num, :, lead_num])
                    plt.title(
                        "Patient 1, Heartbeat {}, Lead = {}, Latent Dim. = {} ".format(heartbeat_num, lead_num, 1))
                    plt.xlabel("Sample Index")

                    plt.legend(['Original', 'Reconstructed'], loc='upper left')
                    plt.show()


#
# if __name__ == "__main__":
#     ## This code sweeps the VAE performance without any data split. Useful for optimization.
#     # patient_mses = {}
#     # for file_index in heartbeat_split.indicies[:1]:
#     #     patient_mses[file_index] = vae_alpha_dim_sweep(file_index, range(1, 2), [1], 0.001, 10, save_results=True)
#     #
#     # outfile = open("Working_Data/vae_sweep_mses.pkl", 'wb')
#     # pickle.dump(patient_mses, outfile)
#     # outfile.close()
#     #
#     # process_vae_sweep()
#
#     # if we want to perform data splitting across a smaller dimension range:
#     for file_index in heartbeat_split.indicies[:1]:
#          plot_data_splitting(file_index, 1/2, range(1, 2), [0], 0.005, 300)



plot_data_splitting(4, 0.5, [1], [1], 0.005, 3)







# filestring = os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(4) + ".npy")
# data_train, data_test, dummy_remainder = patient_split(filestring, 0.5)
# print(np.shape(data_train))
# print(type(data_train))
# print(data_train[0, :, 0])
#
# for i in range(0,4):
#     print(i)
#     plt.plot(data_train[0, :, i])
#     plt.show()

=======
if __name__ == "__main__":
    ## This code sweeps the VAE performance without any data split. Useful for optimization.
    patient_mses = {}
    for file_index in heartbeat_split.indicies:
        patient_mses[file_index] = vae_alpha_dim_sweep(file_index, [10], [0.05], 0.001, 200, save_results=True)
    #
    # outfile = open("Working_Data/vae_sweep_mses.pkl", 'wb')
    # pickle.dump(patient_mses, outfile)
    # outfile.close()
    #
    # process_vae_sweep()

    # if we want to perform data splitting across a smaller dimension range:
    # for file_index in heartbeat_split.indicies[:1]:
    #      plot_data_splitting(file_index, 5/6, range(1, 2), [1], 0.001, 2)
    #      compare_reconstructed_hb(file_index, 100, 'vae', 1)
>>>>>>> 7ed476e2a14d364aaeca744896cc3976d6216f7a
