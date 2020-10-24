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
        batch = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
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
            #print('FUCK')
            #print(np.shape(self.encoder(data)))
            #print(np.shape(z_mean))
            #print(np.shape(z_log_var))
            #print(np.shape(z))
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
    print(np.shape(encoder_inputs))
    x = layers.Conv1D(64, 3, activation="linear", padding="same")(encoder_inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)

    x = layers.Conv1D(32, 3, activation="tanh", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)

    x = layers.Conv1D(16, 3, activation="tanh", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)

    x = layers.Conv1D(8, 3, activation="tanh", padding="same")(x)
    x = layers.MaxPooling1D(4, padding="same")(x)

    x = layers.Conv1D(4, 3, activation="tanh", padding="same")(x)
    x = layers.MaxPooling1D(7, padding="same")(x)

    z_mean = layers.Conv1D(latent_dim, 1, activation="relu", name="z_mean")(x)
    z_log_var = layers.Conv1D(latent_dim, 1, activation="relu", name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    ####################################################################################################################
    # Build the decoder

    latent_inputs = keras.Input(shape=(latent_dim, 1))
    print(np.shape(latent_inputs))
    x = layers.UpSampling1D(2)(latent_inputs)
    x = layers.Conv1D(4, 3, activation="tanh", padding="same")(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(8, 3, activation="tanh", padding="same")(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation="tanh", padding="same")(x)

    x = layers.UpSampling1D(5)(x)
    x = layers.Conv1D(32, 3, activation="tanh", padding="same")(x)

    x = layers.UpSampling1D(5)(x)
    x = layers.Conv1D(64, 3, activation="tanh", padding="same")(x)

    decoder_outputs = layers.Conv1D(4, 2, activation="linear", padding="same")(x)


    #decoder_outputs = layers.Reshape((200, 4))(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    # plot_model(encoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    vae = VAE(encoder, decoder, alpha, latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    vaefit = vae.fit(data, epochs=num_epoch, batch_size=len(data))
    return vae, vaefit

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

            print('EFWEFWEFEF')
            print(np.shape(data_train))


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
            data_stack = np.squeeze(z[0])
            print('TESTSESTEST')
            print(np.shape(data_stack))
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=401))
            plt.title("Latent Variable Means - Train Data")
            plt.show()

            plt.figure()
            data_stack = np.squeeze(z[1])
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=401))
            plt.title("Latent Variable log(Variance) - Train Data")
            plt.show()

            plt.figure()
            data_stack = np.squeeze(z[2])
            plt.hist(data_stack, bins=np.linspace(-5, 5, num=201))
            plt.title("Sampled Latent Variable - Train Data")
            plt.show()

            ###############################################################################################################
            # PUT TEST DATA IN
            # save the z parameter? save the z-mean or z-variance? --> YES
            z = vae.encoder.predict(data_test)
            reconstruction = vae.decoder.predict(z)
            # print(np.shape(z[2]))

            # visualize the loss convergence as we iterate
            plt.figure()
            data_stack = np.squeeze(z[0])
            plt.hist(data_stack, bins=np.linspace(-1.2, 1.2, num=201))
            plt.title("Latent Variable Means - Test Data")
            plt.show()

            plt.figure()
            data_stack = np.squeeze(z[1])
            plt.hist(data_stack, bins=np.linspace(-1.2, -1.2, num=401))
            plt.title("Latent Variable log(Variance) - Test Data")
            plt.show()

            plt.figure()
            data_stack = np.squeeze(z[2])
            plt.hist(data_stack, bins=np.linspace(-5, 5, num=401))
            plt.title("Sampled Latent Variable - Test Data")
            plt.show()

            data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))

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



plot_data_splitting(4, 0.5, [1], [0], 0.0005, 500)





