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
    # x = layers.Dense(100, activation="tanh", name="encode_layer_2")(x)z
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


if __name__ == "__main__":
    file_index = 1
    latent_dim = 10
    alpha = 0.05
    learning_rate = 0.001
    num_epoch = 100

    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))
    vae, vaefit = train_vae(data, latent_dim, alpha, learning_rate, num_epoch)

    ###############################################################################################################
    # PUT TRAIN DATA IN
    # save the z parameter? save the z-mean or z-variance? --> YES
    z = vae.encoder.predict(data)
    reconstruction = vae.decoder.predict(z)
    reconstruction_savename = os.path.join("Working_Data",
                                           "reconstructed_vaeAlpha{}_{}d_Idx{}.npy".format(alpha,
                                                                                           latent_dim,
                                                                                           file_index))
    z_savename = os.path.join("Working_Data",
                              "reduced_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + ".npy")
    np.save(reconstruction_savename, reconstruction)
    np.save(z_savename, z)