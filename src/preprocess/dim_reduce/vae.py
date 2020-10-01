import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import minmax_scale

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
print(np.shape(mnist_digits))

"""
Frank Yang
Created: 9/22/2020, by Frank Yang
Last edited: 10/1/2020, by Frank Yang

Input:  patient data! Further down, you can specify which patients, which leads, 
        and what values of lower-dimension in the latent space (this should be 1,2,3...15)
        
Output: The (1) reduced dimension latent space and (2) the reconstructed data
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
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)

            mse = tf.keras.losses.MeanSquaredError()

            reconstruction_loss = tf.reduce_mean(
                mse(data, reconstruction)
            )

            reconstruction_loss *= 1
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
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

"""
1. Build the variational auto-encoder
2. Import and normalize heartbeat data
3. Run the VAE

file_index: Choose a given patient idx
lead_num: lead numbers are 0,1,2,3
latent_dim: number of dimensions in the latent space, keep this below 15
num_epoch: number of iterations of gradient descent, 200 is a good number
learning_rate: rate of gradient descent, 0.01 is a good number
"""

for file_index in [1]:
    # Load heartbeat data
    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))

    for latent_dim in [2]:
        num_epoch = 200
        learning_rate = 0.001  # this is the default

        # Build the encoder
        encoder_inputs = keras.Input(shape=(100, 4))
        x = layers.Flatten()(encoder_inputs)

        #x = layers.Dense(200, activation="tanh", name="encode_layer_1")(x)
        #x = layers.Dense(100, activation="tanh", name="encode_layer_2")(x)
        #x = layers.Dense(50, activation="tanh", name="encode_layer_3")(x)
        #x = layers.Dense(25, activation="tanh", name="encode_layer_4")(x)
        #x = layers.Dense(10, activation="tanh", name="encode_layer_5")(x)

        z_mean = layers.Dense(latent_dim, activation="tanh", name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, activation="tanh", name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        latent_inputs = keras.Input(shape=(latent_dim,))

        #x = layers.Dense(10, activation="tanh", name="decode_layer_1")(latent_inputs)
        #x = layers.Dense(25, activation="tanh", name="decode_layer_2")(x)
        #x = layers.Dense(50, activation="tanh", name="decode_layer_3")(x)
        #x = layers.Dense(100, activation="tanh", name="decode_layer_4")(x)
        #x = layers.Dense(200, activation="tanh", name="decode_layer_5")(x)
        #x = layers.Dense(400, activation="tanh", name="decode_layer_6")(x)

        x = layers.Dense(400, activation="tanh", name="decode_layer_6")(latent_inputs)
        decoder_outputs = layers.Reshape((100, 4))(x)


        #x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        #x = layers.Reshape((7, 7, 64))(x)
        #decoder_outputs = layers.Conv2DTranspose(100, 4, activation="sigmoid", padding="same")(x)

        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        # Normalize each heartbeat to (min=0,max=1). This is important when using ReLu in the NNs
        # for iter1 in range(len(lead_data)):
        #    lead_data[iter1, :] = minmax_scale(lead_data[iter1, :], feature_range=(0, 1), axis=0, copy=True)

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        #print(np.shape(data))

        vaefit = vae.fit(data, epochs=num_epoch, batch_size=len(data))

        # save the z parameter? save the z-mean or z-variance? --> YES
        z = encoder.predict(data)
        reconstruction = decoder.predict(z)

        # visualize the loss convergence as we iterate

        plt.plot(vaefit.history['loss'])
        plt.plot(vaefit.history['reconstruction_loss'])
        plt.plot(vaefit.history['kl_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'reconstruction loss', 'KL loss'], loc='upper left')
        plt.show()

        htbt_idx = 0
        # visualize the first heartbeat
        for lead_idx in [0, 1, 2, 3]:
            plt.plot([i for i in range(len(data[htbt_idx, :, lead_idx]))], data[htbt_idx, :, lead_idx])
            plt.plot([i for i in range(len(reconstruction[htbt_idx, :, lead_idx]))], reconstruction[htbt_idx, :, lead_idx])
            plt.legend(['original', 'reconstructed'], loc='upper left')
            plt.title("VAE output comparison (heartbeat {}, lead {})".format(htbt_idx, lead_idx+1))
            plt.xlabel('Index')
            plt.show()


        log_filepath = os.path.join("Working_data", "")
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        reconstruction_savename = os.path.join("Working_data", "reconstructed_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + "singleNN.npy")
        z_savename = os.path.join("Working_data", "reduced_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + "singleNN.npy")
        np.save(reconstruction_savename, reconstruction)
        np.save(z_savename, z)