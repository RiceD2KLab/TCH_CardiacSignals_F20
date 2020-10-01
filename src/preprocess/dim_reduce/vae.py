import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import minmax_scale

"""
Frank Yang
Last edited: 9/24/2020, by Frank Yang

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

            reconstruction_loss *= 100
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
    for lead_num in [0,1,2,3]:
        for latent_dim in [1]:
            num_epoch = 200
            learning_rate = 0.01

            if lead_num < 0 or lead_num > 3:
                sys.stderr.write("bad lead number - check for 1-indexing\n")

            # Build the encoder
            encoder_inputs = keras.Input(shape=(100,))
            x = layers.Dense(16, activation="relu")(encoder_inputs)
            x = layers.Dense(16, activation="relu")(x)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
            encoder.summary()

            # Build the decoder
            latent_inputs = keras.Input(shape=(latent_dim,))
            x = layers.Dense(32, activation="relu")(latent_inputs)
            decoder_outputs = layers.Dense(100, activation="relu")(x)
            decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
            decoder.summary()

            # Load heartbeat data
            data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))

            # Select a lead (0,1,2,3)
            lead_data = data[:, :, lead_num]

            # Normalize each heartbeat to (min=0,max=1). This is important when using ReLu in the NNs
            for iter1 in range(len(lead_data)):
                lead_data[iter1, :] = minmax_scale(lead_data[iter1, :], feature_range=(0, 1), axis=0, copy=True)

            vae = VAE(encoder, decoder)
            vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
            vae.fit(lead_data, epochs=num_epoch, batch_size=len(lead_data))

            z = encoder.predict(lead_data)
            reconstruction = decoder.predict(z)

            log_filepath = os.path.join("Working_data", "")
            os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

            reconstruction_savename = os.path.join("Working_data", "reconstructed_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + "_Lead" + str(lead_num) + ".npy")
            z_savename = os.path.join("Working_data", "reduced_vae_" + str(latent_dim) + "d_Idx" + str(file_index) + "_Lead" + str(lead_num) + ".npy")
            np.save(reconstruction_savename, reconstruction)
            np.save(z_savename, z)

plt.plot([i for i in range(len(lead_data[0, :]))], lead_data[0, :])
plt.plot([i for i in range(len(reconstruction[0, :]))], reconstruction[0, :])
plt.title("Real data (blue), VAE reconstruction (red)")
plt.show()
