import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, GaussianNoise, MaxPooling1D, UpSampling1D


data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx" + str(1) + ".npy"))

model = keras.models.Sequential()
model.add(Conv1D(1,kernel_size=5, input_shape=(100,4)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mse', optimizer='adam')
model.summary()

input_window = Input(shape=(100,4))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 1, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(data,data,
                epochs=10,
                batch_size=46364)

