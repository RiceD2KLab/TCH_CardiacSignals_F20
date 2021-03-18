import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

data = np.load(os.path.join("Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(1) + ".npy"))
data = data[0:500, :, :]
# print(data[0:10].reshape(10000,4).shape)

def create_sequences(data):
    Xs, ys = [], []
    time_steps = 10
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)].reshape(100*time_steps,4))
        ys.append(data[i + time_steps].reshape(100,4))

    return np.array(Xs), np.array(ys)

X, y = create_sequences(data)
print(X.shape, y.shape)


model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X.shape[1]))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X.shape[2])))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X, X, epochs=2, batch_size=1, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')], shuffle=False)

# plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

