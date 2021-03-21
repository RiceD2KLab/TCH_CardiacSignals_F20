import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
from src.utils.plotting_utils import set_font_size




def create_model(X):
    model = Sequential()
    model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X.shape[1]))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X.shape[2])))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(X, X, epochs=100, batch_size=1, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')],
                        shuffle=False)

    model.save('Working_Data/lstm_model')
    # model.predict(X[0:10, :])

    # plot the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Working_Data/lstm_loss.png")
    plt.show()

    print("loss of the model is: ")
    print(history.history['loss'])


def create_sequences(data):
    Xs, ys = [], []
    time_steps = 5
    for i in range(max(len(data) - time_steps, 1)):
        Xs.append(data[i:(i + time_steps)].reshape(100*time_steps,4))
        # ys.append(data[i + time_steps].reshape(100,4))

    return np.array(Xs), np.array(ys)


def compute_cusum(originals, predictions):
    mses = []
    # pass
    for i in range(orig_data.shape[0] - 5):
        mses.append(mean_squared_error(originals[i, :, :].flatten(), predictions[i, :, :].flatten()))

    error_signal = mses
    correction = 0.04

    val_data = error_signal[0:1000]  # third hour (assumed healthy)
    sigma = np.std(val_data)  # standard deviation of third hour
    c = np.mean(val_data) + correction  # cusum correction parameter

    cusum = [0]
    for x in error_signal:
        L = (x - c) / sigma
        cusum.append(max(cusum[-1] + L, 0))
    cusum = cusum[1:]
    return cusum

if __name__ == "__main__":
    orig_data = np.load(os.path.join("Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(1) + ".npy"))
    data = orig_data[0:1000, :, :]
    # print(data[0:10].reshape(10000,4).shape)
    X, y = create_sequences(data)
    print(X.shape, y.shape)
    originals = create_sequences(orig_data)[0]

    model = keras.models.load_model('Working_Data/lstm_model')
    predictions = model.predict(originals)
    cusum = compute_cusum(predictions)


    set_font_size()
    # rcParams.update({'figure.autolayout': True})
    plt.plot(cusum)
    plt.title(f"Individual Patient: CUSUM statistic over time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
        # plt.savefig('images/cusum_single_patient.png', dpi=500)

    plt.show()