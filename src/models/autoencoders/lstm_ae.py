"""
LSTM (Long Short Term Memory) Autoencoder

Explored as a substitute for the convolutional autoencoder
"""
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.utils.file_indexer import *
import changepoint
import sys
from src.utils.plotting_utils import set_font_size




def create_LSTM_model(patient_idx, time_steps, save_model=False, plot_loss=False):
    """
    Trains an LSTM model over a patient
    @param patient_idx: number
    @param time_steps: number of concatenated heartbeats per datapoint
    @param save_model: whether to save the model to h5 file
    @param plot_loss: whether to plot the loss during training
    @return:
    """
    orig_data = np.load(os.path.join("Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(patient_idx) + ".npy"))
    data = orig_data[0:1000, :, :]
    # print(data[0:10].reshape(10000,4).shape)
    X, y = create_lstm_datapoints(data, time_steps)

    model = Sequential()
    model.add(LSTM(30, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X.shape[1]))
    model.add(LSTM(30, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X.shape[2])))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(X, X, epochs=100, batch_size=1, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')],
                        shuffle=False)

    if save_model:
        model.save(f"Working_Data/LSTM_Model_Idx{patient_idx}.h5")

    if plot_loss:
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

    print(f"Created LSTM model for patient {patient_idx}")

    return model


def create_lstm_datapoints(data, time_steps):
    """
    Creates the data vectors for the LSTM autoencoder, which consist of "time_steps" concatenated heartbeats
    @param data: original patient data array
    @param time_steps: number of heartbeats to concatenate per data point
    @return: data vectors for the LSTM autoencoder
    """

    Xs, ys = [], []

    for i in range(max(len(data) - time_steps, 1)):
        Xs.append(data[i:(i + time_steps)].reshape(100*time_steps,4))
        # ys.append(data[i + time_steps].reshape(100,4))

    return np.array(Xs), np.array(ys)


def reconstruct_original_heartbeats(sequence_data, time_steps):
    """
    Given a sequence of lstm datapoints, recreate the original array of heartbeats
    @param time_steps: number of heartbeats per lstm datapoint
    @param sequence_data: sequence of lstm datapoints
    @return: original heartbeats
    """
    # preallocate for speed
    recreation = np.zeros([sequence_data.shape[0] + time_steps, 100, 4])
    for i in range(recreation.shape[0] - time_steps):
        recreation[i, :, :] = sequence_data[i, :100, :]
    recreation[-time_steps:, :, :] = sequence_data[-1, :, :].reshape(5, 100, 4)
    return recreation

def compute_cusum(patient_idx, correction):
    """
    Computes the cusum on the LSTM autoencoder
    ** Requires the reconstruction arrays of the LSTM to be saved to the Working_Data directory
    @param correction: correction parameter for cusum
    @param patient_idx: number
    @return:
    """
    return changepoint.cusum(patient_idx, "lstm", 100,  changepoint.kl_divergence, correction=correction, save=True, plot=True)


def compute_reconstruction(patient_idx, model, time_steps, save_reconstruction=False):
    """
    Computes a reconstruction by running patient data through the LSTM autoencoder
    @param save_reconstruction: whether to save reconstruction array
    @param time_steps: number of heartbeats per datapoint
    @param patient_idx: number
    @param model: LSTM model. If None is passed, the function attempts to load the model from memory
    @return: reconstruction numpy array
    """

    # attempt to load the model if no model is passed in
    if not model:
        model = keras.models.load_model(f"Working_Data/LSTM_Model_Idx{patient_idx}.h5")

    orig_data = np.load(os.path.join("Working_Data/Normalized_Fixed_Dim_HBs_Idx" + str(patient_idx) + ".npy"))
    originals = create_lstm_datapoints(orig_data, time_steps)[0]

    predictions = model.predict(originals)
    reconstructions = reconstruct_original_heartbeats(predictions, time_steps)
    np.save(f"Working_Data/reconstructed_lstm_{patient_idx}.npy", reconstructions)
    return reconstructions


def train_and_reconstruct():
    """
    Trains an LSTM model and computes the reconstruction for each patient
    @return: nothing
    """
    patients = get_patient_ids(control=False) + get_patient_ids(control=True)
    time_steps = 5
    for patient in patients:
        try:
            m = create_LSTM_model(patient, time_steps, save_model=True)
            compute_reconstruction(patient, m, time_steps, save_reconstruction=True)
            print(f"Completed reconstruction of {patient}, computing cusum")
            compute_cusum(patient, correction=0.2)
        except Exception as e:
            print(f"Training/Reconstruction could not be completed for patient {patient} because:\n {e}")
    return


if __name__ == "__main__":
    # patients = get_patient_ids(control=False)[:15] + get_patient_ids(control=True)[:5]
    # for patient in patients:
    #     try:
    #         compute_cusum(patient, 0.2)
    #     except Exception as e:
    #         print(e)


    train_and_reconstruct()
    pass

    ###################################
    # PROOF OF CONCEPT CODE (outdated)
    # def compute_cusum(originals, predictions):
    #     mses = []
    #     for i in range(originals.shape[0] - 5):
    #         mses.append(mean_squared_error(originals[i, :, :].flatten(), predictions[i, :, :].flatten()))
    #
    #     error_signal = mses
    #     correction = 0.04
    #
    #     val_data = error_signal[0:1000]  # third hour (assumed healthy)
    #     sigma = np.std(val_data)  # standard deviation of third hour
    #     c = np.mean(val_data) + correction  # cusum correction parameter
    #
    #     cusum = [0]
    #     for x in error_signal:
    #         L = (x - c) / sigma
    #         cusum.append(max(cusum[-1] + L, 0))
    #     cusum = cusum[1:]
    #     return cusum

    # originals = create_sequences(orig_data)[0]
    #
    # model = keras.models.load_model('Working_Data/lstm_model')
    # predictions = model.predict(originals)
    # cusum = compute_cusum(originals, predictions)
    #
    #
    # set_font_size()
    # # rcParams.update({'figure.autolayout': True})
    # plt.plot(cusum)
    # plt.title(f"CUSUM for Patient 1 from LSTM Model")
    # plt.xlabel("Sample")
    # plt.ylabel("CUSUM Score")
    #     # plt.savefig('images/cusum_single_patient.png', dpi=500)
    #
    # plt.show()