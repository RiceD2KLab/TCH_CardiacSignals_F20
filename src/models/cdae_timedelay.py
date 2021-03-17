import os
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv1D, MaxPooling1D, Conv1DTranspose
from tensorflow.keras.models import Sequential, Model
from src.models.patient_split import *
from sklearn.model_selection import train_test_split
from src.models.conv_denoising_ae import *

# train a model, save reconstruction and then move to next time chunk training and reconstruction
# if __name__ == "__main__":

