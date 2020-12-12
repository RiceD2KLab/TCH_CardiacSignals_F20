from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow import keras
import pydot
from tensorflow.keras.utils import model_to_dot
import pydotplus
from tensorflow.keras.utils import plot_model

sig_shape = (100,4)
encode_size = 10

encoder = Sequential()
encoder.add(InputLayer(sig_shape))
encoder.add(Flatten())
# encoder.add(Dense(350, activation = 'tanh'))
encoder.add(Dense(200, activation = 'tanh', kernel_initializer='normal'))
encoder.add(Dense(100, activation = 'tanh', kernel_initializer='normal'))
encoder.add(Dense(25, activation = 'tanh', kernel_initializer='normal'))
encoder.add(Dense(encode_size))

# Decoder

decoder = Sequential()
decoder.add(InputLayer((encode_size,)))
decoder.add(Dense(25, activation = 'tanh',kernel_initializer='normal'))
decoder.add(Dense(100, activation = 'tanh',kernel_initializer='normal'))
decoder.add(Dense(200, activation = 'tanh',kernel_initializer='normal'))
# decoder.add(Dense(350, activation = 'tanh'))
decoder.add(Dense(np.prod(sig_shape), activation = 'linear'))
decoder.add(Reshape(sig_shape))

plot_model(encoder, to_file = 'encoder.png')