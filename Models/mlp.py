from tensorflow import keras 
from keras.layers import Input, Lambda, Dropout, Dense, LeakyReLU, BatchNormalization, Softmax
from keras.models import Model
from keras.regularizers import l2,l1_l2,l1
from constants import FMRI_SIZE, GLOVE_SIZE, N_CONCEPTS

def identity(x):
    return x

def mlp():
    first_layer_output_size = 2000
    rate = 0.4

    input_vector = Input(shape=(FMRI_SIZE,))

    first_dense_layer = Dense(first_layer_output_size)
    first_dense_output = first_dense_layer(input_vector)
    first_dense_output = LeakyReLU(alpha=0.3)(first_dense_output)
    first_dense_output = BatchNormalization()(first_dense_output)
    first_dense_output = Dropout(rate=rate)(first_dense_output)

    regression_layer = Dense(GLOVE_SIZE)
    regression_output = regression_layer(first_dense_output)

    classification_layer = Dense(N_CONCEPTS)
    classification_output = classification_layer(first_dense_output)
    classification_output = Softmax()(classification_output)

    glove_predictions = Lambda(identity, name='glove_predictions')(regression_output)
    classification_predictions = Lambda(identity, name='classification_predictions')(classification_output)

    model = Model(inputs=[input_vector], outputs=[glove_predictions, classification_predictions])
    return model