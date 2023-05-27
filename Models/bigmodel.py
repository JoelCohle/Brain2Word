import numpy as np
import os 
from tensorflow import keras 
from keras.layers import Input, Lambda, Dropout, Dense, LeakyReLU, BatchNormalization, Softmax, Concatenate
from keras.models import Model
from keras.regularizers import l2,l1_l2,l1
from constants import FMRI_SIZE, GLOVE_SIZE, N_ROIS, N_CONCEPTS

def identity(x):
    return x

def bigmodel():
    rate = 0.4

    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/Data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/Data/look_ups/reduced_sizes.npy')

    input_vector = Input(shape=(FMRI_SIZE,))

    roi_outputs = []
    roi_dense_layers = []
    index = 0

    for size, reduced in zip(sizes, reduced):
        cur_input = Lambda(lambda x: x[:, index:index+size])(input_vector)
        
        output_layer = Dense(reduced)
        cur_out = output_layer(cur_input)
        cur_out = LeakyReLU(alpha=0.3)(cur_out)
        cur_out = BatchNormalization()(cur_out)

        roi_dense_layers.append(output_layer)
        roi_outputs.append(cur_out)
        index += size

    roi_output = Concatenate()(roi_outputs)
    first_dense_output = Dropout(rate=rate)(roi_output)

    regression_layer = Dense(GLOVE_SIZE)
    regression_output = regression_layer(first_dense_output)

    classification_layer = Dense(N_CONCEPTS)
    classification_output = classification_layer(first_dense_output)
    classification_output = Softmax()(classification_output)

    glove_predictions = Lambda(identity, name='glove_predictions')(regression_output)
    classification_predictions = Lambda(identity, name='classification_predictions')(classification_output)

    model = Model(inputs=[input_vector], outputs=[glove_predictions, classification_predictions])
    return model