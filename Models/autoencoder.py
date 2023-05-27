import numpy as np
import os
import tensorflow as tf
from tensorflow import keras 
from keras.layers import Input, Lambda, Dropout, Dense, LeakyReLU, Layer, BatchNormalization, Concatenate, Softmax
from keras.models import Model
from keras.regularizers import l2,l1_l2,l1
from keras import backend as K
from keras import activations
from constants import FMRI_SIZE, GLOVE_SIZE, N_ROIS

#Files needed: sizes and reduced sizes
def identity(x):
    return x

class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name='bias',shape=self.dense.input_shape[-1],initializer='zeros')
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0],transpose_b=True)
        return self.activation(z+self.biases)


def autoencoder(trainable, mean):
    rate = 0.4
    latent_space_size = 200

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
    
    roi_concat_output = Concatenate()(roi_outputs)
    roi_output = BatchNormalization()(roi_concat_output)
    roi_output = Dropout(rate=rate)(roi_output)

    second_dense_layer = Dense(latent_space_size)
    second_dense_output_raw = second_dense_layer(roi_output)
    second_dense_output = LeakyReLU(alpha=0.3)(second_dense_output_raw)
    second_dense_output = BatchNormalization()(second_dense_output)
    second_dense_output = Dropout(rate=rate)(second_dense_output)

    regression_layer = Dense(GLOVE_SIZE, trainable=trainable)
    regression_output = regression_layer(second_dense_output)
    regression_output = LeakyReLU(alpha=0.3)(regression_output)
    regression_output = BatchNormalization()(regression_output)
    regression_output = Dropout(rate=rate)(regression_output)

    classification_output = Dense(N_ROIS, trainable=trainable, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005))(regression_output)

    # Decoder Part
    first_transpose_output = DenseTranspose(second_dense_layer)(second_dense_output)
    first_transpose_output = LeakyReLU(alpha=0.3)(first_transpose_output)
    first_transpose_output = BatchNormalization()(first_transpose_output)
    first_transpose_output = Dropout(rate=rate)(first_transpose_output)

    roi_transpose_outputs = []
    for j, (size, reduced) in enumerate(zip(size, reduced)):
        cur_input = Lambda(lambda x: x[:, index:index+size], output_shape=(reduced,))(first_transpose_output)
        
        cur_out = DenseTranspose(roi_dense_layers[j])(cur_input)
        cur_out = LeakyReLU(alpha=0.3)(cur_out)
        cur_out = BatchNormalization()(cur_out)

        roi_transpose_outputs.append(cur_out)
        index += size
    
    roi_transpose_output = Concatenate()(roi_transpose_outputs)

    concat_layer = Lambda(identity, name='concat_layer')(roi_concat_output)
    dense_layer = Lambda(identity, name='dense_layer')(second_dense_output_raw)
    glove_predictions = Lambda(identity, name='glove_predictions')(regression_output)
    classification_predictions = Lambda(identity, name='classification_predictions')(classification_output)
    fmri_output = Lambda(identity, name='fmri_output')(roi_transpose_output)

    if not mean:
        model = Model(inputs=[input_vector], outputs=[glove_predictions, classification_predictions, fmri_output])
    else:
        model = Model(inputs=[input_vector], outputs=[glove_predictions, classification_predictions, fmri_output, concat_layer, dense_layer])
    
    return model