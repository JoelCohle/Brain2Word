import tensorflow as tf
from tensorflow import keras
from keras.losses import cosine_proximity, categorical_crossentropy
import math

def mean_distance_loss(y_true, y_pred):
    total = 0
    total_two = 0
    
    val = 179
    for i in range((val+1)):
        if i == 0:
            total += (val*cosine_proximity(y_true,y_pred))
        else:
            rolled = tf.manip.roll(y_pred, i, axis=0)
            total_two -= cosine_proximity(y_true,rolled)
    return total_two/val + total/val

initial_lrate = 0.001
epochs_drop = 10.0  
epochs=100
batch_size= 180

def step_decay(epoch):

   drop = 0.3       
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate