import keras, math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.metrics import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.clear_session()
K.set_session(sess)

def balanced_cross_entropy(y_true, y_pred, pos_weight=0.2):
    return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
    
def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def R_squared(y, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.div(residual, total))
    return r2

def mean_squared_logarithmic_error(y_true, y_pred):    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)    
    return K.mean(K.square(first_log - second_log), axis=-1)

def MLPClassifier(input_dim, output_dims, loss=balanced_cross_entropy, metrics = [Recall(), Precision(), Accuracy()], opt=Adam()):
    model = Sequential()
    model.add(Dense(int(input_dim/3), input_shape=(input_dim,), activation='sigmoid'))
    model.add(Dense(int(input_dim/6), activation='sigmoid'))
    model.add(Dense(output_dims, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def MLPRegressor(input_dim, output_dims, loss=mean_squared_error, metrics = [R_squared, mean_squared_logarithmic_error], opt=Adam()):
    model = Sequential()
    model.add(Dense(int(input_dim/2), input_shape=(input_dim,), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(int(input_dim/4), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(int(input_dim/6), activation='tanh'))
    model.add(Dense(int(input_dim/8), activation='tanh'))
    model.add(Dense(output_dims, activation='tanh'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model







