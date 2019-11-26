import tensorflow as tf
import numpy as np


def glorot(shape, name=None):
    """Glorot initialization """
    limit = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def uniform(shape, name, limit=0.05):
    """Uniform initialization of Weights."""
    initial = tf.random_uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros initialization (Not Desirable) """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)