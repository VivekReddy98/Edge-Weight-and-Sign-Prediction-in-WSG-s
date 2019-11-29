import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

def glorot(size, name=None):
    """Glorot initialization """
    limit = np.sqrt(6.0/(size[0]+size[1]))
    initial = tf.random_uniform(shape=size, minval=-limit, maxval=limit, dtype=tf.float32)
    return tf.get_variable(name=name, dtype=tf.float32, initializer = initial)

def uniform(size, name, limit=0.05):
    """Uniform initialization of Weights."""
    initial = tf.random_uniform(shape=size, minval=-limit, maxval=limit, dtype=tf.float32)
    return tf.get_variable(name=name, initializer = initial)

def zeros(size, name=None):
    """All zeros initialization (Not Desirable) """
    initial = tf.zeros(shape=size, dtype=tf.float32)
    return tf.get_variable(name=name, initializer = initial)