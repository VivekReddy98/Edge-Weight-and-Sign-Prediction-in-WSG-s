import tensorflow as tf
import numpy as np
#tf.logging.set_verbosity(tf.logging.ERROR)

def glorot(size, name=None, trainable=True):
    """Glorot initialization """
    limit = np.sqrt(6.0/(size[0]+size[1]))
    initial = tf.compat.v1.random_uniform(shape=size, minval=-limit, maxval=limit, dtype=tf.float32)
    return tf.compat.v1.get_variable(name=name, initializer = initial, trainable=trainable)