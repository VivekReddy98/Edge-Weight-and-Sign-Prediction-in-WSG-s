import tensorflow as tf
import numpy as np
import sys
from src.weightSGCN import weightSGCN
from src.initialization import *


class sgcn():
	def __init__():
		pass

	def train():
		pass





a_new = np.reshape(np.arange(60), (5,2,3,2))
a = tf.Variable(a_new)
print(sess.run(a))
y = tf.slice(a, [0,0,1,0], [tf.shape(a)[0],tf.shape(a)[1],1,1])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(a)
	print(result.shape, result)
