from src.initialization import glorot, uniform, zeros
import tensorflow as tf
import numpy as np

class weightSGCN():
	def __init__(self, Layers, Nodes, d_in, d_out):
		self.L = Layers
		self.N = Nodes
		self.d_in = d_in
		self.d_out = d_out

	def weightsLayer(self, name, variant='glorot'):
		''' Weights defined for the layers : 3-D Tensor, shape = A list of size 3 i.e. d_out, 3*d_in, #Layers'''
		shape = (self.d_out, 3*self.d_in, self.L)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)

	def embeddings(self, name, variant='glorot'):
		''' Node Embeddings : 3-D Tensor, shape = A list of size 3 i.e. #Nodes, d_in, #Layers '''
		shape = (self.N, self.d_in, self.L)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)

	def weightsMLG(self, name, variant='glorot'):
		''' Multinomial Logistic Regression Weights : 2-D tensor, shape = list of size 2 3(+, -, ?), d_in'''
		shape = (3, 2*self.d_in)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)


if __name__ == "__main__":
	sess = tf.Session() 
	init_op = tf.global_variables_initializer()
	tf.logging.set_verbosity(tf.logging.ERROR)
	init = weightSGCN(8, 1000, 10, 20)
	WB = init.weightsLayer(name="Weights_Balanced")
	WU = init.weightsLayer(name='Weights_Unbalanced')
	hB = init.embeddings(name='Embeddings_Balanced')
	hU = init.embeddings(name='Embeddings_Unbalanced')
	MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')
	with tf.Session() as sess:
		sess.run(init_op)
		print(sess.run(tf.shape(WB)))
		print(sess.run(tf.shape(WU)))
		print(sess.run(tf.shape(hB)))	
		print(sess.run(tf.shape(hU)))
		print(sess.run(tf.shape(MLG)))

