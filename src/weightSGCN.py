from src.initialization import glorot, uniform, zeros
import tensorflow as tf
import numpy as np

class weightSGCN():
	def __init__(self, Layers, Nodes, d_in, d_out):
		self.L = Layers
		self.N = Nodes
		self.d_in = d_in
		self.d_out = d_out

	def weightsLayer1(self, name, variant='glorot'):
		''' Weights Defined for Layer 1: 3-D Tensor, shape: d_out*2*d_in'''
		shape = (self.d_out, 2*self.d_in)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)

	def weightsLayer2N(self, name, variant='glorot'):
		''' Weights defined for the layers : 3-D Tensor, shape: #Layers, d_out, 3*d_out, '''
		shape = (self.L, self.d_out, 3*self.d_out)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)

	def initialEmbeddings(self, name, values):
		''' Initial Embeddings generated from Signed Spectral Embeddings shape: N, d_in'''
		size = (self.N, self.d_in)
		if values.shape != size:
			raise Exception("Error is the size of input array given")
		#initial = tf.convert_to_tensor(values)
		return tf.constant(values, name=name)

	def interEmbeddings(self, name, variant='glorot'):
		''' Intermedite Node Embeddings for all the layers: 3-D Tensor, shape:  #Nodes, d_out, #Layers, 2 (one for UB and B) '''
		shape = (self.N, self.d_out, self.L, 2)

		with tf.variable_scope("embed"):
			if variant=='glorot':
				return glorot(shape, name=name)
			elif variant=='uniform':
				return uniform(shape, name=name)
			else:
				return zeros(shape, name=name)

	def Embeddings(self, name, variant='glorot'):
		''' Final Concantenated output Embeddings: 2-D Tensor, Shape: #Nodes, 2*d_out'''
		shape = (self.N, 2*self.d_out)

		with tf.variable_scope("embed"):
			if variant=='glorot':
				return glorot(shape, name=name)
			elif variant=='uniform':
				return uniform(shape, name=name)
			else:
				return zeros(shape, name=name)


	def weightsMLG(self, name, variant='glorot'):
		''' Multinomial Logistic Regression Weights : 2-D tensor, shape = 3(+, -, ?), d_in'''
		shape = (3, 2*self.d_out)

		if variant=='glorot':
			return glorot(shape, name=name)
		elif variant=='uniform':
			return uniform(shape, name=name)
		else:
			return zeros(shape, name=name)


# if __name__ == "__main__":
# 	sess = tf.Session() 
# 	init_op = tf.global_variables_initializer()
# 	tf.logging.set_verbosity(tf.logging.ERROR)
# 	init = weightSGCN(8, 1000, 30, 10)
# 	values = np.zeros((1000, 30))

# 	WU0 = init.weightsLayer1(name="Weights_firstLayer")
# 	WB0 = init.weightsLayer1(name="Weights_firstLayer")
# 	h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
# 	WB = init.weightsLayer2N(name="Weights_Balanced")
# 	WU = init.weightsLayer2N(name='Weights_Unbalanced')
# 	hB = init.interEmbeddings(name='Embeddings_Balanced')
# 	hU = init.interEmbeddings(name='Embeddings_Unbalanced')
# 	zUB = init.Embeddings(name='Concat_Embeddings')
# 	MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')

# 	with tf.Session() as sess:
# 		sess.run(init_op)
# 		print(sess.run(tf.shape(WU0)))
# 		print(sess.run(tf.shape(WB0)))
# 		print(sess.run(tf.shape(h0)))
# 		print(sess.run(tf.shape(WB)))
# 		print(sess.run(tf.shape(WU)))
# 		print(sess.run(tf.shape(hB)))	
# 		print(sess.run(tf.shape(hU)))
# 		print(sess.run(tf.shape(zUB)))
# 		print(sess.run(tf.shape(MLG)))

