from src.initialization import glorot
import tensorflow as tf
import numpy as np

class weightSGCN():
	def __init__(self, Layers, Nodes, d_in, d_out):
		self.L = Layers
		self.N = Nodes
		self.d_in = d_in
		self.d_out = d_out
		self.trainable = True

	def weightsLayer1(self, name, variant='glorot'):
		''' Weights Defined for Layer 1: 3-D Tensor, shape: d_out*2*d_in'''
		shape = (self.d_out, 2*self.d_in)
		trainable = self.trainable
		print("TF Variable initilized with name: {}, Trainable: {}, shape: {}".format(name, str(trainable), shape))
		return glorot(shape, name=name)
		

	def weightsLayer1N(self, name, variant='glorot'):
		''' Weights defined for the layers : 3-D Tensor, shape: #d_out, 3*d_out, Layers '''
		shape = (self.d_out, 3*self.d_out, self.L)
		trainable = self.trainable
		print("TF Variable initilized with name: {}, Trainable: {}, shape: {}".format(name, str(trainable), shape))
		return glorot(shape, name=name)

	def initialEmbeddings(self, name, values):
		''' Initial Embeddings generated from Signed Spectral Embeddings shape: N, d_in'''
		size = (self.N, self.d_in)
		trainable = False
		print("TF Variable initilized with name: {}, Trainable: Definitely {}, shape: {}".format(name, str(trainable), size))
		if values.shape != size:
			raise Exception("Error is the size of input array given")
		#initial = tf.convert_to_tensor(values)
		return tf.constant(values, name=name)

	def interEmbeddings(self, name, variant='glorot', trainable = False):
		''' Intermedite Node Embeddings for all the layers: 3-D Tensor, shape:  #Nodes, d_out, (Create this for every Layer initiated) '''
		shape = (self.N, self.d_out)
		print("TF Variable initilized with name: {}, Trainable: {}, shape: {}".format(name, str(trainable), shape))
		
		with tf.compat.v1.variable_scope("embed"):
			return glorot(shape, name=name, trainable=trainable)
			
	def Embeddings(self, name, variant='glorot', trainable = False):
		''' Final Concantenated output Embeddings: 2-D Tensor, Shape: #Nodes, 2*d_out'''
		shape = (self.N, 2*self.d_out)
		trainable = False
		print("TF Variable initilized with name: {}, Trainable: {}, shape: {}".format(name, str(trainable), shape))
		with tf.compat.v1.variable_scope("embed"):
			return glorot(shape, name=name, trainable=trainable)

	def weightsMLG(self, name, variant='glorot'):
		''' Multinomial Logistic Regression Weights : 2-D tensor, shape = 3(+, -, ?), d_in'''
		shape = (3, 4*self.d_out)
		trainable = self.trainable
		print("TF Variable initilized with name: {}, Trainable: {}, shape: {}".format(name, str(trainable), shape))
		return glorot(shape, name=name)
	

