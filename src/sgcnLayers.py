import tensorflow as tf
import numpy as np
import sys
from src.weightSGCN import weightSGCN
from src.initialization import *

class DS():
	def __init__(self, adj_pos, adj_neg):
		''' Input Tensors '''
		self.adj_pos = tf.constant(adj_pos, name='Adj_Pos_Static')
		self.adj_neg = tf.constant(adj_neg, name='Adj_Neg_Static')

class Layer0(tf.keras.layers.Layer):
	def __init__(self, WU0, WB0, h0, adj_pos, adj_neg, **kwargs):
		super(Layer0, self).__init__()
		self.WU0 = WU0
		self.WB0 = WB0
		self.h0 = h0
		self.name_ = 'Immediate_Neighbours_using_precomputed_Embeddings'
		self.adj_pos = adj_pos
		self.adj_neg = adj_neg

	#https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/variable_scope/index.md (Understanding variable scopes)
	def call(self, inputs, start, end, **kwargs):
		''' Inputs are assumed to be Node Id's given as a 1-d list of form [[]]''' 
		''' kwargs should have two adj_matrices, with in and out nodes information '''
		''' ds object having all the required matrices '''

		self_vectors = tf.slice(self.h0, [start, 0], [end-start, tf.shape(self.h0)[-1]]) #shape = b, d_in
		''' For Balanced Sets '''
		mask_pos_neigh = tf.slice(self.adj_pos, [start,0], [end-start,tf.shape(adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_pos_neigh, 1)    # shape = b, 1
		sum_pos_vectors = tf.matmul(mask_pos_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_pos_vectors = sum_pos_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		pos_vectors = tf.concat([sum_pos_vectors, self_vectors], 1) #shape = b, 2*d_in
		tensor = inputs[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(pos_vectors, self.WB0, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		inputs.assign(tensor)
  
		''' For Unbalanced Sets '''
		mask_neg_neigh = tf.slice(self.adj_neg, [start,0], [end-start,tf.shape(adj_neg)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_neg_neigh, 1)    # shape = b, 1
		sum_neg_vectors = tf.matmul(mask_neg_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_neg_vectors = sum_neg_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		neg_vectors = tf.concat([sum_neg_vectors, self_vectors], 1) #shape = b, 2*d_in
		tensor = inputs[start:end, :, 0, 1].assign(tf.nn.relu(tf.matmul(neg_vectors, self.WU0, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		inputs.assign(tensor)
		return inputs

class LayerIntermediate(tf.keras.layers.Layer):
	def __init__(self, layer_id, WB, WU, adj_pos, adj_neg, **kwargs):
		super(LayerIntermediate, self).__init__()
		self.Lid = layer_id
		self.name_ = "Layer_" + str(self.Lid)
		self.WB = WB
		self.WU = WU
		self.adj_pos = adj_pos
		self.adj_neg = adj_neg


	def call(self, h, start, end, **kwargs):
		''' For Balanced Sets '''
		h = self.computeEmbeddings(h, self.WB, start, end, self.adj_pos, self.adj_neg, U=0)
  
		''' For Unbalanced Sets '''
		h = self.computeEmbeddings(h, self.WU, start, end, self.adj_pos, self.adj_neg, U=1)
		return h

	def computeEmbeddings(self, h, W, start, end, adj_pos, adj_neg, U=0):

		shp_h = tf.shape(h)

		self_embeddings = tf.reshape(tf.slice(h, [start, 0, self.Lid-1, U], [end-start, -1, 1, 1]), [end-start, tf.shape(h)[1]]) #shape = b, d_out
		
		# From the Embedding matrix of size N, d_out, L, 2, get embeddings of the given layer id-1 of shape N, d_out.

		# Positive Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_P = tf.slice(adj_pos, [start,0], [end-start,-1]) # shape = b, N where b = batch_size
		sum_neigh_P = tf.reduce_sum(mask_neigh_P, 1)    # shape = b, 1
		embeddings_this_layer_P = tf.reshape(tf.slice(h, [0,0,self.Lid-1,U], [-1, -1, 1, 1]), [tf.shape(h)[0], tf.shape(h)[1]]) # Unbalanced or Balanced based on the value of U
		sum_embeddings_P = tf.matmul(mask_neigh_P, embeddings_this_layer_P) / tf.reshape(sum_neigh_P, (-1, 1)) # shape = b, d_out 
		
		# Negative Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_N = tf.slice(adj_neg, [start,0], [end-start,-1]) # shape = b, N where b = batch_size
		sum_neigh_N = tf.reduce_sum(mask_neigh_N, 1)    # shape = b, 1
		embeddings_this_layer_N = tf.reshape(tf.slice(h, [0,0,self.Lid-1,int(not(U))], [-1, -1, 1, 1]), [tf.shape(h)[0], tf.shape(h)[1]]) # Unbalanced or Balanced based on the value of U
		sum_embeddings_N = tf.matmul(mask_neigh_N, embeddings_this_layer_N) / tf.reshape(sum_neigh_N, (-1, 1)) # shape = b, d_out

		# Concat the found vectors
		concat_vector = tf.concat([sum_embeddings_P, sum_embeddings_N, self_embeddings], 1) #shape = b, 3*d_out
		
		# Get the Weights Corresponding to this layer 
		sliced_weights = tf.reshape(tf.slice(W, [0, 0, self.Lid-1], [-1, -1, 1]), [tf.shape(W)[0], tf.shape(W)[1]])
		
		tensor = h[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(concat_vector, sliced_weights, transpose_b=True))) #shape = b, 3*d_out 
		tensor.eval()
		h.assign(tensor)
		return h

class LayerLast(tf.keras.layers.Layer):
	def __init__(self, layer_id):
		super(LayerLast, self).__init__()
		self.Lid = layer_id
		self.name_ = "End Layer_" + str(self.Lid)

	def call(self, h, **kwargs):
		z_B = tf.reshape(tf.slice(h, [0, 0, self.Lid, 0], [-1, -1, 1, 1]), [tf.shape(h)[0], tf.shape(h)[1]])
		z_U = tf.reshape(tf.slice(h, [0, 0, self.Lid, 1], [-1, -1, 1, 1]), [tf.shape(h)[0], tf.shape(h)[1]])
		zUB = tf.concat([z_B, z_U], axis=1)
		return zUB

# if __name__ != "__main__":
# 	tf.reset_default_graph()

# 	a_new = np.reshape(np.arange(60), (5,2,3,2))
# 	a = tf.Variable(a_new)
# 	y = tf.slice(a, [0,0,2,0], [-1,-1,1,1])

# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		result = sess.run(a)
# 		print(result.shape)
# 		result2 = sess.run(y)
# 		print(result2.shape)

if __name__ == "__main__":

	with tf.Session() as sess:
	
		init = weightSGCN(3, 5, 5, 2)
		values = np.ones((5, 5))
		values = values.astype('float32')

		WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)

		WB = init.weightsLayer1N(name="Weights_Balanced")
		WU = init.weightsLayer1N(name='Weights_Unbalanced')
		h = init.interEmbeddings(name='Embeddings_B_UB')
		zUB = init.Embeddings(name='Concat_Embeddings')
		MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')

		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')

		d = DS(adj_pos, adj_neg)

		sess.run(tf.global_variables_initializer())

		a = h.eval()

		print("........................First Layer...................................")
		L0 = Layer0(WU0, WB0, h0, d.adj_pos, d.adj_neg)
		print(L0.name_)
		sess.run(L0.call(h, 0, 5))
		b = h.eval()
		
		print("..........................Intermediate Layer 1..........................")
		
		L1 = LayerIntermediate(1)
		print(L1.name_)
		sess.run(L1.call(h, WB, WU, 0, 5, d.adj_pos, d.adj_neg))

		print("..........................Intermediate Layer 2..........................")

		L2 = LayerIntermediate(2)
		print(L2.name_)
		sess.run(L2.call(h, WB, WU, 0, 5, d.adj_pos, d.adj_neg))
		c = h.eval()

		print("..........................Final Layer..........................")
		L3 = LayerLast(2)
		print(L3.name_)
		sess.run(L3.call(h, zUB))
		g = zUB.eval()
		print("Shape of zUB is {}".format(tf.shape(zUB)))


		print("..........................Correctness Checking..........................")
		print("Scores of h: ")
		print(np.nansum(a), np.nansum(b), np.nansum(c), np.nansum(g), np.shape(g))



		