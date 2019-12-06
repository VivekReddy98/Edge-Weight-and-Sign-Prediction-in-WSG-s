import tensorflow as tf
import numpy as np
import sys, os
from src.weightSGCN import weightSGCN
from src.initialization import *

class DS():
	def __init__(self, adj_pos, adj_neg):
		''' Input Tensors '''
		self.adj_pos = tf.constant(adj_pos, name='Adj_Pos_Static')
		self.adj_neg = tf.constant(adj_neg, name='Adj_Neg_Static')

class Layer0():
	def __init__(self, h0, WU0, WB0, adj_pos, adj_neg, **kwargs):
		super(Layer0, self).__init__()
		self.WU0 = WU0
		self.WB0 = WB0
		self.h0 = h0
		# self.hU1 = hU1 # Embeddings of Layer 1, (unBalanced) Cause these are the Embeddings which are suppoed to be updated.
		# self.hB1 = hB1 # Embeddings of Layer 1, (Balanced) Cause these are the Embeddings which are suppoed to be updated.
		self.name_ = 'Immediate_Neighbours_using_precomputed_Embeddings'
		self.adj_pos = adj_pos
		self.adj_neg = adj_neg
		print("Layer with Name {} initialized ".format(self.name_))

	#https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/variable_scope/index.md (Understanding variable scopes)
	def call(self, H, start, end, **kwargs): 
		# H: The Data Structure containing UB and B embeddings in a list form. 
		''' Inputs are assumed to be Node Id's given as a 1-d list of form [[]]''' 
		''' kwargs should have two adj_matrices, with in and out nodes information '''
		''' ds object having all the required matrices '''
	
		range_elements = tf.math.subtract(end, start)

		range_indices = tf.range(start, end, 1)

		self_vectors = tf.slice(self.h0, [start, 0], [range_elements, -1]) #shape = b, d_in
		
		''' For Balanced Sets '''
		mask_pos_neigh = tf.slice(self.adj_pos, [start, 0], [range_elements, tf.shape(self.adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_pos_neigh, 1)    # shape = b, 1
		print(mask_pos_neigh, self.h0, self.adj_pos)
		sum_pos_vectors = tf.matmul(mask_pos_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_pos_vectors = sum_pos_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		sum_pos_vectors = tf.where(tf.math.is_nan(sum_pos_vectors), tf.zeros(tf.shape(sum_pos_vectors)), sum_pos_vectors)
		pos_vectors = tf.concat([sum_pos_vectors, self_vectors], 1) #shape = b, 2*d_in
		
  
		''' For Unbalanced Sets '''
		mask_neg_neigh = tf.slice(self.adj_neg, [start,0], [range_elements, tf.shape(self.adj_neg)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_neg_neigh, 1)    # shape = b, 1
		sum_neg_vectors = tf.matmul(mask_neg_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_neg_vectors = sum_neg_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		sum_neg_vectors = tf.where(tf.math.is_nan(sum_neg_vectors), tf.zeros(tf.shape(sum_neg_vectors)), sum_neg_vectors)
		neg_vectors = tf.concat([sum_neg_vectors, self_vectors], 1) #shape = b, 2*d_in

		indices = tf.reshape(tf.range(start, end, delta=1, dtype=tf.int32), (-1, 1))
		H.B[0] = tf.compat.v1.scatter_nd_update(H.B[0], indices, tf.nn.relu(tf.matmul(pos_vectors, self.WB0, transpose_b=True)))
		H.U[0] = tf.compat.v1.scatter_nd_update(H.U[0], indices, tf.nn.relu(tf.matmul(neg_vectors, self.WU0, transpose_b=True)))
	
		#with tf.compat.v1.variable_scope("Embeddings_B_UB"):
		# tensor = inputs[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(pos_vectors, self.WB0, transpose_b=True))) #shape = N, d_out, L
		# inputs = tf.Variable(tensor)
		# tensor = inputs[start:end, :, 0, 1].assign(tf.nn.relu(tf.matmul(neg_vectors, self.WU0, transpose_b=True))) #shape = N, d_out, L
		# inputs = tf.Variable(tensor)
		return H

class LayerIntermediate():
	def __init__(self, layer_id, WB, WU, adj_pos, adj_neg, **kwargs):
		# H: The Data Structure containing UB and B embeddings in a list form.
		super(LayerIntermediate, self).__init__()
		self.Lid = layer_id
		self.name_ = "Layer_" + str(self.Lid)
		self.WB = WB
		self.WU = WU
		self.adj_pos = adj_pos
		self.adj_neg = adj_neg
		print("Layer with Name {} initialized ".format(self.name_))

	def call(self, H, start, end, **kwargs):
		''' For Balanced Sets '''
		H.B = self.computeEmbeddings_pos(H, self.WB, start, end, self.adj_pos, self.adj_neg)
  
		''' For Unbalanced Sets '''
		H.U = self.computeEmbeddings_neg(H, self.WU, start, end, self.adj_pos, self.adj_neg)
		return H

	def computeEmbeddings_pos(self, H, W, start, end, adj_pos, adj_neg):
		# H: The Data Structure containing UB and B embeddings in a list form. 
		range_elements = tf.math.subtract(end, start)
		range_indices = tf.range(start, end, 1)

		self_embeddings = tf.slice(H.B[self.Lid-1], [start, 0], [range_elements, -1]) #shape = b, d_out

		# Positive Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_P = tf.slice(adj_pos, [start,0], [range_elements,-1]) # shape = b, N where b = batch_size
		sum_neigh_P = tf.reduce_sum(mask_neigh_P, 1)    # shape = b, 1
		sum_embeddings_P = tf.matmul(mask_neigh_P, H.B[self.Lid-1]) / tf.reshape(sum_neigh_P, (-1, 1)) # shape = b, d_out 
		sum_embeddings_P = tf.where(tf.math.is_nan(sum_embeddings_P), tf.zeros(tf.shape(sum_embeddings_P)), sum_embeddings_P)

		# Negative Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_N = tf.slice(adj_neg, [start,0], [range_elements,-1]) # shape = b, N where b = batch_size
		sum_neigh_N = tf.reduce_sum(mask_neigh_N, 1)    # shape = b, 1
		sum_embeddings_N = tf.matmul(mask_neigh_N, H.U[self.Lid-1]) / tf.reshape(sum_neigh_N, (-1, 1)) # shape = b, d_out
		sum_embeddings_N = tf.where(tf.math.is_nan(sum_embeddings_N), tf.zeros(tf.shape(sum_embeddings_N)), sum_embeddings_N)
		# Concat the found vectors
		concat_vector = tf.concat([sum_embeddings_P, sum_embeddings_N, self_embeddings], 1) #shape = b, 3*d_out
		
		# Get the Weights Corresponding to this layer 
		sliced_weights = tf.reshape(tf.slice(W, [0, 0, self.Lid-1], [-1, -1, 1]), [tf.shape(W)[0], tf.shape(W)[1]])
		#with tf.compat.v1.variable_scope("Embeddings_B_UB"):
		
		indices = tf.reshape(tf.range(start, end, delta=1, dtype=tf.int32), (-1, 1))
		H.B[self.Lid] = tf.compat.v1.scatter_nd_update(H.B[self.Lid], indices, tf.nn.relu(tf.matmul(concat_vector, sliced_weights, transpose_b=True)))
		return H.B
	
	def computeEmbeddings_neg(self, H, W, start, end, adj_pos, adj_neg):
		# H: The Data Structure containing UB and B embeddings in a list form. 
		range_elements = tf.math.subtract(end, start)
		range_indices = tf.range(start, end, 1)

		self_embeddings = tf.slice(H.U[self.Lid-1], [start, 0], [range_elements, -1]) #shape = b, d_out
		
		# From the Embedding matrix of size N, d_out, L, 2, get embeddings of the given layer id-1 of shape N, d_out.

		# Positive Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_P = tf.slice(adj_pos, [start,0], [range_elements,-1]) # shape = b, N where b = batch_size
		sum_neigh_P = tf.reduce_sum(mask_neigh_P, 1)    # shape = b, 1
		sum_embeddings_P = tf.matmul(mask_neigh_P, H.U[self.Lid-1]) / tf.reshape(sum_neigh_P, (-1, 1)) # shape = b, d_out 
		sum_embeddings_P = tf.where(tf.math.is_nan(sum_embeddings_P), tf.zeros(tf.shape(sum_embeddings_P)), sum_embeddings_P)
		# Negative Neighbours, (Balanced or Unbalanced Weights depends on the value of U)
		mask_neigh_N = tf.slice(adj_neg, [start,0], [range_elements,-1]) # shape = b, N where b = batch_size
		sum_neigh_N = tf.reduce_sum(mask_neigh_N, 1)    # shape = b, 1
		sum_embeddings_N = tf.matmul(mask_neigh_N, H.B[self.Lid-1]) / tf.reshape(sum_neigh_N, (-1, 1)) # shape = b, d_out
		sum_embeddings_N = tf.where(tf.math.is_nan(sum_embeddings_N), tf.zeros(tf.shape(sum_embeddings_N)), sum_embeddings_N)
		# Concat the found vectors
		concat_vector = tf.concat([sum_embeddings_P, sum_embeddings_N, self_embeddings], 1) #shape = b, 3*d_out
		
		# Get the Weights Corresponding to this layer 
		sliced_weights = tf.reshape(tf.slice(W, [0, 0, self.Lid-1], [-1, -1, 1]), [tf.shape(W)[0], tf.shape(W)[1]])
		#with tf.compat.v1.variable_scope("Embeddings_B_UB"):
		
		indices = tf.reshape(tf.range(start, end, delta=1, dtype=tf.int32), (-1, 1))
		H.U[self.Lid] = tf.compat.v1.scatter_nd_update(H.U[self.Lid], indices, tf.nn.relu(tf.matmul(concat_vector, sliced_weights, transpose_b=True)))
		return H.U

class LayerLast():
	def __init__(self, layer_id):
		super(LayerLast, self).__init__()
		self.Lid = layer_id
		self.name_ = "End Layer_" + str(self.Lid)
		print("Layer with Name {} initialized ".format(self.name_))

	def call(self, H, **kwargs):
		# H: The Data Structure containing UB and B embeddings in a list form.
		#indices = tf.reshape(tf.range(start, end, delta=1, dtype=tf.int32), (-1, 1))
		zUB = tf.concat([H.B[self.Lid-1], H.U[self.Lid-1]], axis=1)
		#zUB = tf.compat.v1.scatter_nd_update(zUB, indices, tf.nn.relu(tf.matmul(concat_vector, sliced_weights, transpose_b=True)))
		return zUB


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
		L0 = Layer0(WU0, WB0,self.h0, d.adj_pos, d.adj_neg)
		print(L0.name_)
		sess.run(L0.call(h, 0, 5))
		b = h.eval()
		
		print("..........................Intermediate Layer 1..........................")
		
		L1 = LayerIntermediate(1, WB, WU, d.adj_pos, d.adj_neg)
		print(L1.name_)
		sess.run(L1.call(h, 0, 5))

		print("..........................Intermediate Layer 2..........................")

		L2 = LayerIntermediate(2, WB, WU, d.adj_pos, d.adj_neg)
		print(L2.name_)
		sess.run(L2.call(h, 0, 5))
		c = h.eval()

		print("..........................Final Layer..........................")
		L3 = LayerLast(2)
		print(L3.name_)
		sess.run(L3.call(h))
		g = zUB.eval()
		print("Shape of zUB is {}".format(tf.shape(zUB)))


		print("..........................Correctness Checking..........................")
		print("Scores of h: ")
		print(np.nansum(a), np.nansum(b), np.nansum(c), np.nansum(g), np.shape(g))	