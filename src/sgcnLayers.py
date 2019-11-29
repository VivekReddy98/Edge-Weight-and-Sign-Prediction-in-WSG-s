import tensorflow as tf
import numpy as np
import sys
from src.weightSGCN import weightSGCN
from src.initialization import *

class DS():
	def __init__(self, adj_pos, adj_neg):
		''' Input Tensors '''
		self.adj_pos = tf.Variable(adj_pos, name='Adj_Pos')
		self.adj_neg = tf.Variable(adj_neg, name='Adj_Neg')

		
class Layer1(tf.keras.layers.Layer):
	def __init__(self, WU0, WB0, h0, trainable=True, name='Immediate_Neighbours', **kwargs):
		super(Layer1, self).__init__()
		self.WU0 = WU0
		self.WB0 = WB0
		self.h0 = h0

	#https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/variable_scope/index.md (Understanding variable scopes)
	def call(self, inputs, start, end, adj_pos, adj_neg, **kwargs):
		''' Inputs are assumed to be Node Id's given as a 1-d list of form [[]]''' 
		''' kwargs should have two adj_matrices, with in and out nodes information '''
		''' ds object having all the required matrices '''

		self_vectors = tf.slice(self.h0, [start, 0], [end-start, tf.shape(self.h0)[-1]]) #shape = b, d_in
		''' For Balanced Sets '''
		mask_pos_neigh = tf.slice(adj_pos, [start,0], [end-start,tf.shape(adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_pos_neigh, 1)    # shape = b, 1
		sum_pos_vectors = tf.matmul(mask_pos_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_pos_vectors = sum_pos_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		pos_vectors = tf.concat([sum_pos_vectors, self_vectors], 1) #shape = b, 2*d_in
		tensor = inputs[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(pos_vectors, self.WB0, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		inputs.assign(tensor)
  
		''' For Unbalanced Sets '''
		mask_neg_neigh = tf.slice(adj_neg, [start,0], [end-start,tf.shape(adj_neg)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_neg_neigh, 1)    # shape = b, 1
		sum_neg_vectors = tf.matmul(mask_neg_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_neg_vectors = sum_neg_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		neg_vectors = tf.concat([sum_neg_vectors, self_vectors], 1) #shape = b, 2*d_in
		tensor = inputs[start:end, :, 0, 1].assign(tf.nn.relu(tf.matmul(neg_vectors, self.WU0, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		inputs.assign(tensor)
		return inputs

class LayerIntermediate(tf.keras.layers.Layer):
	def __init__(self, layer_id, trainable=True, **kwargs):
		super(LayerIntermediate, self).__init__()
		self.Lid = layer_id
		self.name_ = "Layer_" + str(self.Lid) + "" + str(layer_id-1) + " :away neighbours" 

	def call(self, h, WB, WU, start, end, d, **kwargs):

		# Lth neighbor information (cause A^2 given information on two-hop neighbours, assuming adj already has L-1 neighbour information)
		adj_L_pos = tf.matmul(d.adj_pos, d.adj_pos)
		adj_L_neg = tf.matmul(d.adj_neg, d.adj_neg) 

		''' For Balanced Sets '''
		h = self.computeEmbeddings(h, WB, start, end, d.adj_pos, adj_L_neg, U=0)
  
		''' For Unbalanced Sets '''
		h = self.computeEmbeddings(h, WU, start, end, d.adj_neg, adj_L_pos, U=1)


		return h

	def computeEmbeddings(self, h, W, start, end, adj, adj_L, U=0):

		shp_h = tf.shape(h)

		self_embeddings = tf.reshape(tf.slice(h, [start, 0, self.Lid-1, 0], [end-start, tf.shape(h)[1], self.Lid, 1]), [end-start, shp_h[1]]) #shape = b, d_out
		# From the Embedding matrix of size N, d_out, L, 2, get embeddings of the given layer id of shape N, d_out.

		# L-1 hop Neighbours
		mask_neigh_L_1 = tf.slice(adj, [start,0], [end-start,tf.shape(adj)[-1]]) # shape = b, N where b = batch_size
		sum_neigh_L_1 = tf.reduce_sum(mask_neigh_L_1, 1)    # shape = b, 1
		embeddings_this_layer = tf.reshape(tf.slice(h, [0,0,self.Lid-1,U], [shp_h[0], shp_h[1], self.Lid, 1]), [shp_h[0], shp_h[1]]) # Unbalanced or Balanced based on the value of U
		sum_embeddings_L_1 = tf.matmul(mask_neigh_L_1, embeddings_this_layer) / tf.reshape(sum_neigh_L_1, (-1, 1)) # shape = b, d_out L-1 hop neighbours 
		
		# L hop Neighbours
		mask_neigh_L = tf.slice(adj_L, [start,0], [end-start,tf.shape(adj)[-1]]) # shape = b, N where b = batch_size
		sum_neigh_L = tf.reduce_sum(mask_neigh_L, 1)    # shape = b, 1
		embeddings_this_layer = tf.reshape(tf.slice(h, [0,0,self.Lid-1,int(not(U))], [shp_h[0], shp_h[1], self.Lid, 1]), [shp_h[0], shp_h[1]]) # Unbalanced or Balanced based on the value of U
		sum_embeddings_L = tf.matmul(mask_neigh_L, embeddings_this_layer) / tf.reshape(sum_neigh_L, (-1, 1)) # shape = b, d_out L-1 hop neighbours 

		# Concat the found vectors
		concat_vector = tf.concat([sum_embeddings_L_1, sum_embeddings_L, self_embeddings], 1) #shape = b, 3*d_out
		
		# Get the Weights Corresponding to this layer 
		sliced_weights = tf.reshape(tf.slice(W, [0, 0, self.Lid-1], [tf.shape(W)[0], tf.shape(W)[1], self.Lid]), [tf.shape(W)[0], tf.shape(W)[1]])
		
		tensor = h[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(concat_vector, sliced_weights, transpose_b=True))) #shape = b, 3*d_out 
		tensor.eval()
		h.assign(tensor)
		return h


if __name__ == "__main__":

	tf.reset_default_graph()

	with tf.Session() as sess:
	
		init = weightSGCN(2, 5, 5, 2)
		values = np.ones((5, 5))
		values = values.astype('float32')

		WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)

		WB = init.weightsLayer2N(name="Weights_Balanced")
		WU = init.weightsLayer2N(name='Weights_Unbalanced')
		h = init.interEmbeddings(name='Embeddings_B_UB')
		zUB = init.Embeddings(name='Concat_Embeddings')
		MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')

		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')

		d = DS(adj_pos, adj_neg)

		sess.run(tf.global_variables_initializer())

		for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
			print(i)   # i.name if you want just a name

		L0 = Layer1(WU0, WB0, h0)
		a = h.eval()
		#print(a)
		print("..........................................................")
		sess.run(L0.call(h, 0, 5, d.adj_pos, d.adj_neg))
		b = h.eval()
		adj_p = d.adj_pos.eval()
		adj_n = d.adj_neg.eval()
		print("Adj_Matrices after first Layer: \n")
		print(adj_p, adj_n)
		print("..........................Intermediate Layers..........................")
		L1 = LayerIntermediate(1)
		print(L1.name_)
		sess.run(L1.call(h, WB, WU, 0, 5, d))
		c = h.eval()
		print("Scores of h: ")
		print(np.nansum(a), np.nansum(b), np.nansum(c))

		print("Adj_Matrices after intermediate Layer")
		d.adj_pos = tf.matmul(d.adj_pos, d.adj_pos)
		d.adj_neg = tf.matmul(d.adj_neg, d.adj_neg)
		adj_p = d.adj_pos.eval()
		adj_n = d.adj_neg.eval()
		print(adj_p, adj_n)



		