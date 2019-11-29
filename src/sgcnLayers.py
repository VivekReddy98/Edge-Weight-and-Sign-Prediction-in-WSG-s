import tensorflow as tf
import numpy as np
import sys
from src.weightSGCN import weightSGCN
from src.initialization import *

class DS():
	def __init__(self, WB, WU, hB, hU, zUB, MLG):
		''' Input Tensors '''
		self.WB = WB
		self.WU = WU
		self.hB = hB
		self.hU = hU
		self.zUB = zUB
		self.MLG = MLG

		
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

	def call(self, h, WB, WU, start, end, adj_pos, adj_neg, **kwargs):

		''' For Balanced Sets '''
		mask_pos_neigh = tf.slice(adj_pos, [start,0], [end-start,tf.shape(adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_pos_neigh, 1)    # shape = b, 1
		# From the Embedding matrix of size N, d_out, L, 2, get embeddings of the given layer id of shape N, d_out.
		shp_h = tf.shape(h)
		embeddings_this_layer = tf.reshape(tf.slice(h, [0,0,self.Lid-1,0], [shp_h[0], shp_h[1], self.Lid, 1]), [shp_h[0], shp_h[1]])
		sum_pos_vectors = tf.matmul(mask_pos_neigh, embeddings_this_layer) # shape = b, d_out
		sum_pos_vectors = sum_pos_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		self_vectors_pos = tf.reshape(tf.slice(h, [start, 0, self.Lid-1, 0], [end-start, tf.shape(h)[1], self.Lid, 1]), [end-start, shp_h[1]]) #shape = b, d_in
		pos_vectors = tf.concat([sum_pos_vectors, self_vectors_pos], 1) #shape = b, 2*d_in
		tensor = h[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(pos_vectors, WB, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		h.assign(tensor)
  
		''' For Unbalanced Sets '''
		mask_neg_neigh = tf.slice(adj_neg, [start,0], [end-start,tf.shape(adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_neg_neigh, 1)    # shape = b, 1
		# From the Embedding matrix of size N, d_out, L, 2, get embeddings of the given layer id of shape N, d_out.
		shp_h = tf.shape(h)
		embeddings_this_layer_UB = tf.reshape(tf.slice(h, [0,0,self.Lid-1,1], [shp_h[0], shp_h[1], self.Lid, 1]), [shp_h[0], shp_h[1]])
		sum_neg_vectors = tf.matmul(mask_neg_neigh, embeddings_this_layer_UB) # shape = b, d_out
		sum_neg_vectors = sum_neg_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		self_vectors_neg = tf.reshape(tf.slice(h, [start, 0, self.Lid-1, 1], [end-start, tf.shape(h)[1], self.Lid, 1]), [end-start, shp_h[1]]) #shape = b, d_in
		neg_vectors = tf.concat([sum_neg_vectors, self_vectors_neg], 1) #shape = b, 2*d_in
		tensor = h[start:end, :, 0, 0].assign(tf.nn.relu(tf.matmul(neg_vectors, WU, transpose_b=True))) #shape = N, d_out, L
		tensor.eval()
		h.assign(tensor)
		return h


if __name__ == "__main__":

	tf.reset_default_graph()

	with tf.Session() as sess:
		adj_pos = np.random.randint(2, size=(5, 5))
		adj_neg = np.random.randint(2, size=(5, 5))
		adj_neg = adj_neg.astype('float32')  
		adj_pos = adj_pos.astype('float32')

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
		adj_pos = tf.constant(adj_pos)
		adj_neg = tf.constant(adj_neg)

		#d = DS(WB, WU, hB, hU, zUB, MLG)

		sess.run(tf.global_variables_initializer())

		for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
			print(i)   # i.name if you want just a name

		L0 = Layer1(WU0, WB0, h0)
		a = h.eval()
		#print(a)
		print("..........................................................")
		sess.run(L0.call(h, 0, 5, adj_pos, adj_neg))
		b = h.eval()
		#print(b)
		print("..........................Intermediate Layers..........................")
		L1 = LayerIntermediate(1)
		print(L1.name_)
		sess.run(L1.call(h, WB, WU, 0, 5, adj_pos, adj_neg))
		c = h.eval()
		print(np.nansum(a), np.nansum(b), np.nansum(c))



		