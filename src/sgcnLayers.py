import tensorflow as tf

class DS():
	def __init__(self, WB, WU, hB, hU, zUB, MLG, adj_pos, adj_neg):
		''' Input Tensors '''
		self.WB = WB
		self.WU = WU
		self.hB = hB
		self.hU = hU
		self.zUB = zUB
		self.MLG = MLG

		
class Layer1(tf.keras.layers.Layer):
	def __init__(self, WU0, WB0, h0, batch_size, num_outputs, trainable=True, name='Immediate_Neighbours', **kwargs):
		super(Layer1, self).__init__()
		#self.num_outputs = num_outputs
		# Required Matrices for Layer 1 
 		self.WU0 = WU0 
		self.WB0 = WB0
		self.h0 = h0

	def call(self, d, start, end, adj_pos, adj_neg, **kwargs):
		''' Inputs are assumed to be Node Id's given as a 1-d list of form [[]]''' 
		''' kwargs should have two adj_matrices, with in and out nodes information '''
		''' ds object having all the required matrices '''

		self_vectors = tf.slice(self.h0, [start, 0], [end-start, tf.shape(self.h0)[-1]]) #shape = b, d_in

		''' For Balanced Sets '''
		mask_pos_neigh = tf.slice(adj_pos, [start,0], [end-start,tf.shape(adj_pos)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_pos_neigh, 0)    # shape = b, 1
		sum_pos_vectors = tf.matmul(mask_pos_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_pos_vectors = sum_pos_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		pos_vectors = tf.concat([sum_pos_vectors, self_vectors], 1) #shape = b, 2*d_in
		d.WB = tf.assign(d.WB[start:end, :, 0], tf.matmul(pos_vectors, self.WB0, transpose_b=True)) #shape = N, d_out, L

		''' For Unbalanced Sets '''

		mask_neg_neigh = tf.slice(adj_neg, [start,0], [end-start,tf.shape(adj_neg)[-1]]) # shape = b, N where b = batch_size
		sum_neigh = tf.reduce_sum(mask_neg_neigh, 0)    # shape = b, 1
		sum_neg_vectors = tf.matmul(mask_neg_neigh, self.h0, transpose_b=False) # shape = b, d_in
		sum_neg_vectors = sum_neg_vectors / tf.reshape(sum_neigh, (-1, 1)) # Shape = b, d_in
		neg_vectors = tf.concat([sum_neg_vectors, self_vectors], 1) #shape = b, 2*d_in
		d.WU = tf.assign(d.WU[start:end, :, 0], tf.matmul(neg_vectors, self.WU0, transpose_b=True)) #shape = N, d_out, L

		return d







		
		

		



    	tf.add(tf.gather_nd(self.h0, inputs))

class lossOptimize():
	pass