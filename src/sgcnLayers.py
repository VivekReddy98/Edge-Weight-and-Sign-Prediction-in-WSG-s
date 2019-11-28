import tensorflow as tf

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
	def __init__(self, WU0, WB0, h0, batch_size, num_outputs, trainable=True, name='Immediate_Neighbours', **kwargs):
		super(Layer1, self).__init__()
		#self.num_outputs = num_outputs
		# Required Matrices for Layer 1 
 		self.WU0 = WU0 
		self.WB0 = WB0
		self.h0 = h0

	def call(self, inputs, ds, adj_pos, adj_neg, **kwargs):
		''' Inputs are assumed to be Node Id's given as a 1-d list of form [[]]''' 
		''' kwargs should have two adj_lists, with in and out nodes information '''
		''' An Adj_List should be of format map of lists '''
		''' ds object having all the required matrices '''
    	tf.add(tf.gather_nd(self.h0, inputs))

class lossOptimize():
	pass