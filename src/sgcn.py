import tensorflow as tf
import numpy as np
from itertools import cycle
from src.weightSGCN import weightSGCN
from src.initialization import *
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen

# Ref Code: https://github.com/tkipf/gcn/blob/master/gcn/models.py
# outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
# Ref Code: https://github.com/tkipf/gcn/blob/master/gcn/train.py

def pairGenerator(dataGen):
	def __init__(self, batch_size=256, epochs, adj_pos, adj_neg):
		self.BS = batch_size
		self.i = 0
		self.N = adj_neg.shape[0]
		self.adj_neg = adj_neg
		self.adj_pos = adj_pos

class sgcn():
	def __init__(self, lambdaa, **kwargs):
		
		self.Layers = []
		
		''' Defined as Placeholders as the values in them might change depending on the batch'''
		self.twins = tf.placeholder(tf.int32)
		self.one_hot_encode = tf.placeholder(tf.float32)
		self.pos_triplets = tf.placeholder(tf.int32)
		self.neg_triplets = tf.placeholder(tf.int32)
		self.start = tf.placeholder(tf.int32)
		self.end = tf.placeholder(tf.int32)
		self.lambdaa = lambdaa

		''' Optimizer and Loss Defined '''
		self.optimizer = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
		self.loss = tf.placeholder(tf.float32)
		

		#self.build(kwargs['num_L'], kwargs['adj_pos'], kwargs['adj_neg'], kwargs['d_out'], kwargs['values'])

	def build(self, num_L, adj_pos, adj_neg, d_out, values):
		''' Values should be of type float32 ''' 
		init = weightSGCN(num_L, values.shape[0], values.shape[1], d_out)
		self.WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		self.WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
		self.WB = init.weightsLayer1N(name="Weights_Balanced")
		self.WU = init.weightsLayer1N(name='Weights_Unbalanced')
		self.zUB = init.Embeddings(name='Concat_Embeddings')
		self.MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')
		self.Layers.append(Layer0(WU0, WB0, h0))
		for i in range(1,num_L):
			self.Layers.append(LayerIntermediate(i, self.WB, self.WU, adj_pos, adj_neg))
		self.Layers.append(LayerLast(num_L-1))
		return None

	def forwardPass(self, h, start, end):
		for index in range(0,self.Layers-1):
			h = self.Layers[index].call(h, start, end)
		self.zUB = self.Layers[-1].call(h)

		self.loss = tf.add(tf.add(tf.math.scalar_mul(-1, self.MLGloss), tf.add(self.BTlossPosl, self.BTlossNeg)))
		self.loss = tf.add(self.loss,self.RegLoss)
		self.opt_op = self.optimizer.minimize(self.loss)
		return None

	def MLGloss(self):
		zi = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,0], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,1], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zij = tf.concat([zi, zj], 1)
		eij = tf.math.exp(tf.matmul(zij, self.MLG, transpose_b=True)) #Batch_size, 3
		eij_mask = tf.math.multiply(eij, self.one_hot_encode)
		eij = tf.math.reciprocal(tf.math.reduce_sum(eij, axis=1))
		eij_mask = tf.math.reduce_sum(eij_mask, axis=1)
		return tf.math.divide(tf.reduce_sum(tf.math.log(tf.math.multiply_no_nan(eij, eij_mask)), axis=0), tf.shape(self.twins)[0])

	# Balance Theory Loss Specified in the Paper for Positive Triplets
	def BTlossPos(self):
		ui = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,0], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))
		uj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,1], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))
		uk = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,2], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))

		uij = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uj)), axis=1)
		uik = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uk)), axis=1)

		subijk = tf.math.subtract(uij,uik)
		bool_sub = tf.greater_equal(subijk, 0.)
		return tf.math.scalar_mul(self.l, tf.math.divide(tf.reduce_sum(tf.where_v2(bool_sub, subijk), axis=0), tf.shape(self.pos_triplets)[0]))

	def BTlossNeg(self):
		ui = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,0], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))
		uj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,1], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))
		uk = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,2], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))

		uij = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uj)), axis=1)
		uik = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uk)), axis=1)

		subijk = tf.math.subtract(uij,uik)
		bool_sub = tf.greater_equal(subijk, 0.)
		return tf.math.scalar_mul(self.l, tf.math.divide(tf.reduce_sum(tf.where_v2(bool_sub, subijk), axis=0), tf.shape(self.neg_triplets)[0]))

	def RegLoss(self):
		MLG_L = tf.norm(self.MLG, ord='fro')
		Z = tf.add(tf.norm(self.WU, ord='fro'), tf.norm(self.WB, ord='fro'))
		Z0 = tf.add(tf.norm(self.WU0, ord='fro'), tf.norm(self.WB0, ord='fro'))
		return tf.math.reduce_sum(tf.stack([MLG_L, Z, Z0]))

if __name__ == "__main__":
	with tf.Session() as sess:
		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')
		values = np.ones((5, 5))
		values = values.astype('float32')
		sgcn = sgcn(4, adj_pos, adj_neg, 2, values)

