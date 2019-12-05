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


class sgcn():
	def __init__(self, **kwargs):
		
		self.Layers = []
		
		''' Defined as Placeholders as the values in them might change depending on the batch'''
		self.twins = tf.compat.v1.placeholder(tf.int32, name='Pairs_MLGLoss')
		self.one_hot_encode = tf.compat.v1.placeholder(tf.float32, name='Class_as_oneHot_encoded_vector')
		self.pos_triplets = tf.compat.v1.placeholder(tf.int32, name='Triplelets_Positive_nodes')
		self.neg_triplets = tf.compat.v1.placeholder(tf.int32, name='Triplelets_Negative_nodes')
		self.start = tf.compat.v1.placeholder(tf.int32, name='Start_index')
		self.end = tf.compat.v1.placeholder(tf.int32, name='End_Index')
		self.l = tf.constant(kwargs['lambdaa'], dtype=tf.float32, name="lambda")

		''' Optimizer and Loss Defined '''
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
		self.loss = tf.compat.v1.placeholder(tf.float32)
		

		#self.build(kwargs['num_L'], kwargs['adj_pos'], kwargs['adj_neg'], kwargs['d_out'], kwargs['values'])

	def build(self, num_L, adj_pos, adj_neg, d_out, values):
		''' Values should be of type float32 ''' 
		self.adj_pos = tf.constant(adj_pos.astype(np.float32), name='Adj_Pos_Static')
		self.adj_neg = tf.constant(adj_neg.astype(np.float32), name='Adj_Neg_Static')

		init = weightSGCN(num_L, values.shape[0], values.shape[1], d_out)
		self.WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		self.WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
		self.WB = init.weightsLayer1N(name="Weights_Balanced")
		self.WU = init.weightsLayer1N(name='Weights_Unbalanced')
		self.zUB = init.Embeddings(name='Concat_Embeddings')
		self.MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')
		self.h = init.interEmbeddings(name='Embeddings_B_UB')
		self.Layers.append(Layer0(self.WU0, self.WB0, h0, self.adj_pos, self.adj_neg))
		for i in range(1,num_L):
			self.Layers.append(LayerIntermediate(i, self.WB, self.WU, self.adj_pos, self.adj_neg))
		self.Layers.append(LayerLast(num_L-1))
		return None

	def forwardPass(self):
		for index in range(0,len(self.Layers)-1):
			self.h = self.Layers[index].call(self.h, self.start, self.end)
		self.zUB = self.Layers[-1].call(self.h)

		self.loss = tf.add(tf.add(self.MLGloss(), tf.add(self.BTlossPos(), self.BTlossNeg())), self.RegLoss())
		return None

	def MLGloss(self):
		zi = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,0], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,1], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zij = tf.concat([zi, zj], 1)
		eij = tf.math.exp(tf.matmul(zij, self.MLG, transpose_b=True)) #Batch_size, 3
		eij_mask = tf.math.multiply(eij, self.one_hot_encode)
		eij = tf.math.reciprocal(tf.math.reduce_sum(eij, axis=1))
		eij_mask = tf.math.reduce_sum(eij_mask, axis=1)
		mlg = tf.math.divide(tf.reduce_sum(tf.math.log(tf.math.multiply_no_nan(eij, eij_mask)), axis=0), tf.dtypes.cast(tf.shape(self.twins)[0], dtype=tf.float32))
		return tf.math.scalar_mul(tf.constant(-1, dtype=tf.float32), mlg)

	# Balance Theory Loss Specified in the Paper for Positive Triplets
	def BTlossPos(self):
		ui = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,0], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))
		uj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,1], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))
		uk = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.pos_triplets, [0,2], [-1,1]), [tf.shape(self.pos_triplets)[0], 1]))

		uij = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uj)), axis=1)
		uik = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uk)), axis=1)

		subijk = tf.math.subtract(uij,uik)
		bool_sub = tf.greater_equal(subijk, 0.)
		zero_tf = tf.zeros(shape=tf.shape(bool_sub), dtype=tf.float32)
		return tf.math.scalar_mul(self.l, tf.math.divide(tf.reduce_sum(tf.where(bool_sub, subijk, zero_tf), axis=0), tf.dtypes.cast(tf.shape(self.pos_triplets)[0], dtype=tf.float32)))

	# Balance Theory Loss Specified in the Paper for Negative Triplets
	def BTlossNeg(self):
		ui = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,0], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))
		uj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,1], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))
		uk = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.neg_triplets, [0,2], [-1,1]), [tf.shape(self.neg_triplets)[0], 1]))

		uij = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uj)), axis=1)
		uik = tf.math.reduce_sum(tf.math.square(tf.math.subtract(ui, uk)), axis=1)

		subijk = tf.math.subtract(uij,uik)
		bool_sub = tf.greater_equal(subijk, 0.)
		zero_tf = tf.zeros(shape=tf.shape(bool_sub), dtype=tf.float32)
		return tf.math.scalar_mul(self.l, tf.math.divide(tf.reduce_sum(tf.where(bool_sub, subijk, zero_tf), axis=0), tf.dtypes.cast(tf.shape(self.neg_triplets)[0], dtype=tf.float32)))

	def RegLoss(self):
		MLG_L = tf.norm(self.MLG, ord='euclidean')
		Z = tf.add(tf.norm(self.WU, ord='euclidean'), tf.norm(self.WB, ord='euclidean'))
		Z0 = tf.add(tf.norm(self.WU0, ord='euclidean'), tf.norm(self.WB0, ord='euclidean'))
		return tf.math.reduce_sum(tf.stack([MLG_L, Z, Z0]))

if __name__ == "__main__":
	with tf.Session() as sess:
		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')
		values = np.ones((5, 5))
		values = values.astype('float32')
		sgcn = sgcn(4, adj_pos, adj_neg, 2, values)

