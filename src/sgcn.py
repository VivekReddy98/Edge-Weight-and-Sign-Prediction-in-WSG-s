import tensorflow as tf
import numpy as np
from itertools import cycle
from src.weightSGCN import weightSGCN
from src.initialization import *
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen

class Embeddings:
	def __init__(self):
		self.U = []
		self.B = []

class Trainable_Weights:
	def __init__(self, num_L, adj_pos, adj_neg, d_out, values):
		''' Values should be of type float32 ''' 
		''' Initializing Constants '''
		self.adj_pos = tf.constant(adj_pos.astype(np.float32), name='Adj_Pos_Static')
		self.adj_neg = tf.constant(adj_neg.astype(np.float32), name='Adj_Neg_Static')

		''' Initializing Variables  '''
		self.init = weightSGCN(num_L, values.shape[0], values.shape[1], d_out)
		self.WU0 = self.init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		self.WB0 = self.init.weightsLayer1(name="Weights_firstLayer_balanced")
		self.h0 = self.init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
		self.WB = self.init.weightsLayer1N(name="Weights_Balanced")
		self.WU = self.init.weightsLayer1N(name='Weights_Unbalanced')
		self.MLG = self.init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')

class BackProp:
	def __init__(self, Weights, **kwargs):
		
		''' Defined as Placeholders as the values in them might change depending on the batch'''
		self.twins = tf.compat.v1.placeholder(tf.int32, name='Pairs_MLGLoss')
		self.one_hot_encode = tf.compat.v1.placeholder(tf.float32, name='Class_as_oneHot_encoded_vector')
		self.pos_triplets = tf.compat.v1.placeholder(tf.int32, name='Triplelets_Positive_nodes')
		self.neg_triplets = tf.compat.v1.placeholder(tf.int32, name='Triplelets_Negative_nodes')
		self.start = tf.compat.v1.placeholder(tf.int32, name='Start_index')
		self.end = tf.compat.v1.placeholder(tf.int32, name='End_Index')
		self.zUB = tf.compat.v1.placeholder(tf.float32, name='Final_layer_weights')

		''' Other Parameters for this model'''
		self.l1 = tf.constant(kwargs['l1'], dtype=tf.float32, name="lambda_Balance_Theory")
		self.l2 = tf.constant(kwargs['l2'], dtype=tf.float32, name="lambda_Regularization")
		self.W = Weights

		''' Inputs are final Layer embeddings generated from previous graph'''
		self.var_list = [self.W.WU0, self.W.WB0, self.W.WB, self.W.WU, self.W.MLG]

		''' Optimizer '''
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs['learning_rate'])

	#Loss Defined 
	def loss_(self):
		return tf.add(tf.add(self.MLGloss(), tf.add(self.BTlossPos(), self.BTlossNeg())), self.RegLoss())

	# MLG Loss defnied
	def MLGloss(self):
		zi = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,0], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zj = tf.gather_nd(self.zUB, tf.reshape(tf.slice(self.twins, [0,1], [-1,1]), [tf.shape(self.twins)[0], 1]))
		zij = tf.concat([zi, zj], 1)
		eij = tf.math.exp(tf.matmul(zij, self.W.MLG, transpose_b=True)) #Batch_size, 3
		eij_mask = tf.math.multiply(eij, self.one_hot_encode)
		eij = tf.math.reciprocal_no_nan(tf.math.reduce_sum(eij, axis=1))
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
		return tf.math.scalar_mul(self.l1, tf.math.divide_no_nan(tf.reduce_sum(tf.where(bool_sub, subijk, zero_tf), axis=0), tf.dtypes.cast(tf.shape(self.pos_triplets)[0], dtype=tf.float32)))

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
		return tf.math.scalar_mul(self.l1, tf.math.divide_no_nan(tf.reduce_sum(tf.where(bool_sub, subijk, zero_tf), axis=0), tf.dtypes.cast(tf.shape(self.neg_triplets)[0], dtype=tf.float32)))

	# Regularization Loss Defined in the Paper (Used euclidean norm)
	def RegLoss(self):
		MLG_L = tf.norm(self.W.MLG, ord='euclidean')
		Z = tf.add(tf.norm(self.W.WU, ord='euclidean'), tf.norm(self.W.WB, ord='euclidean'))
		Z0 = tf.add(tf.norm(self.W.WU0, ord='euclidean'), tf.norm(self.W.WB0, ord='euclidean'))
		RegL = tf.add(tf.add(MLG_L, Z), Z0)
		RegL = tf.math.scalar_mul(self.l2, RegL)
		RegL = tf.math.divide_no_nan(RegL, tf.dtypes.cast(tf.shape(self.neg_triplets)[0], dtype=tf.float32))
		return RegL

class sgcn:
	def __init__(self, W, **kwargs):
		
		self.Layers = []
		self.H = Embeddings()
		self.W = W
		''' Defined as Placeholders as the values in them might change depending on the batch'''
		self.start = tf.compat.v1.placeholder(tf.int32, name='Start_index')
		self.end = tf.compat.v1.placeholder(tf.int32, name='End_Index')
		
	def build(self, num_L):
		''' Initializing Layers and Layer Specific Weights'''
		self.Layers.append(Layer0(self.W.h0, self.W.WU0, self.W.WB0, self.W.adj_pos, self.W.adj_neg))
		for i in range(1,num_L):
			self.H.U.append(self.W.init.interEmbeddings(name='Embeddings_UnBalanced_'+str(i), trainable=False))
			self.H.B.append(self.W.init.interEmbeddings(name='Embeddings_Balanced_'+str(i), trainable=False))
			self.Layers.append(LayerIntermediate(i, self.W.WB, self.W.WU, self.W.adj_pos, self.W.adj_neg))
		self.H.U.append(self.W.init.interEmbeddings(name='Embeddings_UnBalanced_'+str(i+1),trainable=False))
		self.H.B.append(self.W.init.interEmbeddings(name='Embeddings_Balanced_'+str(i+1), trainable=False))
		self.Layers.append(LayerLast(num_L-1))
		return None

	def forwardPass(self):
		for index in range(0,len(self.Layers)-1):
			self.H = self.Layers[index].call(self.H, self.start, self.end)
		zUB = self.Layers[-1].call(self.H)
		return zUB


