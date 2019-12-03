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
	def __init__(self, **kwargs):
		
		self.Layers = []
		
		''' Defined as Placeholders as the values in them might change depending on the batch'''
		self.twins = tf.placeholder(tf.float32)
		self.one_hot_encode = tf.placeholder(tf.float32)
		self.pos_triplets = tf.placeholder(tf.float32)
		self.neg_triplets = tf.placeholder(tf.float32)
		self.start = tf.placeholder(tf.float32)
		self.end = tf.placeholder(tf.float32)

		''' Optimizer and Loss Defined '''
		self.optimizer = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
		self.loss = 0
		self.opt_op = self.optimizer.minimize(self.loss)

		#self.build(kwargs['num_L'], kwargs['adj_pos'], kwargs['adj_neg'], kwargs['d_out'], kwargs['values'])

	def build(self, num_L, adj_pos, adj_neg, d_out, values):
		''' Values should be of type float32 ''' 
		init = weightSGCN(num_L, values.shape[0], values.shape[1], d_out)
		WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
		WB = init.weightsLayer1N(name="Weights_Balanced")
		WU = init.weightsLayer1N(name='Weights_Unbalanced')
		self.zUB = init.Embeddings(name='Concat_Embeddings')
		self.MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')
		self.Layers.append(Layer0(WU0, WB0, h0))
		for i in range(1,num_L):
			self.Layers.append(LayerIntermediate(i, WB, WU, adj_pos, adj_neg))
		self.Layers.append(LayerLast(num_L-1))
		return None

	def predict(self, h, start, end):
		for index in range(0,self.Layers-1):
			h = self.Layers[index].call(h, start, end)
		self.zUB = self.Layers[-1].call(h)
		return zUB

	def MLGloss(self):
		
		pass


if __name__ == "__main__":
	with tf.Session() as sess:
		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')
		values = np.ones((5, 5))
		values = values.astype('float32')
		sgcn = sgcn(4, adj_pos, adj_neg, 2, values)

