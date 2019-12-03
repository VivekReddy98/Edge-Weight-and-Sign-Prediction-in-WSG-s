import tensorflow as tf
import numpy as np
from itertools import cycle
from src.weightSGCN import weightSGCN
from src.initialization import *
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast

# Ref Code: https://github.com/tkipf/gcn/blob/master/gcn/models.py
# outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
# Ref Code: https://github.com/tkipf/gcn/blob/master/gcn/train.py


class dataGen:
	def __init__(self, batch_size, N):
		self.BS = batch_size
		self.N = N
		self.i=0

	def batchIterator(self):
		while (self.i+self.BS) < self.N:
			self.i = self.i+self.BS
			yield (self.i-self.BS, self.i)
		yield (self.i, self.N-1)

	def gen(self):
		a = self.batchIterator()
		while True:
			try:
				yield next(a)
			except:
				self.i=0
				a = self.batchIterator()

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
		self.loss = 0
		self.optimizer = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
		self.opt_op = self.optimizer.minimize(self.loss)
		self.build(kwargs['num_L'], kwargs['adj_pos'], kwargs['adj_neg'], kwargs['d_out'], kwargs['values'])

	def build(self, num_L, adj_pos, adj_neg, d_out, values):
		''' Values should be of type float32 ''' 
		init = weightSGCN(num_L, values.shape[0], values.shape[1], d_out)

		WU0 = init.weightsLayer1(name="Weights_firstLayer_unbalanced")
		WB0 = init.weightsLayer1(name="Weights_firstLayer_balanced")
		h0 = init.initialEmbeddings(name="Pre_Generated_Embeddings", values=values)
		WB = init.weightsLayer1N(name="Weights_Balanced")
		WU = init.weightsLayer1N(name='Weights_Unbalanced')
		zUB = init.Embeddings(name='Concat_Embeddings')
		MLG = init.weightsMLG(name='weights_for_Multinomial_Logistic_Regression')
		self.Layers.append(Layer0(WU0, WB0, h0))
		for i in range(1,num_L):
			self.Layers.append(LayerIntermediate(i, WB, WU, adj_pos, adj_neg))
		self.Layers.append(LayerLast(num_L-1))
		return None

	def predict(self, h, start, end):
		h = self.L0(h, start, end)
		h = self.L1(h, start, end)
		h = self.L2(h, start, end)
		h = self.L3(h, start, end)
		zUB = self.L4(h, start, end)
		return zUB

	




	





	def _loss(self):
		pass


if __name__ == "__main__":
	with tf.Session() as sess:
		adj_pos = np.random.randint(2, size=(5, 5)).astype('float32')
		adj_neg = np.random.randint(2, size=(5, 5)).astype('float32')
		values = np.ones((5, 5))
		values = values.astype('float32')
		sgcn = sgcn(4, adj_pos, adj_neg, 2, values)

