import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

## For a given Batch size and number of nodes, generate (start, end) indexes as a tuple forever.
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
		self.itr = self.gen()

	def genTriplets(neutral_sampling_rate=0.50):
		(start, end) = next(self.itr)
		df_M = pd.Dataframe()
		df_M_plus = pd.Dataframe()
		df_M_minus = pd.Dataframe()
		for i in range(start, end+1):
			neu_ary = adj_pos[i,:]+adj_neg[i,:]
			neu_ary[neu_ary != 0] = 0
			neu_ary[neu_ary == 0] = 1
			total_array = np.sum(neu_ary)
			neu_ary = neu_ary/np.sum(neu_ary)
			neu_ary = np.random.choice(self.N, int(total_array*neutral_sampling_rate), p=neu_ary)

			temp_ary_pos = np.where(adj_pos[i,:]!=0)[0]
			temp_ary_pos = np.append(np.append(np.full((temp_ary_pos.shape[0],) i), temp_ary_pos, axis=1), np.full((temp_ary_pos.shape[0],) 1),  axis=1)
			temp_ary_neg = np.where(adj_neg[i,:]!=0)[0]
			temp_ary_neg = np.append(np.append(np.full((temp_ary_pos.shape[0],) i), temp_ary_pos, axis=1), np.full((temp_ary_pos.shape[0],) -1),  axis=1)
			temp_ary_neu = np.append(np.append(np.full((neu_ary.shape[0],) i), neu_ary, axis=1), np.full((neu_ary.shape[0],) 1),  axis=1)

			df_pos = pd.Dataframe(temp_ary_pos, columns=['ui', 'uj', 's'])
			df_neg = pd.Dataframe(temp_ary_neg, columns=['ui', 'uj', 's'])
			df_neu = pd.Dataframe(temp_ary_neg, columns=['ui', 'uj', 's'])

			df_M = df_M.append(pd.concat[df_pos, df_neu, df_neg])

			df_pos.drop(['s'], axis=1)
			df_neg.drop(['s'], axis=1)
			df_neu.drop(['s'], axis=1)
			df_M_plus = df_M_plus.append(df_pos.join(df_neu, on='ui', how='left'))
			df_M_ = df_M_plus.append(df_pos.join(df_neu, on='ui', how='left').sample(frac=0.1))
			df_M_minus = df_M_minus.aaapend(df_neg.join(df_neu, on='ui', how='left').sample(frac=0.1))
			return None

















