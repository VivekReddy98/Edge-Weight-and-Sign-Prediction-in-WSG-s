import numpy as np
import pandas as pd
import networkx as nx
import random
from sklearn.preprocessing import OneHotEncoder

## For a given Batch size and number of nodes, generate (start, end) indexes as a tuple forever.
class dataGen():
	def __init__(self, batch_size):
		self.BS = batch_size
		self.i=0

	def batchIterator(self):
		while (self.i+self.BS) < self.N:
			self.i = self.i+self.BS
			yield (self.i-self.BS, self.i)
		yield (self.i, self.N-1)

	def gen(self, N):
		self.N = N
		a = self.batchIterator()
		while True:
			try:
				yield next(a)
			except:
				self.i=0
				a = self.batchIterator()


class parseInput():
	def __init__(self, path, column_names=['src', 'dst', 'rating', 'time']):
		self.df = pd.read_csv(path, names=column_names).drop(['time'], axis=1)
		self.N = max(self.df['src'].max(axis=0), self.df['dst'].max(axis=0))
		self.posG = nx.DiGraph()
		self.negG = nx.DiGraph()

	def __generator_function(self, limit):
		for i in range(0,limit):
			yield i

	def generate(self):
		itr = self.__generator_function(self.N)
		self.posG.add_nodes_from(itr)
		self.posG.add_weighted_edges_from(self.df[self.df['rating']>0].values.tolist())
		itr = self.__generator_function(self.N)
		self.negG.add_nodes_from(itr)
		self.posG.add_weighted_edges_from(self.df[self.df['rating']<0].values.tolist())
		self.adj_pos = nx.to_numpy_matrix(self.posG)
		self.adj_neg = nx.to_numpy_matrix(self.negG)
		print(self.N, self.adj_neg.shape[0])
		print("All Necessary Matrices have been computed")
		return None


class pairGenerator():
	def __init__(self, batch_size=256):
		self.BS = batch_size
		self.i = 0

	def genPairs(self, G, neutral_sampling_rate=0.50):
		D = dataGen(self.BS)
		self.itr = D.gen(G.N)
		
		while True:	
			(start, end) = next(self.itr)

			df = G.df[G.df['src'].isin([i for i in range(start, end+1)])]
			df_pos = df[df['rating']>0]
			df_neg = df[df['rating']<0]
			df_neu = pd.DataFrame()
			df_twins = pd.DataFrame()
			set_nodes = set(list([i for i in range(0, G.N)]))

			for i in range(start, end+1):
				pos_neigh = set(list(G.posG.neighbors(i)))
				neg_neigh = set(list(G.negG.neighbors(i)))
				neutral_nodes = set_nodes.difference(pos_neigh).difference(neg_neigh)
				neutral_neigh = list(random.sample(neutral_nodes, int(len(neutral_nodes)*neutral_sampling_rate)))
				df_neu_temp = pd.DataFrame(neutral_neigh, columns=['dst'])
				df_neu_temp['src'] = i
				df_neu_temp['rating'] = 0
				df_neu = df_neu.append(df_neu_temp)

			df_twins = df_twins.append(pd.concat([df_pos, df_neg, df_neu]))
			print(df_neu.head())
			df_pos = df_pos.drop(['rating'], axis=1)
			df_neg = df_neg.drop(['rating'], axis=1)
			df_neu = df_neu.drop(['rating'], axis=1)

			df_neu.rename(columns={"dst": "uk"}, inplace=True) 

			df_M_plus = df_pos.sample(frac=0.1).merge(df_neu, on='src', how='left').sample(frac=neutral_sampling_rate/3)
			df_M_minus = df_neg.sample(frac=0.1).merge(df_neu, on='src', how='left').sample(frac=neutral_sampling_rate/3)

			feed_dict = {"twins":df_twins, "pos_triplets": df_M_plus, "neg_triplets": df_M_minus}
			yield feed_dict
			break



if __name__ == "__main__":
	G = parseInput(path="datasets/soc-sign-bitcoinalpha.csv")
	G.generate()
	itr = pairGenerator().genPairs(G)
	feed_dict = next(itr)
	print(feed_dict)












