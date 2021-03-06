import numpy as np
import pandas as pd
import networkx as nx
import random, math
import os, os.path
from numpy import linalg as LA
import networkx as nx
from networkx.linalg.laplacianmatrix import directed_laplacian_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


## For a given Batch size and number of nodes, generate (start, end) indexes as a tuple, forever.
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

class preprocessed_graph:
	def __init__(self, path, colnames=['source','target','weight']):
		self.path = path
		old_dataframe = pd.read_csv(path, names=colnames)
		old_nodes = self.get_old_node_ids(old_dataframe)
		self.nodes = self.get_new_node_ids(old_nodes)
		old_new_node_map = dict(zip(old_nodes,self.nodes))
		self.sources = self.get_new_source_ids(old_dataframe, old_new_node_map)
		self.targets = self.get_new_target_ids(old_dataframe, old_new_node_map)
		self.weights = old_dataframe.weight.tolist()
		self.edges = tuple(zip(self.sources, self.targets, map(lambda x: {'weight':x}, self.weights)))
		self.d = {colnames[0]:self.sources, colnames[1]:self.targets, colnames[2]:self.weights}

	def remap(self,old_list, old_new_map):
		num_nodes_in_list = len(old_list)
		new_list = old_list
		for i in range(0,num_nodes_in_list):
			new_list[i] = old_new_map[old_list[i]]
		return new_list
	
	def get_new_source_ids(self,old_dataframe,old_new_node_map):
		old_sources = old_dataframe.source.tolist()
		new_sources = self.remap(old_sources,old_new_node_map)
		return new_sources
		
	def get_new_target_ids(self,old_dataframe,old_new_node_map):
		old_targets = old_dataframe.target.tolist()
		new_targets = self.remap(old_targets,old_new_node_map)
		return new_targets
		
	def get_new_node_ids(self, old_node_ids):
		num_nodes = len(old_node_ids)
		new_node_ids = range(0,num_nodes)
		return new_node_ids

	def get_old_node_ids(self,dataframe):
		old_source_ids = dataframe.source.tolist()
		old_target_ids = dataframe.target.tolist()
		old_node_ids = list(set(old_source_ids).union(set(old_target_ids)))
		return old_node_ids

	def get_new_df(self):
		self.graph_dataframe = pd.DataFrame(data = self.d)
		self.graph_dataframe['BinaryWeight'] =  np.where(self.graph_dataframe['weight']>0, 1, -1)
		self.df_train, self.df_test,  scrap1, scrap2 = train_test_split(self.graph_dataframe[['source','target','weight']], self.graph_dataframe[['BinaryWeight']], test_size=0.15, stratify=self.graph_dataframe[['BinaryWeight']])
		self.df_train.reset_index(drop=True, inplace=True)
		self.df_test.reset_index(drop=True, inplace=True)
		return None
	
	def save_df(self, dir):
		self.graph_dataframe.to_csv(os.path.join(dir, "mod_{}".format(os.path.split(self.path)[-1])))
		self.df_test.to_csv(os.path.join(dir, "test_{}".format(os.path.split(self.path)[-1])))
		self.df_train.to_csv(os.path.join(dir, "train_{}".format(os.path.split(self.path)[-1])))
		return None

	def __generator_function(self, limit):
		for i in range(0,limit):
			yield i

	def dataframe(self,):
		return self.df_train

	def graph(self):
		N = max(max(self.sources), max(self.targets))
		itr = self.__generator_function(N)
		graph = nx.Graph()
		graph.add_nodes_from(itr)
		graph.add_weighted_edges_from(self.df_train.values.tolist())
		return graph

	def digraph(self):
		N = max(max(self.sources), max(self.targets))
		itr = self.__generator_function(N)
		digraph = nx.DiGraph()
		digraph.add_nodes_from(itr)
		digraph.add_weighted_edges_from(self.df_train.values.tolist())
		return digraph

class parseInput():
	def __init__(self, path, D_in, prePro, column_names=['source', 'target', 'weight']):
		self.prepro = prePro 
		self.df = self.prepro.dataframe()
		#self.df.head()
		#self.df.drop('time', axis=1, inplace=True)
		self.N = max(self.df['source'].max(axis=0), self.df['target'].max(axis=0))
		self.DiG = self.prepro.digraph()
		self.G = self.prepro.graph()
		adj = nx.to_numpy_matrix(self.G)
		self.adj_pos = np.where(adj > 0 , adj, 0)
		self.adj_neg = np.where(adj < 0 , -1*adj, 0)
		self.path = path
		self.X = self.create_X(D_in)
			
	def create_X(self, D_in):
		A = np.squeeze(np.asarray(nx.adjacency_matrix(self.G).todense()))
		degrees = abs(A).sum(axis=1)
		D = np.diag(degrees)
		L = D - A
		w,v = LA.eig(L)
		w,v = np.real(w),np.real(v) #removing imaginary parts caused due to roundoff
		w_argsort = w.argsort()
		v_sorted_T = []
		for i in range(0,len(w)):
		    v_sorted_T.append(v[:,w_argsort[i]])
		v_sorted = np.array(v_sorted_T).T
		X = v_sorted[:,4:68]
		return X

class pairGenerator():
	def __init__(self, batch_size=256, ):
		self.BS = batch_size
		self.i = 0

	def genPairs(self, GraphParser, neutral_sampling_rate=0.50):
		D = dataGen(self.BS)
		self.itr = D.gen(GraphParser.N)
		
		while True:	
			(start, end) = next(self.itr)

			df = GraphParser.df[GraphParser.df['source'].isin([i for i in range(start, end+1)])]
			df_pos = df[df['weight']>0]
			df_neg = df[df['weight']<0]
			df_neu = pd.DataFrame()
			df_twins = pd.DataFrame()
			set_nodes = set(list([i for i in range(0, GraphParser.N)]))

			for i in range(start, end+1):
				#neigh = set(list(G.DiG.neighbors(i)))
				neigh = set([GraphParser.DiG.neighbors(i)])
				neutral_nodes = set_nodes.difference(neigh)
				neutral_neigh = list(random.sample(neutral_nodes, int(len(neutral_nodes)*neutral_sampling_rate)))
				df_neu_temp = pd.DataFrame(neutral_neigh, columns=['target'])
				df_neu_temp['source'] = i
				df_neu_temp['weight'] = 0
				df_neu = df_neu.append(df_neu_temp.sample(frac=0.2))


			df_twins = df_twins.append(pd.concat([df_pos, df_neg, df_neu], sort=True))

			df_twins.dropna(inplace=True)
			df_twins.reset_index(drop=True)
			df_twins['weight'] = df_twins['weight'].apply(lambda x: 1 if x > 0 else (2 if x < 0 else 0))
			df_twins = df_twins[['source', 'target', 'weight']]
			#print(df_twins.sample(5))

			df_twins = df_twins.values


			df_twins_x = df_twins[:,0:2]
			y = df_twins[:,2]

			df_twins_y = np.zeros((y.size, 3))
			df_twins_y[np.arange(y.size),y] = 1
			
			#print(df_twins_y.shape, df_twins.shape)

			df_pos = df_pos.drop(['weight'], axis=1)
			df_neg = df_neg.drop(['weight'], axis=1)
			df_neu = df_neu.drop(['weight'], axis=1)

			df_neu.rename(columns={"target": "uk"}, inplace=True) 
			
			df_M_plus = df_pos.sample(frac=math.sqrt(self.BS)/self.BS).merge(df_neu, on='source', how='left').sample(frac=neutral_sampling_rate/4)
			df_M_minus = df_neg.sample(frac=math.sqrt(self.BS)/self.BS).merge(df_neu, on='source', how='left').sample(frac=neutral_sampling_rate/4)

			df_M_plus.dropna(inplace=True)

			df_M_minus.dropna(inplace=True)
			print(df_twins_x.shape, df_twins_y.shape, df_M_plus.shape, df_M_minus.shape)
			
			feed_dict = {"twins_X":df_twins_x, "twins_Y":df_twins_y, "pos_triplets": df_M_plus.values, "neg_triplets": df_M_minus.values, "range": (start, end)}
			yield feed_dict










