from src.weightSGCN import weightSGCN
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen, preprocessed_graph
from src.sgcn import sgcn, Trainable_Weights, BackProp
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.optimizers import *
tf.get_logger().setLevel('ERROR')
#tf.disable_eager_execution()
import csv, h5py

''' Define all the hyper parameters here'''
epochs = 3
batch_size = 1024
loss_epochs = 6
path = os.path.join("datasets", "original", "soc-sign-bitcoinalpha.csv")
D_in = 512 # Dimension generated by SSE
D_out = 128 # Embeddings will be 
l1 = 5
l2 = 0.5
learning_rate = 0.009
numLayers = 3
print(path.split(os.path.sep)[-1].split(".")[0])


#https://github.com/tensorflow/tensorflow/issues/28287
global sess
global graph

## Required When Multiple Precesses are running on GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf.keras.backend.get_session()
graph = tf.compat.v1.get_default_graph()

# Pre-req's for the model
prePro = preprocessed_graph(path=path)
prePro.get_new_df()
prePro.save_df("datasets\\mod")
G = parseInput(path=path, D_in=D_in, prePro=prePro)
itr = pairGenerator(batch_size=batch_size).genPairs(G)


with sess.as_default():

	#https://github.com/tensorflow/cleverhans/issues/1117
	#https://github.com/horovod/horovod/issues/511
	#https://github.com/keras-team/keras/issues/13550
	#https://github.com/tensorflow/tensorflow/issues/28287
	#https://github.com/tensorflow/tensorflow/issues/24371

	tf.keras.backend.set_session(sess)

	print("\n")
	print(''' Build the Computation Graphs for Forward Pass and BackPropagation (Forward Prop and Back Prop are Isolated, Back Prop Takes zUB as an input, although obviously they share Weights) ''')
	# Initialize all the variables
	Weights = Trainable_Weights(numLayers, G.adj_pos, G.adj_neg, D_out, G.X.astype(np.float32))
	
	# Model to compute Node Embeddings
	modelfwd = sgcn(Weights)
	modelfwd.build(numLayers) # Num layers
	zUB = modelfwd.forwardPass()

	# BackProp to Optimize Weights
	bckProp = BackProp(Weights, l1=l1, l2=l2, learning_rate=learning_rate)
	bckProp.loss = tf.add(tf.add(bckProp.MLGloss(), tf.add(bckProp.BTlossPos(), bckProp.BTlossNeg())), bckProp.RegLoss())
	''' Initialize all the Variables '''
	sess.run(tf.global_variables_initializer())

	#writer = tf.summary.FileWriter("datasets\\", graph=graph)

	print("\n")
	print(''' A Sanity Check to check all the Trainable Variables Defined ''')
	variables_names = [v.name for v in tf.trainable_variables()]
	values = sess.run(variables_names)
	for k, v in zip(variables_names, values):
		print("Variable: ", k)
		print("Shape: ", v.shape)
		#print(v)
	
	print("Singe Shot - Formward Pass .... ... .....")

	''' Only Start and End Indexes are Required to run get zUB'''
	Final_Layer_Embeddings = sess.run(zUB, feed_dict={modelfwd.start: np.array(0).astype(np.int32),
													modelfwd.end: np.array(G.N+1).astype(np.int32)})
	print(np.sum(Final_Layer_Embeddings), Final_Layer_Embeddings.shape)

	print("Completed the forward pass ")
	glob_loss_list = []
	loss_list = []

	for j in range(0, epochs):
	
		''' Miniminzing Loss  '''
		for i in range(0, loss_epochs):

			print("........................................................Epoch {}........................................................................".format(j))
			print("Loss Epoch : {}".format(i))

			feed_dict = next(itr)
			grads = tf.gradients(bckProp.loss, bckProp.var_list)
			
			''' Final Layer Embeddings Generated From the last layer is used to run the loss '''
			opt = sess.run([bckProp.loss, bckProp.MLGloss(), bckProp.RegLoss(), bckProp.BTlossPos(), bckProp.BTlossNeg(), bckProp.optimizer.apply_gradients(zip(grads, bckProp.var_list))], 
																											feed_dict={bckProp.twins: feed_dict['twins_X'],
																														bckProp.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
																														bckProp.pos_triplets: feed_dict['pos_triplets'],
																														bckProp.neg_triplets: feed_dict['neg_triplets'],
																														bckProp.start: np.array(feed_dict['range'][0]).astype(np.int32),
																														bckProp.end: np.array(feed_dict['range'][1]).astype(np.int32),
																														bckProp.zUB: Final_Layer_Embeddings.astype(np.float32)
																														})
			print(opt[0:4])
			loss_list.append(opt[0:5])

		glob_loss_list.append(loss_list)
		print("Singe Shot - Forward Pass epoch {}".format(j))

		''' Running zUB on updated weights '''
		Final_Layer_Embeddings = sess.run(zUB, feed_dict={modelfwd.start: np.array(0).astype(np.int32),
														modelfwd.end: np.array(G.N+1).astype(np.int32)})
		print(np.sum(Final_Layer_Embeddings), Final_Layer_Embeddings.shape)

		print("Completed the forward pass for epoch {}".format(j))

	with open("results\\out_{}_{}.csv".format(batch_size, numLayers, path.split(os.path.sep)[-1].split(".")[0]),"w") as f:
		wr = csv.writer(f)
		wr.writerows(loss_list)
		#writer = tf.summary.FileWriter("datasets\\", graph=graph)

	MLG_W = sess.run(Weights.MLG)

	with h5py.File('embeddings\\Embeddings_{}_{}.h5'.format(path.split(os.path.sep)[-1].split(".")[0], numLayers), 'w') as hf:
		hf.create_dataset("Embeddings",  data=Final_Layer_Embeddings)
		hf.create_dataset("MLGWeights",  data=MLG_W)

	

		