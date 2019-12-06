from src.weightSGCN import weightSGCN
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen
from src.sgcn import sgcn, Trainable_Weights, BackProp
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import *
tf.get_logger().setLevel('ERROR')
#tf.disable_eager_execution()
import csv, h5py

''' Define all the hyper parameters here'''
batch_size = 128
epochs = 40
path = "datasets/soc-sign-bitcoinotc.csv"
D_in = 512 # Dimension generated by SSE
D_out = 32 # Embeddings will be 
l1 = 30
l2 = 0.1
learning_rate = 0.009
numLayers = 8


#https://github.com/tensorflow/tensorflow/issues/28287
global sess
global graph

loss_list = []

sess = tf.Session()
#sess = tf.keras.backend.get_session()
graph = tf.compat.v1.get_default_graph()

# Pre-req's for the model
G = parseInput(path=path, D_in=D_in)
itr = pairGenerator(batch_size=batch_size).genPairs(G)


with sess.as_default():

	#https://github.com/tensorflow/cleverhans/issues/1117
	#https://github.com/horovod/horovod/issues/511
	#https://github.com/keras-team/keras/issues/13550
	#https://github.com/tensorflow/tensorflow/issues/28287
	#https://github.com/tensorflow/tensorflow/issues/24371

	tf.keras.backend.set_session(sess)

	print("\n")
	print(''' Build the Computation Graphs (Both are Isolated, Back Prop Takes zUB as an input, although obviously they share Weights) for Forward Pass and BackPropagation''')
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
	print("Going into the Loss Function")	

	''' Start The Execution '''
	for i in range(0, epochs):

		print("................................................................................................................................")
		print("Epoch : {}".format(i))
		feed_dict = next(itr)

		''' Only Start and End Indexes are Required to run get zUB'''
		Final_Layer_Embeddings = sess.run(zUB, feed_dict={modelfwd.start: np.array(feed_dict['range'][0]).astype(np.int32),
								   					  modelfwd.end: np.array(feed_dict['range'][1]).astype(np.int32)})


		#Final_Layer_Embeddings = np.ones((3783, 64))

		print(np.sum(Final_Layer_Embeddings), Final_Layer_Embeddings.shape)

		grads = tf.gradients(bckProp.loss, bckProp.var_list)
		
		''' Final Layer Embeddings Generated From Previous layer is used to run the loss '''
		opt = sess.run([bckProp.loss, bckProp.MLGloss(), bckProp.RegLoss(), bckProp.BTlossPos(), bckProp.BTlossNeg(),bckProp.optimizer.apply_gradients(zip(grads, bckProp.var_list))], feed_dict={bckProp.twins: feed_dict['twins_X'],
																													bckProp.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
																													bckProp.pos_triplets: feed_dict['pos_triplets'],
																													bckProp.neg_triplets: feed_dict['neg_triplets'],
																													bckProp.start: np.array(feed_dict['range'][0]).astype(np.int32),
																													bckProp.end: np.array(feed_dict['range'][1]).astype(np.int32),
																													bckProp.zUB: Final_Layer_Embeddings.astype(np.float32)
																													})
		print(opt[0:4])
		loss_list.append(opt[0:4])

	with open("results\\out_{}.csv".format(batch_size),"w") as f:
		wr = csv.writer(f)
		wr.writerows(loss_list)
		#writer = tf.summary.FileWriter("datasets\\", graph=graph)

	Final_Layer_Embeddings = sess.run(zUB, feed_dict={modelfwd.start: np.array(0).astype(np.int32),
								   					  modelfwd.end: np.array(G.N).astype(np.int32)})

	MLG_W = sess.run(Weights.MLG)

	with h5py.File('embeddings\\Embeddings_{}_{}_{}.h5'.format(batch_size, path.split("/")[1].split(".")[0], numLayers), 'w') as hf:
		hf.create_dataset("Embeddings",  data=Final_Layer_Embeddings)
		hf.create_dataset("MLGWeights",  data=MLG_W)
	

		