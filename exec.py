from src.weightSGCN import weightSGCN
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen
from src.sgcn import sgcn, Trainable_Weights, BackProp
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.optimizers import *
tf.get_logger().setLevel('ERROR')
#tf.disable_eager_execution()

#https://github.com/tensorflow/tensorflow/issues/28287
global sess
global graph

sess = tf.Session()
#sess = tf.keras.backend.get_session()
graph = tf.compat.v1.get_default_graph()

epochs = 20
# Pre-req's for the model
G = parseInput(path="datasets/soc-sign-bitcoinalpha.csv", D_in=256)
itr = pairGenerator(batch_size=256).genPairs(G)

## Inital Embedding Computed using various methods
#values = np.random.rand(G.N, 256).astype(np.float32)

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
	Weights = Trainable_Weights(4, G.adj_pos, G.adj_neg, 32, G.X.astype(np.float32))
	
	# Model to compute Node Embeddings
	modelfwd = sgcn(Weights)
	modelfwd.build(4) # Num layers
	zUB = modelfwd.forwardPass()

	# BackProp to Optimize Weights
	bckProp = BackProp(Weights, l1=5, l2=0.02, learning_rate=0.001)
	bckProp.loss = tf.add(tf.add(bckProp.MLGloss(), tf.add(bckProp.BTlossPos(), bckProp.BTlossNeg())), bckProp.RegLoss())
	''' Initialize all the Variables '''
	sess.run(tf.global_variables_initializer())

	#writer = tf.summary.FileWriter("datasets\\", graph=graph)

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


		print("\n")
		print(''' A Sanity Check to check all the Trainable Variables Defined ''')
		variables_names = [v.name for v in tf.trainable_variables()]
		values = sess.run(variables_names)
		for k, v in zip(variables_names, values):
			print("Variable: ", k)
			print("Shape: ", v.shape)
			#print(v)
		print("Going into the Loss Function")	

		# with tf.GradientTape(persistent=True) as tape:
		# 	tape.watch(bckProp.var_list)
		# 	grads = tape.gradient(bckProp.loss, bckProp.var_list)
		# 	print(grads)

		grads = tf.gradients(bckProp.loss, bckProp.var_list)
		# out_loss = sess.run(grads, feed_dict={bckProp.twins: feed_dict['twins_X'],
		# 										   bckProp.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
		# 										   bckProp.pos_triplets: feed_dict['pos_triplets'],
		# 										   bckProp.neg_triplets: feed_dict['neg_triplets'],
		# 										   bckProp.start: np.array(feed_dict['range'][0]).astype(np.int32),
		# 										   bckProp.end: np.array(feed_dict['range'][1]).astype(np.int32),
		# 										   bckProp.zUB: Final_Layer_Embeddings.astype(np.float32)
		# 										  })
		#print(out_loss)
		#print(sess.run(grads))
		''' Final Layer Embeddings Generated From Previous layer is used to run the loss '''
		opt = sess.run(bckProp.optimizer.apply_gradients(zip(grads, bckProp.var_list)), feed_dict={bckProp.twins: feed_dict['twins_X'],
												   bckProp.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
												   bckProp.pos_triplets: feed_dict['pos_triplets'],
												   bckProp.neg_triplets: feed_dict['neg_triplets'],
												   bckProp.start: np.array(feed_dict['range'][0]).astype(np.int32),
												   bckProp.end: np.array(feed_dict['range'][1]).astype(np.int32),
												   bckProp.zUB: Final_Layer_Embeddings.astype(np.float32)
												  })
		print(opt)
		#writer = tf.summary.FileWriter("datasets\\", graph=graph)
		