from src.weightSGCN import weightSGCN
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen
from src.sgcn import sgcn
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

#https://github.com/tensorflow/tensorflow/issues/28287
global sess
global graph
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


# Pre-req's for the model
epochs = 20
G = parseInput(path="datasets/soc-sign-bitcoinalpha.csv", D_in=256)
itr = pairGenerator(batch_size=20).genPairs(G)

## Inital Embedding Computed using various methods
#values = np.random.rand(G.N, 256).astype(np.float32)

with graph.as_default():

	tf.compat.v1.keras.backend.set_session(sess)

	''' 
	1) Model Initialization 
	2) Model Build (Stacking Layers)
	3) Building the Computation graph and loss
	'''
	model = sgcn(lambdaa=0.5, learning_rate=0.005)
	model.build(4, G.adj_pos, G.adj_neg, 32, G.X.astype(np.float32))
	model.forwardPass()

	'''
	1) Running the model for number of epochs given
	2) For every Epoch, a feed dict with necessary values is created to be fed into the computation graph 
	'''

	# Initialize all the variables

	sess.run(tf.compat.v1.global_variables_initializer())

	list_trainable_var = sess.run(tf.compat.v1.trainable_variables())
	print(len(list_trainable_var))

	for i in range(0, epochs):

		print("................................................................................................................................")
		print("Epoch : {}".format(i))
		feed_dict = next(itr)
		
		print("Going into the Loss Function")

		#LOSS = tf.function(model.loss)
		print(feed_dict['range'])
		#print(sess.run(model.MLGloss(ses=sess)))
		#print(sess.run(model.BTlossPos()))
		#print(sess.run(model.BTlossNeg()))
		out_loss = sess.run([model.MLGloss(),model.BTlossPos(), model.BTlossNeg(), model.loss], feed_dict={model.twins: feed_dict['twins_X'],
													model.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
													model.pos_triplets: feed_dict['pos_triplets'],
													model.neg_triplets: feed_dict['neg_triplets'],
													model.start: np.array(feed_dict['range'][0]).astype(np.int32),
													model.end: np.array(feed_dict['range'][1]).astype(np.int32)
												  })
		print(out_loss)
		
		outs = sess.run(model.optimizer.minimize(model.loss_, tf.compat.v1.train.get_or_create_global_step(), model.var_list))

		print(outs)




