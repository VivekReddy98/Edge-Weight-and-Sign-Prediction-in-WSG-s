from src.weightSGCN import weightSGCN
from src.sgcnLayers import DS, Layer0, LayerIntermediate, LayerLast
from src.generatorUtils import parseInput, pairGenerator, dataGen
from src.sgcn import sgcn
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


# Pre-req's for the model
epochs = 20
G = parseInput(path="datasets/soc-sign-bitcoinalpha.csv", D_in=256)
G.generate()
itr = pairGenerator(batch_size=2).genPairs(G)

## Inital Embedding Computed using various methods
values = np.random.rand(G.N, 256).astype(np.float32)
''' 
1) Model Initialization 
2) Model Build (Stacking Layers)
3) Building the Computation graph and loss
'''
model = sgcn(lambdaa=0.5, learning_rate=0.005)
model.build(4, G.adj_pos, G.adj_neg, 32, values.astype(np.float32))
model.forwardPass()

'''
1) Running the model for number of epochs given
2) For every Epoch, a feed dict with necessary values is created to be fed into the computation graph 
'''
with tf.compat.v1.Session() as sess:

	# Initialize all the variables

	sess.run(tf.compat.v1.global_variables_initializer())

	for i in range(0, epochs):

		print("................................................................................................................................")
		print("Epoch : {}".format(i))
		feed_dict = next(itr)

		print("Going into the Loss Function")
		outs = sess.run([model.optimizer.minimize(model.loss), model.loss], feed_dict={model.twins: feed_dict['twins_X'],
																					   model.one_hot_encode: feed_dict['twins_Y'].astype(np.float32),
																					   model.pos_triplets: feed_dict['pos_triplets'],
																					   model.neg_triplets: feed_dict['neg_triplets'],
																					   model.start: np.array(feed_dict['range'][0]).astype(np.int32),
																					   model.end: np.array(feed_dict['range'][1]).astype(np.int32)
																						})

		print(outs)




