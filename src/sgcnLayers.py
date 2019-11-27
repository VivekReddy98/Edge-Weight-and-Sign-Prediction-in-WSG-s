import tensorflow as tf

# class GodLayer():
# 	def __init__(self, WB, WU, hB, hU):
# 		''' Input Tensors '''
		
class Layer1():
	def __init__(self, WB, WU, hB, hU, num_outputs, trainable=True, name='Immediate_Neighbours', **kwargs):
		super(Layer1, self).__init__()
		self.num_outputs = num_outputs
		# Expected Matrices for Layer 1 
 		self.WB = WB   
		self.WU = WU
		self.hB = hB
		self.hU = hU

	def call(self, inputs, **kwargs):


class lossOptimize():
	pass



def validate_kwargs(kwargs, allowed_kwargs,
                    error_message='Keyword argument not understood:'):
  """Checks that all keyword arguments are in the set of allowed keys."""
  for kwarg in kwargs:
    if kwarg not in allowed_kwargs:
      raise TypeError(error_message, kwarg)