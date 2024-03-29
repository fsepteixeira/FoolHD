import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math

torch.manual_seed(124) 

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
	def __init__(self, context, input_dim, output_dim, full_context=True):
		"""
		Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
		By default, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
		"""
		super(TDNN,self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim

		self.check_valid_context(context)
		self.kernel_width, context = self.get_kernel_width(context,full_context)
		self.register_buffer('context',torch.LongTensor(context))
		self.full_context = full_context

		stdv = 1./math.sqrt(input_dim)
		self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
		self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))

	def forward(self,x):
		"""
		x is one batch of data
		x.size(): [batch_size, sequence_length, input_dim]
		sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features

		output size: [batch_size, output_dim, len(valid_steps)]
		"""

		return self.__temporal_convolution_(x)

	def __temporal_convolution_(self, x):

		"""
		This function performs weighted multiplications given an arbitrary context. 
		Cannot directly use convolution because in case of only particular frames of context, one needs to select 
		only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
		"""

		input_shape = x.shape
		assert len(input_shape) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

		kernel = self.weight.float().cuda() # Is this necessary?
		bias = self.bias.float().cuda()

		batch_size, input_sequence_length, input_dim = input_shape
		x = x.transpose(1,2).contiguous()

		# Allocate memory for output
		valid_steps = self.get_valid_steps(self.context, input_sequence_length)
		xs = Variable(bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))

		# Perform the convolution with relevant input frames
		for c, i in enumerate(valid_steps):
			features = torch.index_select(x, 2, self.context+i)
			xs[:,:,c] = F.conv1d(features, kernel, bias=bias)[:,:,0]

		return xs

	@staticmethod
	def check_valid_context(context):
		# here context is still a list
		assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

	@staticmethod
	def get_kernel_width(context, full_context):
		if full_context:
			context = range(context[0],context[-1]+1)
		return len(context), context

	@staticmethod
	def get_valid_steps(context, input_sequence_length):
		start = 0 if context[0] >= 0 else -1*context[0]
		end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
		return range(start, end)
