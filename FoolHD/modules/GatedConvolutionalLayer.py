import torch
import torch.nn as nn

class GatedConvolution(nn.Module):
	def __init__(self, input_channels, output_channels, 
					   kernel_size, stride=1, padding=0, 
					   dilation=1, groups=1, bias=False, activation_fn=nn.Sigmoid(), 
					   **kwargs):

		super(GatedConvolution, self).__init__()

		self.parameters = {
			'in_channels'  : input_channels,
			'out_channels' : output_channels,
			'kernel_size'  	  : kernel_size,
			'stride'	      : stride,
			'dilation'	      : dilation,
			'groups'	      : groups,
			'bias'		      : bias
		}

		self.pad = nn.ConstantPad2d(padding, value=0)
		self.conv = nn.Sequential(self.pad, nn.Conv2d(**self.parameters))
		self.activation = activation_fn

		self.gate = nn.Sequential(self.pad, nn.Conv2d(**self.parameters), self.activation)
	
	def forward(self, X):
		return self.conv(X) * self.gate(X)
