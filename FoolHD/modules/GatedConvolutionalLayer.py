import torch
import torch.nn as nn

class GatedConvolution(nn.Module):
	def __init__(self, input_channels, output_channels, 
					   kernel_size, stride=1, padding=0, 
					   padding_mode='zeros', dilation=1, 
					   groups=1, bias=False, activation_fn=nn.Sigmoid(), 
					   **kwargs):

		super(GatedConvolution, self).__init__()

		self.parameters = {
			'in_channels'  : input_channels,
			'out_channels' : output_channels,
			'kernel_size'  	  : kernel_size,
			'stride'	      : stride,
			'padding'	      : padding,
			'padding_mode'    : padding_mode,
			'dilation'	      : dilation,
			'groups'	      : groups,
			'bias'		      : bias
		}

		self.conv = nn.Conv2d(**self.parameters)
		self.activation = activation_fn

		self.gate = nn.Sequential(nn.Conv2d(**self.parameters), self.activation)
	
	def forward(self, X):
		return self.conv(X) * self.gate(X)
