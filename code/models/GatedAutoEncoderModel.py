import torch
import torch.nn as nn

from modules.GatedConvolutionalLayer import GatedConvolution
from utils.mdct import MDCT as mdct

class GatedAutoEncoder(nn.Module):

	'''
		In : (N, sentence_len)
		Out: (N, sentence_len, embd_size)
	'''

	def __init__(self, parameters_encoder, parameters_decoder, parameters_conv, frequency_domain=False, parameters_freq={}, **kwargs):

		super(GatedAutoEncoder, self).__init__()

		self.parameters_encoder = parameters_encoder
		self.parameters_decoder = parameters_decoder
		self.parameters_conv = parameters_conv

		self.encoder = self._init_encoder_decoder_(parameters_encoder)
		self.decoder = self._init_encoder_decoder_(parameters_decoder)

		self.frequency_domain = frequency_domain

		if self.frequency_domain:
			self.parameters_freq = parameters_freq
			self.mdct        = mdct()

	def _init_encoder_decoder_(self, params):

		layers = []
		for i in range(0,params['n_layers']):
			layers.append(GatedConvolution(
					input_channels=params['input_channels'][i],
					output_channels=params['output_channels'][i],
					kernel_size=params['kernel_size'][i],
					stride=params['stride'][i],
					padding=params['padding'][i],
					bias=params['bias'],
					activation_fn=params['activation']))

			if params['batch_norm']:
				layers.append(nn.BatchNorm2d(num_features=params['output_channels'][i]))

			if params['dropout']:
				layers.append(nn.Dropout(params['dropout_value']))

		return nn.Sequential(*layers)

	def forward(self, input):
		if self.frequency_domain:
			input_MDCT = self.mdct.transform(input.squeeze(dim=2))
			input_MDCT_input = self.mdct.inverse(input_MDCT)

		input_MDCT_normalised = (input_MDCT-input_MDCT.mean())/input_MDCT.std()
		input_MDCT_normalised = input_MDCT_normalised.unsqueeze(dim=1)

		X = self.encoder(input_MDCT_normalised)
		X = torch.cat([input_MDCT_normalised, X], dim=1)
		X = self.decoder(X)
		X = X.squeeze(dim=1)

		X_deNormalised= X*input_MDCT.std()+input_MDCT.mean()
		if self.frequency_domain:
			Y = self.mdct.inverse(X_deNormalised)
			Y= input.abs().max()*Y/Y.abs().max()

		return Y.unsqueeze(dim=-1)


