import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticsPooling(nn.Module):

	def __init__(self, input_dim, output_dim=1, attentive=True):

		super(StatisticsPooling,self).__init__()
		self.attentive = attentive

		if attentive:
			self.layer = nn.Linear(input_dim,output_dim) # 4

	def forward(self, x):
		if self.attentive:
			return self.attentive_statistics_pooling_(x)
		else:
			return self.statistics_pooling_(x)

	def statistics_pooling_(self,x):

		# Mean
		mean = torch.mean(x,dim=1)

		# Standard deviation
		std = torch.std(x,dim=1)

		# Concatenate mean and second order statistics
		att_stat = torch.cat((mean,std), dim=1)

		return att_stat

	def attentive_statistics_pooling_(self, x):

		"""
		Attentive statistic pooling layer
		:param x:
		:return x_utt:
		"""

		# Compute attention coef
		att_coef = torch.tanh(self.layer(x))
		att_coef = F.softmax(att_coef,dim=1)

		# Compute attention mean
		multiplication = torch.mul(att_coef,x)
		mean = torch.sum(multiplication,dim=1)

		# Compute second order statistics
		mean2 = multiplication * x
		mean2 = torch.sum(mean2,dim=1)
		deviation = torch.sqrt(mean2.sub(mean * mean) + 1e-8)

		# Concatenate mean and second order statistics
		att_stat = torch.cat((mean,deviation),dim=1)

		return att_stat
