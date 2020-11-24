import os, torch
torch.manual_seed(124) 

import numpy as np
np.random.seed(124)

from torch import nn
from torch.nn import functional as F

class ConcordanceCorCoeff(nn.Module):

	def __init__(self):
		super(ConcordanceCorCoeff, self).__init__()
		
		self.mean = torch.mean
		self.std  = torch.std
		self.var  = torch.var

		self.sum  = torch.sum
		self.sqrt = torch.sqrt

	def forward(self, prediction, ground_truth):

		mean_gt   = self.mean(ground_truth, 0)
		mean_pred = self.mean(prediction, 0)

		var_gt   = self.var(ground_truth, 0)
		var_pred = self.var(prediction, 0)

		v_gt   = ground_truth - mean_gt
		v_pred = prediction - mean_pred

		sd_gt   = self.std(ground_truth)
		sd_pred = self.std(prediction)

		cor = self.sum (v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))

		numerator = 2*cor*sd_gt*sd_pred
		denominator = var_gt + var_pred + (mean_gt - mean_pred)**2

		ccc = numerator / denominator

		return 1-ccc

class ContrastiveLoss(nn.Module):
	"""
	Contrastive loss
	Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
	"""

	def __init__(self):
		super(ContrastiveLoss, self).__init__()
		self.margin = 30
		self.eps = 1e-9

	def forward(self, output1, output2, size_average=True):
		distances = (output2 - output1).pow(2).sum(1)  # squared distances
		losses = F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
		return losses.mean() if size_average else losses.sum()

class AdversarialLoss(nn.Module):

	"""
	Implementation of adversarial loss function described in:
	https://arxiv.org/pdf/1608.04644.pdf
	"""

	def __init__(self, num_classes=250, kappa=0, is_targeted=False, **kwargs):
		self.is_targeted = is_targeted
		self.kappa 		 = kappa
		self.num_classes = num_classes

	def forward(self, logits, target):


		target_one_hot = torch.eye(self.num_classes).type(logits.type())[target.long()]
		
		real = torch.sum(target_one_hot*logits.squeeze())

		# subtract large value from target class to find other max value
		# -> see https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
		other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
		kappa = torch.zeros_like(other).fill_(self.kappa)

		if self.is_targeted:
			return torch.sum(torch.max(other-real, kappa))

		return torch.sum(torch.max(real-other, kappa))

