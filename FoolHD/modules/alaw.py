import torch
import torch.nn as nn
import torch.nn.functional as F

class ALawQuantization(nn.Module):

	def __init__(self, A=87.6, **kwargs):
		super(ALawQuantization, self).__init__()
		self.A = A
	
	def forward(self, x):
		# Step 1 - Normalize input between [-1,1] if not in this range
		if x.max() > 1 or x.min() < -1:
			x /= x.abs().max()

		# Avoid repeating abs operation
		x_abs = x.abs()

		# Step 2 - Split equation in two terms -> smaller than (1/A) and larger than (1/A)
		lt = x_abs <  (1/self.A)
		gt = x_abs >= (1/self.A)

		return torch.sign(x) * (lt*(self.A * x_abs)  + gt*(1 + torch.log(self.A * x_abs))) / (1 + torch.log(self.A))

	def inverse(self, y):
		# Step 1 - Normalize input between [-1,1] if not in this range
		if y.max() > 1 or y.min() < -1:
			y /= y.abs().max()

		# Avoid repeating abs operation
		y_abs = y.abs()

		# Step 2 - Split equation in two terms -> smaller than (1/A) and larger than (1/A)
		lt = y_abs <  (1/(1+torch.log(self.A)))
		gt = y_abs >= (1/(1+torch.log(self.A)))

		return torch.sign(y) * (lt*(y_abs * (1 + torch.log(self.A))) + gt*(torch.exp(y_abs * (1 + torch.log(self.A)) - 1))) / self.A

class MuLawQuantization(nn.Module):

	def __init__(self, nb=8, **kwargs):
		super(MuLawQuantization, self).__init__()
		self.u = (2**nb) - 1

	def forward(self, x):

		# Step 1 - Normalize input between [-1,1] if not in this range
		if x.max() > 1 or x.min() < -1:
			x /= x.abs().max()

		# Step 2 - Apply companding transformation and return
		return torch.sign(x) * torch.log(1 + self.u * x.abs()) / torch.log(1 + self.u)

	def inverse(self, y):

		# Step 1 - Normalize input between [-1,1] if not in this range
		if y.max() > 1 or y.min() < -1:
			y /= y.abs().max()

		# Step 2 - Apply inverse transformation and return
		return torch.sign(y) * (1/self.u) * (torch.pow(1+self.u, y.abs()) - 1)
