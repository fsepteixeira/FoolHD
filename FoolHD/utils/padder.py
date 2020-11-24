import torch, time

class PadBatch(object):
	'''
	Class to pad batches of sequences - Also sorts by sequence length from shortest to largest
	: X - List of uneven sequence matrices
	: y - Corresponding labels to sort
	: (Optional) lengths - Numpy array containing the length of each sequence in X
	Returns:
	: X - Padded with zeros, sorted by sequence length
	: y - Sorted in the same way as X
	'''
	
	def __call__(self, data):

		# Get data arrays
		(X, y, f) = zip(*data)

		if len(X) < 1:
			return torch.Tensor([]), None, None
		
		if type(y[0]) != str:
			y = torch.LongTensor(y)

		if len(X) == 1:
			return X[0].unsqueeze(dim=0).transpose(2,1), y, f

		# Pad sequences
		X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
		return X.transpose(2,1), y, f
