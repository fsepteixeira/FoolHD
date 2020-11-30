import torch
from utils.vad_cmvn import *
from utils.torchaudio_local import mfcc

class MFCCVadCMVNPadBatch(object):
	'''
	Class to pad batches of sequences - Also sorts by sequence length from shortest to largest
	: X - List of uneven sequence matrices
	: y - Corresponding labels to sort
	: (Optional) lengths - Numpy array containing the length of each sequence in X
	Returns:
	: X - Padded with zeros, sorted by sequence length
	: y - Sorted in the same way as X
	'''

	def __init__(self, num_ceps=30, num_mel_bins=30, 
					   low_freq=20, high_freq=4000, sampling_frequency=8000,
					   use_energy=True, raw_energy=True, snip_edges=False, 
					   remove_dc_offset=True, subtract_mean=False, 
					   dither=0.0, energy_floor=0.0, threshold=5.5, proportion_threshold=0.12, 
					   mean_scale=0.5, context=2, pad=True, mean=True, std=False,
					   mfcc_shift=1, window_length=300, step_size=1, collate=False):

		self.vad   = VAD(threshold=threshold, proportion_threshold=proportion_threshold, 
						 mean_scale=mean_scale, context=context, pad=pad)

		self.cmvn  = CMVN(mfcc_shift=mfcc_shift, window_length=window_length, 
						  step_size=step_size, mean=mean, std=std)

		self.mfccs = lambda x: mfcc(x, 
			num_ceps=num_ceps, 
			num_mel_bins=num_mel_bins, 
			low_freq=low_freq, 
			high_freq=high_freq, 
			use_energy=use_energy, 
			raw_energy=raw_energy, 
			snip_edges=snip_edges,
			remove_dc_offset=remove_dc_offset,
			subtract_mean=subtract_mean,
			dither=dither,
			energy_floor=energy_floor,
			sample_frequency=sampling_frequency,
			device=x.device
		)

		self.collate = collate

	def __call__(self, data, before_vad=False):
	
		# Get data arrays
		if self.collate:
			(X, y, f) = zip(*data)		
		else:
			X = data

		if len(X) < 1:
			if not collate:
				return []
			else:
				return [], [], []

		# Extract mfccs
		X_out = torch.stack([self.mfccs(X[i]) for i in range(0, len(X))])
		if before_vad:
			X_mfcc = X_out

		# Compute VAD and pad sequence
		X_out = self.vad(X_out)

		# Perform CMVN
		X_out = self.cmvn(X_out)
		
		if self.collate:
			return_list = []
			if type(y[0]) != str:
				return_list = [X_out, torch.LongTensor(y)]
			else:
				return_list = [X_out, y]
			if before_vad:
				return_list.append(X_mfcc)	
			return_list.append(f)
			return tuple(return_list)
		else:
			if not before_vad:
				return X_out
			else:
				return X_out, X_mfcc
