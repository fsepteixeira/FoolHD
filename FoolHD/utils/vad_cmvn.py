
import torch
from torch 				import nn
from torch.nn 			import functional as F
from torch.nn.utils.rnn import pad_sequence

from math import floor, ceil

"""
## KALDI VAD:

void ComputeVadEnergy(const VadEnergyOptions &opts,
                      const MatrixBase<BaseFloat> &feats,
                      Vector<BaseFloat> *output_voiced) {

  int32 T = feats.NumRows();
  output_voiced->Resize(T);

  if (T == 0) {
    KALDI_WARN << "Empty features";
    return;
  }

  Vector<BaseFloat> log_energy(T);
  log_energy.CopyColFromMat(feats, 0); // column zero is log-energy.

  BaseFloat energy_threshold = opts.vad_energy_threshold;
  if (opts.vad_energy_mean_scale != 0.0) {
    KALDI_ASSERT(opts.vad_energy_mean_scale > 0.0);
    energy_threshold += opts.vad_energy_mean_scale * log_energy.Sum() / T;
  }

  KALDI_ASSERT(opts.vad_frames_context >= 0);
  KALDI_ASSERT(opts.vad_proportion_threshold > 0.0 &&
               opts.vad_proportion_threshold < 1.0);
  for (int32 t = 0; t < T; t++) {
    const BaseFloat *log_energy_data = log_energy.Data();
    int32 num_count = 0, den_count = 0, context = opts.vad_frames_context;
    for (int32 t2 = t - context; t2 <= t + context; t2++) {
      if (t2 >= 0 && t2 < T) {
        den_count++;
        if (log_energy_data[t2] > energy_threshold)
          num_count++;
      }
    }
    if (num_count >= den_count * opts.vad_proportion_threshold)
      (*output_voiced)(t) = 1.0;
    else
      (*output_voiced)(t) = 0.0;
  }
}

"""

class VAD(nn.Module):

	def __init__(self, threshold=5.5, proportion_threshold=0.12, mean_scale=0.5, context=2, pad=False, **kwargs):

		self.threshold = threshold
		self.proportion_threshold = proportion_threshold
		self.mean_scale = mean_scale
		self.context = context
		self.diff_zero = mean_scale != 0
		self.unfold_size = 2 * context + 1
		self.pad = pad

	def __call__(self, input):

		assert(input.dim() == 3)

		energy_threshold = torch.tensor([self.threshold]).to(input.device)

		# Input should have shape (batch_size, seq_len, n_feats)
		batch_size, seq_len, n_feats = input.size()

		log_energy = input[:,:,0]			# Get the energy MFCC for all samples in
											# the batch and all frames in the sequence
		if self.diff_zero:
			energy_threshold = energy_threshold + self.mean_scale * log_energy.mean(dim=1)

		mask = torch.ones_like(log_energy)								# Mask with same shape as log_energy
		mask = F.pad(mask, pad=(self.context, self.context), value=1.0) # Pad borders with symmetric context
		mask = mask.unfold(dimension=1, size=self.unfold_size, step=1)	# Get all (overlapping) "windows" of "convolution"

		den_count = mask.sum(dim=-1)	# Number of values included in each window
										# Required due to padding, otherwise it would always be (2*context + 1)
		# Pad borders with symmetric context
		log_energy = F.pad(log_energy, pad=(self.context, self.context))				  

		# Get all (overlapping) "windows" of "convolution"
		log_energy = log_energy.unfold(dimension=1, size=self.unfold_size, step=1) 

		# Number of values in window above threshold
		num_count = log_energy.gt(energy_threshold.unsqueeze(-1).unsqueeze(-1)).sum(dim=-1)

		# If the number of frames in window above the
		# threshold is larger than den_count*cte  -> Voiced Frame
		# Else: unvoiced
		vad_frames = num_count.ge(den_count*self.proportion_threshold)
		# Remove unvoiced frames from input and return
		input = [torch.masked_select(input[i], mask=vad_frames[i].unsqueeze(-1)).reshape(-1, n_feats) for i in range(0,batch_size)]

		if self.pad:
			input = pad_sequence(input, batch_first=True)

		return input

class CMVN(nn.Module):

	def __init__(self, mfcc_shift=10, window_length=300, step_size=1, mean=True, std=False, **kwargs):
		self.mfcc_shift = mfcc_shift
		self.window_length = window_length
		self.frames = int(window_length / mfcc_shift)
		self.step_size = step_size
		self.mean = mean
		self.std = std

		if self.frames % 2 == 0:	# Only centered cmvn window considered
			self.frames += 1

	def __call__(self, input):

		if not self.mean and not self.std:
			return input

		# Input should have shape (batch_size, seq_len, n_feats)
		assert(input.dim() == 3)
		batch_size, seq_len, n_feats = input.size()

		if seq_len == 0:
			return input

		windows = F.pad(input, pad=(0, 0, floor((self.frames-1)/2), ceil((self.frames-1)/2), 0, 0)) # Pad borders with symmetric context -> Assuming centered window
		windows = windows.unfold(dimension=1, size=self.frames, step=self.step_size)  # Get all (overlapping) "windows" of "convolution"

		if self.mean:
			input -= windows.mean(dim=-1)

		if self.std:
			input /= windows.std(dim=-1)

		return input
