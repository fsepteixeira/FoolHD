import torch
import torch.nn.functional as F
import numpy as np

from scipy.signal import kaiser

def mdct_basis_(N):
    n0 = ((N//2) + 1) /2
    idx   = np.arange(0,N,1).reshape(N, 1)  
    kn    = np.multiply(idx + n0,(idx[:(N//2),:] + 0.5).T)
    basis = np.cos((2*np.pi/N)*kn)
    return torch.FloatTensor(basis.T)

def kbd_window_(win_len, filt_len, alpha=4):
    window = np.cumsum(kaiser(int(win_len/2)+1,np.pi*alpha))
    window = np.sqrt(window[:-1] / window[-1])

    if filt_len > win_len:
        pad =(filt_len - win_len) // 2
    else:
        pad = 0

    window = np.concatenate([window, window[::-1]])
    window = np.pad(window, (np.ceil(pad).astype(int), np.floor(pad).astype(int)), mode='constant')
    return torch.FloatTensor(window)[:,None]

class MDCT(torch.nn.Module):
    def __init__(self, filter_length=640, window_length=None, **kwargs):
        """
        This module implements an MDCT using 1D convolution and 1D transpose convolutions.
        This code only implements with hop lengths that are half the filter length (50% overlap
        between frames), to ensure TDAC conditions and, as such, perfect reconstruction. 
       
        Keyword Arguments:
            filter_length {int} -- Length of filters used - only powers of 2 are supported (default: {1024})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
        """
        super(MDCT, self).__init__()
        
        self.filter_length = filter_length
        assert((filter_length % 2) == 0)

        self.hop_length    = filter_length // 2  
        self.window_length = window_length if window_length else filter_length
        self.pad_amount    = filter_length // 2

        # get window and zero center pad it to filter_length
        assert(filter_length >= self.window_length)
        self.window = kbd_window_(self.window_length, self.filter_length, alpha=4)

        forward_basis = mdct_basis_(filter_length)
        forward_basis *= self.window.T

        inverse_basis = forward_basis.T

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data, **kwargs):
        """Take input data (audio) to MDCT domain.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)
        """

        # Pad data with win_len / 2 on either side
        num_batches, num_samples = input_data.size()
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (np.ceil(self.pad_amount).astype(int), np.floor(self.pad_amount).astype(int),0,0), mode='constant')
        input_data = input_data.squeeze(1)

        output = F.conv1d(input_data, 
                    self.forward_basis.unsqueeze(dim=1), 
                    stride=self.hop_length, padding=0)

        # Return magnitude -> MDCT only includes real values
        return output

    def inverse(self, magnitude, **kwargs):
        """Call the inverse MDCT (iMDCT), given magnitude and phase tensors produced 
        by the ```transform``` function.
        
        Arguments:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)
        
        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        inverse_transform = F.conv_transpose1d(magnitude, 
                            self.inverse_basis.unsqueeze(dim=1).T, 
                            stride=self.hop_length, padding=0)

        return (inverse_transform[..., np.ceil(self.pad_amount).astype(int):-np.floor(self.pad_amount).astype(int)]).squeeze(1)*(4/self.filter_length)

    def forward(self, input_data, **kwargs):
        """Take input data (audio) to MDCT domain and then back to audio.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        magnitude = self.transform(input_data)
        reconstruction = self.inverse(magnitude)
        return reconstruction
