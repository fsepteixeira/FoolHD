import os
import torch

import torchaudio
from torch.utils.data import Dataset

def walk_files_general(root, prefix=True, suffix=None, remove_suffix=False):
    root = os.path.expanduser(root)

    for folder, _, files in os.walk(root):
        for f in files:
            if suffix:
                if f.endswith(suffix):

                    if remove_suffix:
                        f = f[:-len(suffix)]
                    if prefix:
                        f = os.path.join(folder, f)

                    yield f
            else:
                if prefix:
                    f = os.path.join(folder, f)

                yield f

def load_gpd_item(fileid, path, ext_audio):

    file_audio = os.path.join(path, fileid + ext_audio)

    # Load audio
    try:
        waveform, sample_rate = torchaudio.load(file_audio)
    except:
        raise ValueError("Error reading file: ", str(file_audio))

    return (
        waveform,
        sample_rate,
        fileid,
    )

def load_voxceleb_item(fileid, path, ext_audio):

    file_audio = os.path.join(path, fileid + ext_audio)
    speaker_id = fileid.split("/")[-2]

    # Load audio
    try:
        waveform, sample_rate = torchaudio.load(file_audio)
    except:
        print("\nError reading file: ", str(file_audio))
        return [], None, None

    return (waveform, speaker_id, fileid.split("/"))    
        
class GeneralPurposeDataset(Dataset):

    _ext_audio = ".wav"

    def __init__(self, root, folder="", partition="", load_item_fn=load_gpd_item, load_fn='voxceleb', **kwargs):

        folder_in_archive = os.path.join(folder, partition)
        self._path = os.path.join(root, folder_in_archive)

        walker = walk_files_general(self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True)
        self._walker = list(walker)
        
        self._walker.sort()
        self._walker=[f for f in self._walker if f.split('/')[-1][0]!='.']
        
        if load_fn == 'voxceleb':
        	self.load_item_fn = load_voxceleb_item
        else:
            self.load_item_fn = load_item_fn
    

    def __getitem__(self, n):
        fileid = self._walker[n]
        return self.load_item_fn(fileid, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)

    def trim_dataset(self, start, end):
        self._walker = self._walker[start:end]    

