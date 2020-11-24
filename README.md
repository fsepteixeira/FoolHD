## [FoolHD](https://fsept11.github.io/FoolHD/): Fooling speaker identification by Highly imperceptible adversarial Disturbances
<emph>Ali Shahin Shamsabadi<sup>1*</sup>, Francisco Sepúlveda Teixeira<sup>2*</sup>, Alberto Abad<sup>2</sup>, Bhiksha Raj<sup>3</sup>, Andrea Cavallaro<sup>1</sup>, Isabel Trancoso<sup>2</sup></emph>

<sup>1</sup>CIS, Queen Mary University of London, UK.  
<sup>2</sup>INESC-ID/IST, University of Lisbon, Portugal.  
<sup>3</sup>Carnegie Mellon University, USA.

<sup>\*</sup>Authors contributed equally.  

Speaker identification models are vulnerable to carefully designed adversarial perturbations of their input signals that induce misclassification. 
In this work, we propose a white-box steganography-inspired adversarial attack that generates imperceptible adversarial  perturbations against a speaker identification model.
Our approach, FoolHD, uses a Gated Convolutional Autoencoder that operates in the DCT domain and is trained with a multi-objective loss function, in order to generate and conceal the adversarial perturbation within the original audio files. In addition to hindering speaker identification performance, this multi-objective loss accounts for human perception through a frame-wise cosine similarity between MFCC feature vectors extracted from the original and adversarial audio files. We validate the effectiveness of FoolHD with a 250-speaker identification x-vector network, trained using VoxCeleb, in terms of accuracy, success rate, and imperceptibility.
Our results show that FoolHD generates highly imperceptible adversarial audio files (average PESQ scores above 4.30), while achieving a success rate of 99.6% and 99.2% in misleading the speaker identification model, for untargeted and targeted settings, respectively.

<p align="center"><img src="figs/BlockDiagram.png" alt="Block diagram" title="Block diagram of the proposed approach." width="60%" heigh="60%"/></p>
<p align="center"><b>Fig 1. Block diagram of the proposed approach.</b></p>

### Code available [here](https://github.com/fsept11/FoolHD/tree/main/FoolHD).

### Clone repository
``` 
git clone https://github.com/fsept11/FoolHD.git 
```
### Setup environment
```
conda create --name FoolHD --file requirements.txt
conda activate FoolHD 
```
### Usage
```
usage: main.py [-h] [-task {untargeted,targeted,bim,fgsm}] [-load_model]
               [-configuration CONFIGURATION] [-create_adversarial_examples]
               [-score_adversarial_examples] [--start START] [--end END]

optional arguments:
  -h, --help            show this help message and exit
  -task {untargeted,targeted,bim,fgsm}
                        Which task to run. (e.g. untargeted, tageted, bim,
                        fgsm)
  -load_model           Whether to load pre-trained model.
  -configuration CONFIGURATION
                        Configuration file - This file needs to be modified in order to accomodate for data path changes. 
  -create_adversarial_examples
  -score_adversarial_examples
  --start START         Dataset sample start index (0 first/-1 last) - Useful
                        to compute adversarial samples in parallel.
  --end END             Dataset sample end index (0 first/-1 last) - Useful to
                        compute adversarial samples in parallel.
```
### Adversarial audio samples
You can find our adversarial samples [here](http://fsept11.github.io/FoolHD/samples).

### Reference
If you would like to cite our work, please use:
```
@article{shamsabadi2020foolhd,
      title={FoolHD: Fooling speaker identification by Highly imperceptible adversarial Disturbances}, 
      author={Ali Shahin Shamsabadi and Francisco Sepúlveda Teixeira and Alberto Abad and Bhiksha Raj and Andrea Cavallaro and Isabel Trancoso},
      year={2020},
      eprint={2011.08483},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
