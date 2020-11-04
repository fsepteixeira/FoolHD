## FoolHD: Fooling speaker identification by Highly imperceptible adversarial Disturbances
_Ali Shahin Shamsabadi, Francisco Sepúlveda Teixeira, Alberto Abad, Bhiksha Raj, Andrea Cavallaro, Isabel Trancoso_


Speaker identification models are vulnerable to carefully designed adversarial perturbations of their input signals that induce misclassification. 
In this work, we propose a white-box steganography-inspired adversarial attack that generates imperceptible adversarial  perturbations against a speaker identification model.
Our approach, FoolHD, uses a Gated Convolutional Autoencoder that operates in the DCT domain and is trained with a multi-objective loss function, in order to generate and conceal the adversarial perturbation within the original audio files. In addition to hindering speaker identification performance, this multi-objective loss accounts for human perception through a frame-wise cosine similarity between MFCC feature vectors extracted from the original and adversarial audio files. We validate the effectiveness of FoolHD with a 250-speaker identification x-vector network, trained using VoxCeleb, in terms of accuracy, success rate, and imperceptibility.
Our results show that FoolHD generates highly imperceptible adversarial audio files (average PESQ scores above 4.30), while achieving a success rate of 99.6% and 99.2% in misleading the speaker identification model, for untargeted and targeted settings, respectively.

<p align="center"><img src="include/BlockDiagram.png" alt="Block diagram" title="Block diagram of the proposed approach." width="60%" heigh="60%"/></p>
<p align="center"><b>Fig 1. Block diagram of the proposed approach.</b></p>

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
python main.py -configuration conf/parameters_train.conf 
               -task targeted 
               -start 0 
               -end 10
```
### Adversarial audio samples
Please find our adversarial samples [here](include/samples.html).

<audio controls>
  <source src="samples/original/id00012/id00012_21Uxsk56VDQ_00006_00000.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

### Reference
If you would like to cite our work, please use:
```
@article{shamsabadi2020fool,
  title={FoolHD: Fooling speaker identification by Highly imperceptible adversarial Disturbances},
  author={Shamsabadi, Ali S. and Teixeira, Francisco S. and Abad, Alberto  and Raj, Bhiksha and Cavallaro, Andrea and Trancoso, Isabel},
  journal={arXiv preprint arXiv:!!!!!!},
  year={2020}
}
```
