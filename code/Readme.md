# FoolHD
This is the official PyTorch implementation of [FoolHD: Fooling speaker identification by Highly imperceptible adversarial Disturbances](https://arxiv.org/pdf/2011.08483.pdf).

### Setup
1. Download source code from GitHub
  ``` 
  git clone https://github.com/fsept11/FoolHD.git 
  ```
2. Go to FoolHD code directory
  ``` 
  cd FoolHD/code 
  ```
3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
  ```
  module load python3/anaconda
  conda create --name FoolHD 
  ```  
4. Activate conda environment 
  ```
  source activate FoolHD 
  ```
5. Install requirements
  ```
  pip install -r requirements.txt
  ```
6. Set the path to the dataset in line 106 of configuration file

   
### Description
This code generates both targeted and untargeted adversarial audio files by training a gated convolutional autoencoder in the MDCT domain using a perceptual loss and an adversarial loss.


### Generate FoolHD adversarial audio files
```
python main.py [-h] [-task {untargeted,targeted}] [-load_model]
               [-configuration CONFIGURATION]
               [--start START] [--end END]

optional arguments:
  -h, --help            show this help message and exit
  -task {untargeted,targeted}
                        Which task to run. (e.g. untargeted, tageted)
  -load_model           Whether to load pre-trained model.
  -configuration CONFIGURATION
                        Configuration file - This file needs to be modified in order to accomodate for data path changes. 
  --start START         Dataset sample start index (0 first/-1 last) - Useful
                        to compute adversarial samples in parallel.
  --end END             Dataset sample end index (0 first/-1 last) - Useful to
                        compute adversarial samples in parallel.
```

### Outputs
* Adversarial audio files saved with the same name as the original audio files in FoolHD/samples directory; this directory can be changed in the configuration file.
* Metadata with the following structure: filename, ground_truth identity, original predicted identity, adversarial predicted identity.

### Acknowledgments
The authors would like to thank and acknowledge [Thomas Rolland](https://github.com/Usanter) and Catarina Botelho for their contributions in the development of the Pytorch implementation of the x-vector network used in this work.
