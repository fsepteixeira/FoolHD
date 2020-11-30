import argparse

import numpy as np
np.random.seed(124)

import torch
import torch.nn as nn
torch.manual_seed(124) 

from models.GatedAutoEncoderModel import GatedAutoEncoder as GAEModel
from models.tdnn_model 			  import TDNNetwork 	  as ADVModel

from utils.losses   import ContrastiveLoss
from utils.datasets import GeneralPurposeDataset as Dataset
from utils.helpers  import arg_parser, create_datasets

from utils.nn_gae_untargeted import train_gae as train_gae_untargeted
from utils.nn_gae_untargeted import compile_gae as compile_gae_untargeted

from utils.nn_gae_targeted import train_gae as train_gae_targeted
from utils.nn_gae_targeted import compile_gae as compile_gae_targeted

def read_parameters(path):
	parameters = eval(open(path, 'r').read())
	for k, v in parameters.items():
		globals()[k] = v
	return parameters

def main():

	# Get arguments
	opt = arg_parser()

	print("==================SETUP================")
	# Get paramters	
	params = read_parameters(opt.configuration)
	print(params)

	print("\n============CUDA INFORMATION============")
	print("Cuda: ",torch.cuda.is_available())

	if torch.cuda.is_available():
		device = "cuda:0"
		print("Using device:", device)
	
	print("\n============CREATE DATASET============")
	dataset = create_datasets(**params)
	print("Dataset created.")
	
	print("\n============CREATE MODELS============")
	adv_model = ADVModel(**params['parameters_adv'])
	adv_model = torch.load(params['model_adv_path'])
	adv_model.to(device)

	if opt.task == 'untargeted' or opt.task == 'targeted':
		gae_model = GAEModel(**params['parameters_gae'])

	print("\n============COMPILATION============")
	if opt.task == 'untargeted':
		_, gae_optimizer = compile_gae_untargeted(gae_model, **params)
	elif opt.task == 'targeted':
		_, gae_optimizer = compile_gae_targeted(gae_model,   **params)
	else:
		raise ValueError("Task {} not yet implemented".format(opt.task))

	print("Done!")
	print()

	if opt.load_model:
		print("==================Load Model================")
		gae_model.load_state_dict(torch.load(params['model_gae_path']))
	gae_model.to(device)

	print("==================Creating adversarial example=================")
	if opt.task == 'untargeted':
		adv_model.eval()
		train_gae_untargeted(gae_model, adv_model, dataset, gae_optimizer, opt.start, opt.end, **params)

	elif opt.task == 'targeted':
		adv_model.eval()
		train_gae_targeted(gae_model, adv_model, dataset, gae_optimizer, opt.start, opt.end, **params)

	else:
		raise ValueError("Task {} not yet implemented".format(opt.task))

if __name__ == "__main__":
	main()
