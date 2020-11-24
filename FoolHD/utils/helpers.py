import argparse, os, torch

from utils.datasets import GeneralPurposeDataset as Dataset
from utils.losses import ContrastiveLoss

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', default='untargeted', choices=['untargeted', 'targeted', 'bim', 'fgsm'], 
						help="Which task to run. (e.g. untargeted, tageted, bim, fgsm)")

	parser.add_argument('-load_model', 	   dest='load_model', 	  action="store_true", help="Whether to load pre-trained model.")
	parser.add_argument('-configuration',  default='conf/parameters_untargeted.conf',  help="Configuration file")
	parser.add_argument('-create_adversarial_examples', dest='create_adversarial_examples', action="store_true")
	parser.add_argument('-score_adversarial_examples',  dest='score_adversarial_examples',  action="store_true")
	parser.add_argument('--start', type=int, default=0,  help="Dataset sample start index (0 first/-1 last) - Useful to compute adversarial samples in parallel.")
	parser.add_argument('--end',   type=int, default=-1, help="Dataset sample end index   (0 first/-1 last) - Useful to compute adversarial samples in parallel.")

	opt = parser.parse_args()
	opt.start = int(opt.start)
	opt.end   = int(opt.end)
	
	return opt

def create_datasets(root, folder, partition, load_fn, **kwargs):
	return Dataset(root=root, load_fn=load_fn)

def createLogFiles():
	log_path = 'results/logs/'
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	log_name = log_path+'log.txt'
	logFile = open(log_name,"w")

	return log_name

