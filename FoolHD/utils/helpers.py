import argparse, os, torch

from utils.datasets import GeneralPurposeDataset as Dataset
from utils.losses import ContrastiveLoss

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-task',		   default='gae', 				   help="Which task to run. (e.g. gae, spk)")
	parser.add_argument('-configuration',  default='conf/parameters.conf', help="Configuration file")
	parser.add_argument('-load_model', 	   dest='load_model', 	  action="store_true")
	parser.add_argument('-train', 		   dest='train', 	  	  action="store_true")
	parser.add_argument('-test', 		   dest='test', 	  	  action="store_true")
	parser.add_argument('-encode', 		   dest='encode', 	      action="store_true")
	parser.add_argument('-use_multi_gpus', dest='use_multi_gpus', action="store_true")
	parser.add_argument('--start', type=int, required=True)
	parser.add_argument('--end', type=int, required=True)

	opt = parser.parse_args()
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

