from __future__ import division
from datetime import datetime

import numpy as np
np.random.seed(124)

import pickle as pkl
import pandas as pd

import torch
import torch.nn as nn
torch.manual_seed(124) 

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from torch.utils.data import DataLoader
from utils.datasets  import *
from utils.padder    import PadBatch
from utils.extractor import MFCCVadCMVNPadBatch as mfcc_extractor

class AdvModel(nn.Module):
	def __init__(self, model, extractor):
		super(AdvModel, self).__init__()
		self.model = model
		self.extractor = extractor

	def forward(self, X):
		return self.model(self.extractor(X.transpose(1,2)))

def createLogFiles(log_name=None):
	f_name = 'logs/fgsm/log_fgsm_{}.log'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
	f_file = open(f_name,"w")
	return f_file, f_name
	
def test_fgsm(adv_model, dataset, loss_fn, optimizer, batch_size=32, num_workers=20, device='cuda:0', attack='fgsm', **kwargs):
	
	"""
	Train the model with the given training data
	:param x:
	:param y:
	:param epochs:
	"""

	epsilons =[0.00001, 0.0001, 0.004, 0.01, 0.1, 1, 10, 100] 
	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))

	extractor = mfcc_extractor(collate=False)
	adv_classifier = PyTorchClassifier(model=AdvModel(adv_model.cpu(), extractor.cpu()),
										loss=loss_fn,
										optimizer=optimizer,
										input_shape=[1, 32000],
										nb_classes=250)
	# Create Dataloader
	dataloader = DataLoader(dataset=dataset['eval'],
	  			batch_size=batch_size, 
				shuffle=False,
				num_workers=num_workers,
				collate_fn=PadBatch())

	n_iterations = len(dataloader)

	f_log_all, f_name_all = createLogFiles('all')
	with open(f_name_all, 'a+') as f_log_all:
		f_log_all.write("\n\n #################################### Begin #####################################")
		f_log_all.write("\n New Log: {}".format(datetime.now()))

	# Loop over all the training data for generator	
	n_files = 0
	accuracy = 0
	adv_acc_eps = {e: 0.0 for e in epsilons}
	success_eps = {e: 0.0 for e in epsilons}
	for i, (X, y, f) in enumerate(dataloader):
		
		if label_dict:
			y = torch.LongTensor([label_dict[y_] for y_ in y])

		# send data to the GPU
		y = y.to(device)

		x_mfccs, labels = extractor((X.to(device).transpose(1,2))), y
		clean_logits = adv_model.forward(x_mfccs)
		clean_class  = clean_logits.argmax(dim=-1)

		n_files 	 += len(X)
		tmp_accuracy = torch.sum(clean_class == y).detach().cpu()
		accuracy 	 += tmp_accuracy

		# Epsilon loop
		for e in epsilons:
	
			# FGSM
                        if attack == 'fgsm':
        		    attack = FastGradientMethod(estimator=adv_classifier, eps=e)
                        elif attack == 'bim':
                            attack = ProjectedGradientDescent(estimator=adv_classifier, eps=e, eps_step=e/5, max_iter=100)

			X_fgsm = torch.Tensor(attack.generate(x=X)).to(device)

			assert(len(X_fgsm) == len(X))

			pred_mfccs, labels_preds = extractor(X_fgsm.transpose(1,2)), y
			adv_logits = adv_model.forward(pred_mfccs)
			adv_class  = adv_logits.argmax(dim=-1)

			tmp_success = torch.sum(clean_class != adv_class).detach().cpu()
			tmp_adv_acc = torch.sum(y           == adv_class).detach().cpu()

			success_eps[e] += tmp_success
			adv_acc_eps[e] += tmp_adv_acc			

			# Update total loss and acc
			with open(f_name_all, 'a+') as f_log_all:
				f_log_all.write('File {}\tBatch {}\tEps {}\tTarg {}\tClean {}\tAdv {}\n'.format(
					f[0][-1], i+1, e, y.cpu().detach().numpy(), 
					clean_class.cpu().detach().numpy(),
					adv_class.cpu().detach().numpy()))
			
			for wav, fi in zip(X_fgsm, f):
				adv_path="samples/fgsm/{}".format(fi[-2])
				if not os.path.exists(adv_path):
					os.makedirs(adv_path)
				torchaudio.save("{}/{}_{}.wav".format(adv_path,fi[-1], e),  wav.squeeze().detach().cpu(), 8000)

			print("Epsilon: {}".format(e),
				  "Tmp Acc: {:.3f}".format((tmp_accuracy + 0.0) / len(X)),
				  "Tmp Adv: {:.3f}".format((tmp_adv_acc + 0.0)  / len(X)),
				  "Tmp Suc: {:.3f}".format((tmp_success + 0.0)  / len(X)))

	accuracy        = (accuracy + 0.0) / n_files
	adv_acc_eps     = {k : v / n_files for k, v in adv_acc_eps.items()}
	success_eps     = {k : v / n_files for k, v in success_eps.items()}


	with open(f_name_all, 'a+') as f_log_all:
		f_log_all.write('Epsilons: {} - Accuracy: {}%\tAdv Accuracy: {}%\tSuccess rate: {}%\n'.format(e, accuracy, adv_acc_eps, success_eps))

	return


