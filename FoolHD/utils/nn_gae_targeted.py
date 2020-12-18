import argparse, time

import numpy as np
np.random.seed(124)

from utils.mdct import MDCT as mdct

import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
torch.manual_seed(124) 

from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.datasets	 import *
from utils.vad_cmvn  import *
from utils.padder    import PadBatch
from utils.losses    import ContrastiveLoss
from utils.losses    import AdversarialLoss as AdvLoss
from utils.extractor import MFCCVadCMVNPadBatch as mfcc_extractor

from utils.torchaudio_local import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def createLogFiles(log_name):
	f_name = './log_{}.txt'.format(log_name)
	f_file = open(f_name,"w")
	return f_file, f_name
	
def compile_gae(gae_model, learning_rate=0.0001, weight_decay=0.0, 
			loss_fn_gae=torch.nn.MSELoss, optimization_fn=torch.optim.Adam, **kwargs):
		
	"""
	Define the loss and the optimizer (with the initial learning rate)
	:param learning_rate:
	"""
		
	# Define loss function
	loss_gae = loss_fn_gae()
		
	# Define the optimizer
	gae_optimizer = optimization_fn(gae_model.parameters(),
								lr=learning_rate, 
								weight_decay=weight_decay)
		
	return loss_gae, gae_optimizer

def train_gae(gae_model, adv_model, dataset, gae_optimizer, idx_begin, idx_end, n_speakers=250, max_iterations=1000,
		      save_path_gae="", log_file="logs/log.log", sample_dir="samples/", device='cuda:0', **kwargs):
	
	"""
	Train the model with the given training data
	:param x:
	:param y:
	:param epochs:
	"""

	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))
	start = idx_begin
	end   = idx_end
	
	if start != None or end != None:
		if start == None:
			start = 0
		if end == None:
			end = -1
		dataset.trim_dataset(start,end)

	# Create Dataloader
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

	cos = nn.CosineSimilarity(dim=2, eps=1e-6)
	adversarial_loss = AdvLoss(n_speakers, is_targeted=True)

	gae_epoch_loss = 0
	adv_epoch_loss = 0
	number_of_samples = 0

	n_iterations = len(dataloader)

	gae_model.train()
	adv_model.eval()
	extractor = mfcc_extractor()

	for param in adv_model.parameters():
		param.requires_grad = False

	# Loop over all the training data for generator	
	f_log_all, f_name_all   = createLogFiles('logs/' + log_file + 'all_{}to{}.log'.format(idx_begin,  idx_end if idx_end>=0 else n_iterations))
	f_log_loss, f_name_loss = createLogFiles('logs/' + log_file + 'loss_{}to{}.log'.format(idx_begin, idx_end if idx_end>=0 else n_iterations))

	for i, (X, y_temp, f) in tqdm(enumerate(dataloader)):

		if label_dict:
			y = torch.LongTensor([label_dict[y_] for y_ in y_temp])
			
		# Get the data from the dataloader and extract features
		X = X.to(device)
		x_mfccs_vad, vad, labels = *extractor(X.transpose(1,2), return_vad=True), y

		clean_logits = adv_model.forward(x_mfccs_vad)

		clean_probs = F.log_softmax(clean_logits,dim=-1).data.squeeze(dim=1)
		clean_class	= clean_probs.argmax(dim=1)
		clean_class_prob = clean_class[:, clean_class]

		adv_path = sample_dir + "{}".format(f[0][-2])
		if not os.path.exists(adv_path):
			os.makedirs(adv_path)

		# Select target class
		rand = np.random.randint(0, high=250)
		target_class = torch.tensor(rand,device='cuda:0')

		while target_class == clean_class:
			rand 		 = np.random.randint(0, high=250)
			target_class = torch.tensor(rand,device='cuda:0')

		# ------------------
		# Train AutoEncoder
		# ------------------

		text = None
		for itrs in range(max_iterations):
				
				# Foward pass
				gae_prediction = gae_model.forward(X)
				gae_prediction = gae_prediction.squeeze(dim=1)

				# Perceptual loss	
				pred_mfccs_vad, labels_preds = *extractor(gae_prediction.transpose(1,2), pre_computed_vad=True, vad=vad), y		
				freq_loss = (1 - cos(pred_mfccs_vad, x_mfccs_vad)).sum()
			
				adv_logits = adv_model.forward(pred_mfccs_vad)

				adv_probs  = F.log_softmax(adv_logits, dim=-1).data.squeeze(dim=1)
				adv_class 	   = adv_probs.argmax(dim=1)
				adv_class_prob = adv_probs[:,adv_class]

				# Adversarial loss
				adv_batch_loss   = adversarial_loss.forward(adv_logits, target_class)
				gae_total_loss   = freq_loss + adv_batch_loss

				if (adv_class.cpu().detach().numpy()[0] == target_class.cpu().detach().numpy()):
						f_log_all = open(f_name_all, 'a+')
						f_log_all.write('{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\n'.format(f[0][-1], itrs+1, y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0], gae_total_loss))
						f_log_all.close()
						torchaudio.save("{}/{}.wav".format(adv_path,f[0][-1]),  gae_prediction.squeeze().detach().cpu(), 8000)
						text=('{}\t{}\t{}\t{}\t{}'.format(f[0][-1], y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0]))
			
				gae_model.zero_grad()
				gae_total_loss.backward()
				gae_optimizer.step()
				torch.save(gae_model.state_dict(), save_path_gae)
	
				print("Sample:", start + i,"of", end if end >= 0 else n_iterations, "-", f[-1][-1], "| Itr=" + str(itrs+1) + "/" + str(max_iterations), "| Adv Loss:", "{:.3f}".format(adv_batch_loss.item()), "| Percpt Loss:", "{:.3f}".format(freq_loss.item()), "\t\t\t\t\t\t", end="\r")

		f_log_loss = open(f_name_loss, 'a+')
		if text == None:
			f_log_loss.write('{}\t{}\t{}\t{}\t{}\t0\n'.format(f[0][-1], y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0]))
		else:
			f_log_loss.write(text + '\t\n')
		f_log_loss.close()

	return adv_class
