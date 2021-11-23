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
	f_name = '{}'.format(log_name)
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

def train_gae(gae_model, adv_model, dataset, gae_optimizer, idxBeg,    idxEnd,  
		      num_workers=20, shuffle=False, save_path_gae="", save_intermediate_path_gae="", 
			  log_file="logs/log.log", sample_dir="samples/", device='cuda:0', **kwargs):
	
	"""
	Train the model with the given training data
	:param x:
	:param y:
	:param epochs:
	"""
	gae_model.mdct.pad_amount=640

	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))
	start=idxBeg
	end=idxEnd
	
	if start != None or end != None:
		if start == None:
			start = 0
		if end == None:
			end = -1
		dataset.trim_dataset(start,end)

	# Create Dataloader
	train_dataloader = DataLoader(dataset=dataset,
				      batch_size=1, 
				      shuffle=False,
				      drop_last=True,
			    	      num_workers=num_workers,
				      collate_fn=PadBatch())
	gae_epoch_loss = 0
	gen_epoch_loss = 0
	adv_epoch_loss = 0
	number_of_samples = 0

	n_iterations = len(train_dataloader)

	adv_model.eval()
	extractor = mfcc_extractor()

	for param in adv_model.parameters():
		param.requires_grad = False

	# Loop over all the training data for generator	
	f_log_all, f_name_all   = createLogFiles('logs/'+log_file+'all_{}to{}.log'.format(idxBeg,idxEnd if idxEnd>=0 else n_iterations))
	f_log_loss, f_name_loss = createLogFiles('logs/'+log_file+'loss_{}to{}.log'.format(idxBeg,idxEnd if idxEnd>=0 else n_iterations))

	cos = nn.CosineSimilarity(dim=2, eps=1e-6)
	for i, (train_data, y_temp, f) in tqdm(enumerate(train_dataloader)):

		print(idxBeg+i,idxEnd, f[-1][-1], end="\r")
		if len(train_data) < 1:
			continue

		if label_dict:
			y = torch.LongTensor([label_dict[y_] for y_ in y_temp])
			
		# Get the data from the dataloader
		X = train_data

		# send data to the GPU
		X = X.to(device)
		x_mfccs_vad, x_mfccs, labels = *extractor(X.transpose(1,2), before_vad=True), y

		clean_logits = adv_model.forward(x_mfccs_vad)

		clean_probs  = F.log_softmax(clean_logits,dim=-1).data.squeeze(dim=1)
		clean_probs_sorted, clean_idx_sorted = clean_probs.sort(1, True)

		clean_class  	 = clean_idx_sorted[:,0]
		clean_class_prob = clean_probs_sorted[:,0]

		adv_path = sample_dir+"{}".format(f[0][-2])
		if not os.path.exists(adv_path):
			os.makedirs(adv_path)

		# ------------------
		# Train AutoEncoder
		# ------------------
		gae_model.train()
		adv_model.eval()
		gen_min_loss=100
		MaxItrs=1000
		rand = np.random.randint(0, high=250)
		target_class=torch.tensor(rand,device='cuda:0')
		while target_class == clean_class:
			rand = np.random.randint(0, high=250)
			target_class=torch.tensor(rand,device='cuda:0')
		text=None
		for itrs in range(MaxItrs):
				
				# Foward pass
				gae_prediction = gae_model.forward(X)
				gae_prediction = gae_prediction.squeeze(dim=1)

				# Perceptual loss	
				pred_mfccs_vad, pred_mfccs, labels_preds 		 = *extractor(gae_prediction.transpose(1,2), before_vad=True),y		
				Freq_loss = (1-cos(pred_mfccs, x_mfccs)).sum()

				
				adv_logits = adv_model.forward(pred_mfccs_vad)
				adv_probs  = F.log_softmax(adv_logits, dim=-1).data.squeeze(dim=1)
				adv_probs_sorted, adv_idx_sorted = adv_probs.sort(1, True)

				adv_class 	   = adv_idx_sorted[:,0]
				adv_class_prob = adv_probs_sorted[:,0]

				# Adversarial loss
				adversarial_loss = AdvLoss(250, is_targeted=True)
				adv_batch_loss = adversarial_loss.forward(adv_logits, target_class)
				gae_total_loss = Freq_loss+adv_batch_loss

				if (adv_class.cpu().detach().numpy()[0] == target_class.cpu().detach().numpy()): #and gae_total_loss<0.0007:
					if Freq_loss < gen_min_loss:
							
							f_log_all = open(f_name_all, 'a+')
							f_log_all.write('{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\n'.format(f[0][-1], itrs+1, y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0], gae_total_loss))
							f_log_all.close()
							torchaudio.save("{}/{}.wav".format(adv_path,f[0][-1]),  gae_prediction.squeeze().detach().cpu(), 8000)
							text=('{}\t{}\t{}\t{}\t{}'.format(f[0][-1], y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0]))
							gen_min_loss=gae_total_loss

			
				gae_model.zero_grad()
				gae_total_loss.backward()
				gae_optimizer.step()
				torch.save(gae_model.state_dict(), save_path_gae)
	
				print("Sample:", idxBeg + i,"of", idxEnd if idxEnd >= 0 else n_iterations, "-", f[-1][-1], "| Itr=" + str(itrs+1) + "/" + str(MaxItrs), "| Adv Loss:", "{:.3f}".format(adv_batch_loss.item()), "| Percpt Loss:", "{:.3f}".format(Freq_loss.item()), "\t\t\t\t\t\t", end="\r")
		f_log_loss = open(f_name_loss, 'a+')		
		if text == None:
			f_log_loss.write('{}\t{}\t{}\t{}\t{}\t0\n'.format(f[0][-1], y.cpu().detach().numpy()[0], clean_class.cpu().detach().numpy()[0], target_class.cpu().detach().numpy(), adv_class.cpu().detach().numpy()[0]))
		else:
			f_log_loss.write(text+'\t1\n')
		f_log_loss.close()

	return adv_class
