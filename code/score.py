import argparse, os, scipy

import pickle as pkl
import numpy as np

import torch, torchaudio

from os import listdir
from tqdm import tqdm

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

torch.cuda.set_device(0)

def LoadAudio(audio_name, label_dict):

	# Read audio
	wrong_id 			  = audio_name.split("/")[-2]
	waveform, sample_rate = torchaudio.load(audio_name)
	true_id 			  = torch.LongTensor([label_dict[wrong_id]])

	return waveform.unsqueeze(dim=0).transpose(2,1).to("cuda:0"),true_id


def PredictClass(X,y, classifier):

	extractor = mfcc_extractor()

	# Feed to the classifier
	x_mfccs_vad, x_mfccs, labels = *extractor(X.transpose(1,2), before_vad=True),y
	clean_logits 				 = classifier.forward(x_mfccs_vad)
	clean_probs  				 = F.log_softmax(clean_logits,dim=-1).data

	clean_probs_sorted, clean_idx_sorted = clean_probs.sort(1, True)

	clean_class  	 = clean_idx_sorted[:,0]
	clean_class_prob = clean_probs_sorted[:,0]
	
	return (clean_class.cpu().numpy()[0], clean_class_prob.cpu().numpy()[0])

def main():

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, required=True)
	
	args   = parser.parse_args()
	
	adv_root = '/samples/{}/'.format(args.method)
	org_root = '/samples/original/'

	subs_adv_root = [dI for dI in os.listdir(adv_root) if os.path.isdir(os.path.join(adv_root,dI))]
	subs_adv_root.sort()

	all_audio_list = []
	for sub_adv_root in subs_adv_root:
		audio_list 	   = [f for f in listdir(adv_root+sub_adv_root+'/') if not f.startswith('.')]
		all_audio_list = all_audio_list + audio_list

	all_audio_list.sort()

	classifier = torch.load('external/pretrained_models/tdnn_state_dict.pth')	
	classifier.cuda()
	classifier.eval()
	
	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))

	sr_per_audio=0
	acc_per_clean_audio=0
	acc_per_adv_audio=0

	num_files = len(all_audio_list)
	for idx in tqdm(range(num_files)):

		org_audio,true_id = LoadAudio(org_root + '/' + all_audio_list[idx].split('_')[0] 
											   + '/' + "_".join(all_audio_list[idx].split('_')[0:-1]) 
											   + '.wav', label_dict)

		adv_audio,_       = LoadAudio(adv_root + '/' + all_audio_list[idx].split('_')[0] 
											   + '/' + all_audio_list[idx], label_dict)

		predicted_id_org, predicted_id_prob_org = PredictClass(org_audio,true_id,classifier)
		predicted_id_adv, predicted_id_prob_adv = PredictClass(adv_audio,true_id,classifier)
		
		true_id = true_id.cpu().numpy()[0]

		if (predicted_id_org != predicted_id_adv):
			sr_per_audio += 1

		if (predicted_id_org == true_id):
			acc_per_clean_audio += 1	

		if (predicted_id_adv == true_id):
			acc_per_adv_audio += 1			

	SR        = float(sr_per_audio) 	   / num_files
	ACC_Clean = float(acc_per_clean_audio) / num_files
	ACC_Adv   = float(acc_per_adv_audio)   / num_files

	print(SR, ACC_Adv, ACC_Clean)
	# Success Rate, Accuracy over adversarial samples, Accuracy over clean samples
	#       0.9898,		 					  0.06122, 						0.9286


if __name__ == '__main__':
	main()
