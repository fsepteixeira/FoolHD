from tqdm import tqdm
import pickle as pkl
import numpy  as np
np.random.seed(124)

import torch
import torch.nn as nn
torch.manual_seed(124) 

from torch.utils.data import DataLoader
from .extractor  import MFCCVadCMVNPadBatch as PadBatch

from sklearn.metrics import accuracy_score

def compile_adv(model, loss_function=torch.nn.CrossEntropyLoss, optimization_fn=torch.optim.Adam, learning_rate=0.001, weight_decay=0, adapt=False, **kwargs):

	# Init loss function
	loss = loss_function()

	adapt = False
	if not adapt:
		optimizer = optimization_fn(model.parameters(),
							    lr=learning_rate,
							    weight_decay=weight_decay)	
	else:
		optimizer = optimization_fn([{'params':model.module.linear.parameters()}, 
									 {'params':model.module.output_layer.parameters()}],
									 lr=learning_rate,
								     weight_decay=weight_decay)
	return loss, optimizer

def train_adv(model, dataset, optimizer, loss_fn, params, epochs=10, batch_size=32, lr_decay=0.1, period=21,
			  num_workers=20, shuffle=False, begin=0, save_path="", save_intermediate_path="", device="cuda:0", adapt=False, **kwargs):
		
	"""
	Train the model with the given training data
	:param x:
	:param y:
	:param epochs:
	"""

	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))

	# Create Dataloader
	dataloader = DataLoader(dataset=dataset['train'], 
							batch_size=batch_size,
							shuffle=shuffle,
							num_workers=num_workers,
							collate_fn=PadBatch(collate=True))

	n_iterations = len(dataloader)

	#if adapt:
	#	model = freeze_layers(model)

	model.train()

	# Epoch loop
	for e in range(begin, epochs):

		torch.cuda.empty_cache()
		epoch_loss	   = 0
		epoch_accuracy = 0

		# Reduce the learning rate
		if e % period == (period-1):
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * lr_decay

		# Loop over all the training data
		model.train()
		number_of_samples = 0
		for i, data in enumerate(dataloader):

			# Get the data from the dataloader
			X, y, _ = data

			if label_dict:
				y = torch.LongTensor([label_dict[y_] for y_ in y])

			# send data to the GPU
			X = X.to(device)
			y = y.to(device)

			# Foward pass
			if alpha_scheduler:
				prediction = model.forward(x=X, alpha=alphas[e])
			else:
				prediction = model.forward(X)

			# Compute the loss
			loss = loss_fn(prediction, y)

			# Update total loss and acc
			epoch_loss     += loss.detach().cpu().item()
			epoch_accuracy += torch.sum(prediction.argmax(dim=-1) == y).detach().cpu().item()
			
			# Update weights
			model.zero_grad()
			loss.backward()
			optimizer.step()

			# Update number of samples
			number_of_samples += len(y.detach())

			# Plot useful information
			print("Epoch", e, " - Batch ", i+1, "/", n_iterations, 
				  " - Train Accuracy: ", "{0:4f}".format(epoch_accuracy/number_of_samples),
				  " || Batch Loss: ", "{0:4f}".format(loss.detach().cpu().item()), end='\r')

			del(X)
			del(y)
			del(loss)
			del(prediction)
			torch.cuda.empty_cache()
	
			if i % 100 == 0:
				try :
					torch.save(model, save_intermediate_path)
				except BaseException as err:
					print("Failed to save model at batch", i+1, "as an exception occured:", '{}'.format(err))

		# Compute avg loss and accuracy for this epoch
		if number_of_samples != 0:
			epoch_loss	   /= n_iterations
			epoch_accuracy /= number_of_samples
		else:
			epoch_loss	   = 0.0
			epoch_accuracy = 0.0
				
		# Save the model
		try :
			torch.save(model, save_intermediate_path.split('.')[0] + "_ep" + str(e) + ".pth")
		except BaseException as err:
			print("Failed to save model at epoch", e, "as an exception occured:", '{}'.format(err))

		if kwargs['evaluate_each_epoch']:
			# Compute loss and acc for val
			dev_accuracy, _ = score_adv(model, dataset['devel'], loss_fn=loss_fn, batch_size=batch_size, num_workers=num_workers, device=device)

			# Plot final epoch informations
			print("Epoch ", e, "   Train Accuracy: ", epoch_accuracy," || Train Loss: ", epoch_loss, " || Development Accuracy:", dev_accuracy)
		else:
			print("Epoch ", e, "   Train Accuracy: ", epoch_accuracy," || Train Loss: ", epoch_loss)

		print()

	# Save the model
	try:
		torch.save(model, save_path)
	except BaseException as err:
		print("Failed to save final model as an exception occured:", '{}'.format(err))

	return model

def score_adv(model, dataset, loss_fn, batch_size=512, num_workers=20, device="cuda:0", **kwargs):

	"""
	Compute the accuracy and the loss value for a given dataset
	:param x: Dataset values
	:param y: Dataset labels
	"""
	label_dict = pkl.load(open('external/speaker2int_7323.pkl','rb'))

	# Define the dataloader
	dataloader = DataLoader(dataset,
					batch_size=batch_size,
					shuffle=False,
					num_workers=num_workers,
					collate_fn=PadBatch(collate=True))

	# Loop over all the data
	number_of_batches = len(dataloader)
	number_of_samples = 0
	accuracy = 0
	loss = 0

	# Set evaluation mode
	model.eval()

	# Init lists
	targets = []
	predictions = []

	# Iterate through data in batches
	for i, data in enumerate(dataloader):

		# Get the curent batch data
		X, y, f = data
		number_of_samples += len(y)

		# Send data_x and data_y to the device (GPU or CPU)
		X = X.to(device)
		if label_dict:
			y = torch.LongTensor([label_dict[y_] for y_ in y])

		y = y.to(device)
			
		# Foward pass
		prediction = model.forward(X)

		# Update scores
		accuracy += torch.sum(prediction.argmax(dim=-1) == y).detach().cpu().item()
		loss	 += loss_fn(prediction, y).detach().cpu().item()

		targets.extend(y.detach().cpu().tolist())			   
		predictions.extend(prediction.argmax(dim=-1).detach().cpu().tolist())		   
		del(X)
		del(y)
		del(prediction)
		torch.cuda.empty_cache()
		print("Batch", i+1, "/", len(dataloader), " - Accuracy: {}".format(accuracy_score(targets, predictions)), end="\r")

	print("Final accuracy: ", accuracy_score(targets, predictions))

	if number_of_samples != 0:
		return (accuracy/number_of_samples), (loss/number_of_batches)
	else:
		return 0.0, 0.0

