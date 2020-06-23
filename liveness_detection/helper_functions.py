from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import time
import os
import copy
import pdb

class Livenet(nn.Module):
	def __init__(self,):
		super(Livenet, self).__init__()
		
		# Layers
		self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 16,kernel_size = 3,padding = 1)
		self.bn1 = nn.BatchNorm2d(num_features = 16,eps = 0.001,momentum = 0.5)
		self.conv2 = nn.Conv2d(16,16,kernel_size = 3,padding = 1)
		self.bn2 = nn.BatchNorm2d(num_features = 16,eps = 0.001,momentum = 0.5)
		self.drop1 = nn.Dropout(p = 0.25)

		self.conv3 = nn.Conv2d(16,32,kernel_size = 3,padding = 1)
		self.bn3 = nn.BatchNorm2d(num_features = 32,eps = 0.001,momentum = 0.5)
		self.conv4 = nn.Conv2d(32,32,kernel_size = 3,padding = 1)
		self.bn4 = nn.BatchNorm2d(num_features = 32,eps = 0.001,momentum = 0.5)
		self.drop2 = nn.Dropout(p = 0.25)

		self.flatten = nn.Flatten()
		self.dense1 = nn.Linear(32*8*8,64)
		self.bn5 = nn.BatchNorm1d(num_features = 64,eps = 0.001,momentum = 0.5)
		self.drop3 = nn.Dropout(p = 0.5)

		self.dense2 = nn.Linear(64,2)

	def forward(self,x):
		
		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)
		
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn2(x)

		x = F.max_pool2d(x,2)
		x = self.drop1(x)


		x = self.conv3(x)
		x = F.relu(x)
		x = self.bn3(x)
		
		x = self.conv4(x)
		x = F.relu(x)
		x = self.bn4(x)

		x = F.max_pool2d(x,2)
		x = self.drop2(x)

		x = self.flatten(x)
		x = self.dense1(x)
		x = self.bn5(x)
		x = self.drop3(x)

		x = self.dense2(x)

		return F.softmax(x,dim = 1)

# def train_model(model,dataloaders,criterion,optimizer,opt_schedule,num_epochs):
def train_model(model,dataloaders,criterion,optimizer,num_epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	since = time.time()
	val_acc_history = []
	train_acc_history = []

	# model_wts = copy.deepcopy(model.state_dict())

	last_model_wts = copy.deepcopy(model.state_dict())
	last_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		print('-' * 10)

        # Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			input_no = 0
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			# pdb.set_trace()
			pdb.set_trace()
			for inputs, labels in dataloaders[phase]:
				input_no += 1
				if input_no%30 == 0:
					print('[ {} progress = {} % ]'.format(
						phase,100*8*input_no/len(dataloaders[phase].dataset)))
				inputs = inputs.to(device)

				# Normalizing every image individually
				# for i,image in enumerate(inputs):
				# 	inputs = inputs/torch.max(inputs[0])

				# print(torch.max(inputs),torch.min(inputs))

				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					
					# print('model : ',model)
					# print('input shape : {}'.format(inputs.shape))
					# print(labels.shape)
					# print(inputs.shape)
					# print(model)

					# print(torch.max(inputs[0]), torch.min(inputs[0]))
					# exit(0)
					outputs = model(inputs)
					# print(outputs)
					loss = criterion(outputs, labels)

					# if(phase == 'val'):
					# 	print(outputs[0])
					
					if phase == 'train':
						loss.backward()
						optimizer.step()						

					_, preds = torch.max(outputs, 1)


				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			# confusion matrix
			confusion_stack = torch.stack((labels,preds),dim = 1)
			confusion = torch.zeros(2,2,dtype = torch.int32)
			# print(labels.shape,preds.shape)
			for elem in confusion_stack:
				label,pred = elem.tolist()
				confusion[label,pred] = confusion[label,pred] + 1
			print(confusion)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'val':
				val_acc_history.append(epoch_acc)
				# acc = epoch_acc
				# model_wts = copy.deepcopy(model.state_dict())
				if (epoch_acc > last_acc or epoch_acc > 0.98):
					last_acc = epoch_acc
					last_model_wts = copy.deepcopy(model.state_dict())

			if phase == 'train':
				train_acc_history.append(epoch_acc)


		print()

	# opt_schedule.step()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	
	print('last val Acc: {:4f}'.format(last_acc))

	# load last model weights
	model.load_state_dict(last_model_wts)

	# load model weights
	# model.load_state_dict(model_wts)
	return model, [train_acc_history, val_acc_history]
