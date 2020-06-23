import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pdb

train_batch = 8
test_batch = 8
lr = 0.01
momentum = 0.5
size = 32

def load_dataset(dset,transform):
	
	data_path = '/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/project3_v2/live_dataset'
			
	dataset = datasets.ImageFolder(
		root = data_path,
		transform = transform)
	
	# for image_num, path in enumerate(dataset.samples[:]):
	# 	if path[0].split('/')[-1][:2] == '._':
	# 		arr = path[0].split('/')
	# 		arr[-1] = arr[-1][2:]
	# 		arr = '/'.join(arr)
	# 		tup = (str(arr),dataset.samples[image_num][1])
	# 		dataset.samples[image_num] = tup

	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [5413,1804])
		# pdb.set_trace()

	train_loader = DataLoader(train_dataset,
		batch_size = train_batch,
		num_workers = 0,
		shuffle = True)

	test_loader = DataLoader(test_dataset,
		batch_size = test_batch,
		num_workers = 0,
		shuffle = True)
	pdb.set_trace()

	if dset == 'train':	
		print('train input shape : {}'.format(train_dataset[0][0].shape))
		return train_loader
	if dset == 'test':
		print('test input shape : {}'.format(test_dataset[0][0].shape))
		return test_loader
