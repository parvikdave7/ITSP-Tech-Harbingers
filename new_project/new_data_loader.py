import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import pdb

train_batch = 32
# test_batch = 32

def load_dataset(transform):
	
	# data_path = '/home/ojas/Desktop/itsp/project/new_dataset'
	data_path = '/home/ojas/Desktop/itsp/project_1/liveness-detection-opencv/dataset'
	
	train_dataset = datasets.ImageFolder(
		root = data_path,# + '/training',
		transform = transform)

	for image_num, path in enumerate(train_dataset.samples[:]):
		if path[0].split('/')[-1][:2] == '._':
			arr = path[0].split('/')
			arr[-1] = arr[-1][2:]
			arr = '/'.join(arr)
			tup = (str(arr),train_dataset.samples[image_num][1])
			train_dataset.samples[image_num] = tup

	# pdb.set_trace()

	train_loader = DataLoader(train_dataset,
		batch_size = train_batch,
		num_workers = 0,
		shuffle = True)
		
	return train_loader