import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

train_batch = 32
test_batch = 32
lr = 0.01
momentum = 0.5
size = 320

# class dataset(Dataset):
#     def __init__(self,data,target,size,transform = None):
#         self.transform = transform
#         self.data = data.reshape((-1,size,size)).astype(np.float32)[:,:,:,None]
#         self.target = torch.from_numpy(target).long()

#     def __getitem__(self,index):
#         return self.transform(self.data[index]),self.target[index]

#     def __len__(self):
#         return len(list(self.data))

def load_dataset(dataset,transform):
	
	data_path = '/home/ojas/Desktop/itsp/project/dataset'
	if dataset == 'train':
		
		train_dataset = datasets.ImageFolder(
			root = data_path + '/training',
			transform = transform)

		for image_num, path in enumerate(train_dataset.samples[:]):
			if path[0].split('/')[-1][:2] == '._':
				arr = path[0].split('/')
				arr[-1] = arr[-1][2:]
				arr = '/'.join(arr)
				tup = (str(arr),train_dataset.samples[image_num][1])
				train_dataset.samples[image_num] = tup

		train_loader = DataLoader(train_dataset,
			batch_size = train_batch,
			num_workers = 0,
			shuffle = True)
		
		return train_loader
	
	if dataset == 'test':
	
		test_dataset = datasets.ImageFolder(
			root = data_path + '/testing',
			transform = transform)

		for image_num, path in enumerate(test_dataset.samples[:]):
			if path[0].split('/')[-1][:2] == '._':
				arr = path[0].split('/')
				arr[-1] = arr[-1][2:]
				arr = '/'.join(arr)
				tup = (str(arr),test_dataset.samples[image_num][1])
				test_dataset.samples[image_num] = tup
	
		test_loader = DataLoader(test_dataset,
			batch_size = test_batch,
			num_workers = 0,
			shuffle = False)
		
		return test_loader
