from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import time
import os
import copy
import pdb
from data_loader import load_dataset
from helper_functions import train_model, set_parameter_requires_grad, initialize_model

torch.cuda.empty_cache()

transform = transforms.Compose([transforms.Resize([320, 320]),
	transforms.ToTensor(),
	transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)])

dataloaders_dict = {'train': load_dataset('train',transform),
'val':load_dataset('test',transform)}

print('\n[Data loaded successfully]')
# print('Shape of the first five batches : ')

# for i, (inputs,data) in enumerate(dataloaders_dict['train']):
# 	if i < 5:
# 		print(inputs.shape,data.shape)
# 	else:
# 		break
# pdb.set_trace()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Details ------------------------------------------------------
model_name = "resnet"
num_classes = 2
feature_extract = True
num_epochs = 5

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained = True)

# Print the model we just instantiated
print(model_ft)

model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr = 0.001, momentum = 0.9)

# Setup the loss function
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs = num_epochs, is_inception = (model_name == "inception"))

torch.save(model_ft.state_dict(), '/home/ojas/Desktop/itsp/project/own/resnet.pth')