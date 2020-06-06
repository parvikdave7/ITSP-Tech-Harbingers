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
from new_data_loader import load_dataset
from new_helper_functions import train_model, set_parameter_requires_grad, initialize_model

torch.cuda.empty_cache()

size_x = 150
size_y = 220

transform = transforms.Compose([transforms.Resize([size_x,size_y]),
	transforms.ToTensor(),
    transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)])

dataloader = load_dataset(transform)

print('\n[Data loaded successfully]')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Details ------------------------------------------------------
model_name = "resnet"
num_classes = 2
feature_extract = True
num_epochs = 15

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained = True)

# Print the model we just instantiated
print(model_ft)

model_ft = model_ft.to(device)

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
optimizer_ft = optim.SGD(params_to_update, lr = 0.01, momentum = 0.5)

# Setup the loss function
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs = num_epochs, is_inception = (model_name == "inception"))

torch.save(model_ft.state_dict(), '/home/ojas/Desktop/itsp/project/models/resnet8.pth')