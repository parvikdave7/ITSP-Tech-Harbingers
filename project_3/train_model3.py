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
from data_loader3 import load_dataset
from helper_functions3 import Livenet, train_model
from pytorch_model_summary import summary

torch.cuda.empty_cache()

size_x = 32
size_y = 32

transform = transforms.Compose([transforms.RandomResizedCrop(size_x, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees = 10),
	transforms.Resize([size_x,size_y]),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
    transforms.ToTensor()])

dataloaders_dict = {'train': load_dataset('train',transform),
'val':load_dataset('test',transform)}

print('\n[Data loaded successfully]')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Details ------------------------------------------------------

num_epochs = 5

# Initialize the model for this run
model_ft = Livenet()

# summary of the model

# summary(model, *inputs, batch_size=-1, show_input=False, show_hierarchical=False,
#         print_summary=False, max_depth=1, show_parent_layers=False):

# print(summary(model_ft,torch.zeros((32,3,32,32)),show_input = False,show_hierarchical = True))
# Print the model we just instantiated
# print(model_ft)
# pdb.set_trace()

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)

params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr = 0.01, momentum = 0.5)

# optimizer_ft = optim

# Setup the loss function
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, plots = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs = num_epochs)
plt.plot(plots[0])
plt.plot(plots[1])
plt.legend(['training','validation'])
plt.savefig('history.png')
torch.save(model_ft.state_dict(), '/home/ojas/Desktop/itsp/project/models/livenet5.pth')
plt.show()
# pdb.set_trace()