import numpy as np
import cv2
import face_recognition as recog
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
from helper_functions3 import Livenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model,_ = initialize_model(model_name = 'resnet', num_classes = 2, feature_extract = False, use_pretrained = False)
print('[ Loading Model ... ]')
model = Livenet()
model.load_state_dict(torch.load('/home/ojas/Desktop/itsp/project/models/livenet9.pth'))
model = model.to(device)
model.eval()
print('[ Model loaded successfully ]')

# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
# criterion = nn.CrossEntropyLoss()

size_x = 32
size_y = 32
video = cv2.VideoCapture(0)
result = {0:'fake',1:'real'}

while True:

	check, frame = video.read()
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]
	# rgb_frame = frame[:, :, ::-1]

	face_locs = recog.face_locations(rgb_small_frame)
	# face_locs = recog.face_locations(rgb_frame)

	# Real or fake
	labels = []
	
	faces = np.array([[[]]])
	for i,face_loc in enumerate(face_locs):
		top,right,bottom,left = face_loc
		# print("face located at : top {},right {},bottom {},left {}".format(top,right,bottom,left))

		# face_img = frame[top:bottom, left:right]
		face_img = small_frame[top:bottom, left:right]
		# print(face_img.shape)
		face_img = cv2.resize(face_img,(size_x,size_y))

		if i == 0:
			faces = face_img
		else:
			faces = np.concatenate((faces,face_img))

	faces = torch.from_numpy(faces.reshape(len(face_locs),3,size_x,size_y))
	# print(faces.shape)

	# Predictions
	# print('faces shape: {}'.format(faces.shape))
	# print(faces)
	if len(faces):

		# optimizer.zero_grad()

		with torch.set_grad_enabled(False):

			# print('model : ',model)
			# print('input shape : {}'.format(faces.shape))

			faces = faces.float().cuda()/255
			# print(faces.shape)
			# print(torch.max(faces[0]), torch.min(faces[0]))
			# for p in model.parameters():
				# print(p.data)
			# exit(0)
			outputs = model(faces)
			
			print('fake : {:.4f}'.format(outputs[0][0].item()), 'real : {:.4f}'.format(outputs[0][1].item()))
			_,preds = torch.max(outputs,1)

		# print(preds)


		for (top,right,bottom,left),pred in zip(face_locs, preds):
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# Box around the face
			cv2.rectangle(frame,(left+2,top+2),(right+2,bottom+2),(200,0,0),2)
			# Area for the label, though unneeded
			cv2.rectangle(frame,(left+2,bottom + 28),(right+2,bottom+2),(200,0,0),2)
			# Label text
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame,result[pred.item()],(left + 12,bottom + 22), font, 0.6, (0,255,0), 1)

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


video.release()
cv2.destroyAllWindows()