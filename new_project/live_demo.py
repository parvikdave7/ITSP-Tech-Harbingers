import numpy as np
import cv2
import face_recognition as recog
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
from helper_functions import initialize_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model,_ = initialize_model(model_name = 'resnet', num_classes = 2, feature_extract = False, use_pretrained = False)
model.load_state_dict(torch.load('/home/ojas/Desktop/itsp/project/models/resnet8.pth'))
model = model.to(device)
model.eval()
print('[ Model loaded successfully ]')

size_x = 150
size_y = 220
video = cv2.VideoCapture(0)
result = {0:'fake',1:'real'}

while True:

	check, frame = video.read()
	small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
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
		# pil_img = Image.fromarray(face_img)

		if i == 0:
			faces = face_img
		else:
			faces = np.concatenate((faces,face_img))

	faces = torch.from_numpy(faces.reshape(len(face_locs),3,size_x,size_y))
	# print(faces.shape)

	# Predictions
	if len(faces):
		faces = faces.float().cuda()
		outputs = model(faces)
		_,preds = torch.max(outputs,1)


		for (top,right,bottom,left),pred in zip(face_locs, preds):
			top *= 2
			right *= 2
			bottom *= 2
			left *= 2

			# Box around the face
			cv2.rectangle(frame,(left,top),(right,bottom),(200,0,0),1)
			# Area for the label, though unneeded
			cv2.rectangle(frame,(left,bottom + 26),(right,bottom),(200,0,0),1)
			# Label text
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame,result[pred.item()],(left + 12,bottom + 20), font, 0.6, (0,255,0), 1)

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


video.release()
cv2.destroyAllWindows()