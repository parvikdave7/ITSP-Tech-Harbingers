import face_recognition as recog
from PIL import Image
import numpy as np
import cv2

size_x = 150
size_y = 220
video = cv2.VideoCapture(0)
a = 0

while True:

	check, frame = video.read()
	rgb_frame = frame[:, :, ::-1]
	
	pil_img = Image.fromarray(rgb_frame)
	pil_img.save("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/face_dataset_ojas/{}.jpg".format(0 + a))
	a += 1

	# face_locs = recog.face_locations(rgb_frame)
	
	# for i,face_loc in enumerate(face_locs):
		
	# 	a += 1
	# 	top,right,bottom,left = face_loc
	# 	face_img = rgb_frame[top:bottom, left:right]
	# 	face_img = cv2.resize(face_img,(size_x,size_y))

	# 	pil_img = Image.fromarray(face_img)
		# pil_img.save("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/face_dataset_ojas/{}.jpg".format(0 + a))

	# for top,right,bottom,left in face_locs:

	# 	cv2.rectangle(frame,(left,top),(right,bottom),(200,0,0),1)


	cv2.imshow('Video', frame)
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break

video.release()
cv2.destroyAllWindows()