from PIL import Image
import numpy as np
from torchvision import models
import cv2
import pdb
import glob
import face_recognition as recog

nishant11 = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/liveness_detection/live_dataset/fake/nishant11/*.png")

def decrease_brightness(img, fracn = 1/2):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	
	v = (v*fracn).astype(np.int)
	# v[v < value] = 
	# v[v >= value] -= value
	pdb.set_trace()

	final_hsv = cv2.merge((h, s, v))
	pdb.set_trace()
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

for i,name in enumerate(nishant11):
	image = recog.load_image_file(name)
	image = decrease_brightness(image)
	print(i)
	pil_img = Image.fromarray(image)
	pil_img.save("{}.png".format(300+i))



# i = 0

# labels = []
# face = np.array([[[]]])
# for face_loc in face_locs:

# 	top,right,bottom,left = face_loc
# 	print("face located at : top {},right {},bottom {},left {}".format(
# 		top,right,bottom,left))

# 	face_img = image[top:bottom, left:right]
# 	face_img = cv2.resize(face_img,(size,size))
# 	face = torch.from_numpy(face_img.reshape(1,3,size,size))
# 	face = face.float().cuda()
# 	print(model(face))
# 	_,pred = torch.max(model(face),1)
# 	print(pred)

# 	i += 1

