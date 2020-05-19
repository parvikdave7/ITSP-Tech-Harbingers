import face_recognition as recog
from PIL import Image

image = recog.load_image_file("people.jpg")
face_locs = recog.face_locations(image)

print("images found : {}".format(len(face_locs)))

i = 0

for face_loc in face_locs:

	top,right,bottom,left = face_loc
	print("face located at : top {},right {},bottom {},left {}".format(top,right,bottom,left))

	face_img = image[top:bottom, left:right]
	pil_img = Image.fromarray(face_img)
	pil_img.save("people-{}.jpg".format(i))

	i += 1