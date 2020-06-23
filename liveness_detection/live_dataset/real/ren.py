import os
import glob

jpg_names = glob.glob("/home/ojas/Desktop/itsp/project/dataset/real/*.jpg")
png_names = glob.glob("/home/ojas/Desktop/itsp/project/dataset/real/*.png")
for i,names in enumerate(png_names):
	os.rename("{}".format(names),"{}.png".format(i))

for i,names in enumerate(jpg_names):
	os.rename("{}".format(names),"{}.jpg".format(i+len(png_names)))