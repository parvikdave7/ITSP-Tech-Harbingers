import os
import glob

jpg_names = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/real/adit/*.jpg")
# png_names = glob.glob("/home/ojas/Desktop/itsp/project/dataset/real/*.png")
# for i,names in enumerate(png_names):
# 	os.rename("{}".format(names),"{}.png".format(i))

for i,names in enumerate(jpg_names):
	os.rename("{}".format(names),"{}.jpg".format(i+2600))

# import os
# for i in range(164):
# 	os.rename("{}.png".format(i),"{}.png".format(i+2509))