import os
import glob
import pdb

adit = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/adit/*.png")
adrian = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/adrian/*.png")
mummy = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/mummy/*.png")
nishant = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/nishant/*.png")
nishant3 = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/nishant3/*.png")
parvik = glob.glob("/home/ojas/Desktop/itsp/project/github/ITSP-Tech-Harbingers/full pipeline/dataset/fake/parvik/*.png")
# png_names = glob.glob("/home/ojas/Desktop/itsp/project/dataset/real/*.png")
# for i,names in enumerate(png_names):
# 	os.rename("{}".format(names),"{}.png".format(i))
pdb.set_trace()

t = 2611

for i,names in enumerate(adit):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1

for i,names in enumerate(adrian):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1

for i,names in enumerate(mummy):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1

for i,names in enumerate(nishant):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1

for i,names in enumerate(nishant3):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1

for i,names in enumerate(parvik):
	os.rename("{}".format(names),"{}.png".format(t))
	t += 1


# import os
# for i in range(164):
# 	os.rename("{}.png".format(i),"{}.png".format(i+2509))