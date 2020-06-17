from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
from torchvision import models, transforms
import torch.nn as nn
import torch
from helper_functions3 import Livenet
import os


print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print('[ Loading Model ... ]')
model = Livenet()
model.load_state_dict(torch.load
    ('/home/ojas/Desktop/itsp/project/models/livenet10.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print('[ Model loaded successfully ]')

trans = transforms.Compose([transforms.ToTensor()])

size = 32
video = VideoStream(src = 0).start()
result = {0:'fake',1:'real'}

while True:

    frame = video.read()
    frame = imutils.resize(frame, width=600)
    #Convert to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (size, size))
            face = trans(face)
            face = face.reshape(1,3,size,size)
            face = face.float().cuda()
            # print(torch.max(face),torch.min(face))

            torch.max(face)
            outputs = model(face)
            val,preds = torch.max(outputs,1)

            # draw the label and bounding box on the frame
            for pred in preds:
                cv2.putText(frame, 
                    str(result[pred.item()]) + str(' : {:.4f}'.format(val.item())),
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the (q) key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()
