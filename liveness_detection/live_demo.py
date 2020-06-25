from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
from torchvision import models, transforms
import torch.nn as nn
import torch
import os
import pdb
from helper_functions import Livenet
from face_helper_funcs import set_parameter_requires_grad,  initialize_model

print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print('[INFO] face detector loaded successfully ')
print('[INFO] Loading liveness detector Model ... ')
live_model = Livenet()
live_model.load_state_dict(torch.load(
    'live_models/livenet21.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
live_model = live_model.to(device)
live_model.eval()
print('[INFO] liveness detector loaded successfully ')
print('[INFO] Loading face recognition model ... ')
face_model, recog_size = initialize_model(model_name = 'squeezenet', num_classes = 5,
    feature_extract = False, use_pretrained = False)
face_model.load_state_dict(torch.load(
    'face_models/squeezenet2.pth'))
face_model = face_model.to(device)
face_model.eval()
print('[INFO] face recognition model loaded successfully ')

print('[INFO] streaming ...')

live_trans = transforms.Compose([transforms.ToTensor()])
recog_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

live_size = 32
video = VideoStream(src = 0).start()
live_result = {0:'fake',1:'real'}
recog_result = {0:'Adit',1:'Nishant',2:'Ojas',3:'Parvik',4:'Unknown',5:' '}

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
            face_recog = cv2.resize(face, (recog_size, recog_size))
            face = cv2.resize(face, (live_size, live_size))
            face = live_trans(face)
            face = face.reshape(1,3,live_size,live_size)
            face = face.float().cuda()

            #liveness prediction
            # torch.max(face)
            outputs = live_model(face)
            val,live_preds = torch.max(outputs,1)

            face_preds = torch.tensor([5])
            if live_result[live_preds.item()] == 'real':
                face_recog = recog_trans(face_recog)
                face_recog = face_recog.reshape(1,3,recog_size,recog_size)
                # pdb.set_trace()
                face_recog = face_recog.float().cuda()
                # print(face_recog.shape)
                outputs = face_model(face_recog)
                _,face_preds = torch.max(outputs,1)

            # print(torch.max(face),torch.min(face))


            # draw the label and bounding box on the frame
            for pred in live_preds:
                cv2.putText(frame, 
                    str(live_result[pred.item()]) + str(' : {:.4f}'.format(val.item())) +'  '+ str(recog_result[face_preds.item()]),
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, 'Press (q) to quit', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,150), 1)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    # if the (q) key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()