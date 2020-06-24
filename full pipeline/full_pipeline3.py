# USAGE
# python full_pipeline3.py --encodings encodings.pickle  --recognizer output/recognizer.pickle --le output/le.pickle 
# python full_pipeline3.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from tensorflow.keras.models import load_model
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
import glob
import tensorflow as tf
from helper_functions import Livenet
from torchvision import models, transforms
import torch.nn as nn
import torch

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder for recognition")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print('[INFO] Loading liveness detector Model ... ')
live_model = Livenet()
live_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
live_model.load_state_dict(torch.load(
    'live_models/livenet20.pth',map_location=device))
live_model = live_model.to(device)
live_model.eval()
live_trans = transforms.Compose([transforms.ToTensor()])
recog_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

live_result = {0:'fake',1:'real'}
print('[INFO] liveness detector loaded successfully ')

print("[INFO] loading face recognizer...")
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    probs = []
    # loop over the facial embeddings
    for ((top, right, bottom, left), encoding) in zip(boxes, encodings):
        
        face = frame[top:bottom, left:right]

        face = cv2.resize(face, (live_size, live_size))
        face = live_trans(face)
        face = face.reshape(1,3,live_size,live_size)
        face = face.float()
        #liveness prediction
        outputs = live_model(face)
        val,live_preds = torch.max(outputs,1)

        # draw the label and bounding box on the frame
        if live_result[live_preds.item()] == 'fake':
            name="Fake"
            proba=val.item()
            names.append(name)
            probs.append(proba)
        else:
        # perform classification to recognize the face
            preds = recognizer.predict_proba(encoding.reshape(1,-1))[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            names.append(name)
            probs.append(proba)

    # loop over the recognized faces
    for ((top, right, bottom, left), name, proba) in zip(boxes, names, probs):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, "{}: {:.2f}%".format(name, proba*100), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
    

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# stop the timer and display FPS information

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
