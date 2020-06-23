# USAGE
# python static_pipeline.py --encodings encodings.pickle --model liveness.model --recognizer output/recognizer.pickle --le output/le.pickle --le_liveness le_liveness.pickle --display 0
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0 --le-liveness le_liveness.pickle

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
import face_recognition
import argparse
import imutils
from PIL import Image
import pickle
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
import glob

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-m", "--model", type=str, default="liveness.model",
    help="path to trained model")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder for recognition")
ap.add_argument("-l2", "--le_liveness", default="le_liveness.pickle",
    help="path to label encoder for liveness")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le_liveness=pickle.loads(open(args["le_liveness"], "rb").read())

print("[INFO] loading face recognizer...")
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
###vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

img_dir = "trial" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)



# loop over frames from the video file stream
###while True:
a=0
for datum in data:
    # grab the frame from the threaded video stream
    ###frame = vs.read()
    frame=datum
    
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    probs = []
    i=0
    # loop over the facial embeddings
    for ((top, right, bottom, left), encoding) in zip(boxes, encodings):
        
        face_img = rgb[top:bottom, left:right]
        face_img = cv2.resize(face_img, (32, 32))
        face_img = face_img.astype("float") / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        preds = model.predict(face_img)[0]
        j = np.argmax(preds)
        label = le_liveness.classes_[j]

        # draw the label and bounding box on the frame
        if label == "fake":
            name="Fake"
            proba=preds[j]
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
    
    true_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(true_frame)
    pil_img.save("results\{}.jpg".format(0 + a))
    a=a+1

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces todisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
