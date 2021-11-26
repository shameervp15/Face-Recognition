from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


#command line arguments
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--dataset", required=True, help="path input of images consisting of faces")
argp.add_argument("-e", "--encoding", required=True, help="path to store facial encodings")
argp.add_argument("-d", "--detection-method", type=str, default="cnn", help="type of detection : cnn or hog")
args = vars(argp.parse_args())

#input image paths listing out
print("[INFO] quantifying faces...")
imgpaths = list(paths.list_images(args['dataset']))

#initializing the know encodings and known names
knownEncodings = []
knownNames = []

#looping over the impgpaths
for (i, imgpath) in enumerate(imgpaths):
    #getting the name from the path
    print(f"[INFO] processing image {i+1}/{len(imgpaths)}...")
    name = imgpath.split(os.path.sep)[-2]

    #loading and converting the input image to opencv order
    #to dlib ordering (RGB)
    image = cv2.imread(imgpath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #detecting the (x,y) coordinates of boxes of each faces
    boxes = face_recognition.face_locations(rgb, model=args["detection-method"])

    #computing facial embedding
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        #appending names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)


#storing the encodings and names into file
print("[INFO] Serializing encodings...")
data = {"encodings" : knownEncodings,
        "names"     : knownNames}
f = open(args["encoding"], "wb")
f.write(pickle.dumps(data))
f.close()


