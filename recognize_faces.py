import face_recognition
import pickle
import argparse
import cv2

#commandline args
argp = argparse.ArgumentParser()
argp.add_argument("-e", "--encodings", required=True, help="path to serialized db to encodings")
argp.add_argument("-i", "--image", required=True, help="path to input image")
argp.add_argument("-d", "--detection-method", type=str, default="cnn", help="detection method: cnn or hog")
args = vars(argp.parse_args())

#LOADING THE EMBEDDING OR ENCODINGS
print("[INFO] Loading the encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

#load the input image and converting from bgr to rgb
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#detect the (x,y) coordinates of each input image faces
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection-method"])
encodings = face_recognition.face_encodings(rgb, boxes)

#initializing names for each face
names = []

for encoding in encodings:
    #comparing each faces with dataset encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    #matches returns either True or False
    if True in matches:
        #finding the indexes of all matched faces
        matchindexes = [i for (i,b) in enumerate(matches) if b]
        counts = {}

        #getting the names from dataset with the index
        for i in matchindexes:
            name = data["names"][i]
            #count dict adding names as keys and counts as value, by default=0
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)

    names.append(name)

#recognized faces visualizing
for ((top, right, bottom, left), name) in zip(boxes, names):
    #drawing the rectangle boxes
    cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)

    #text position
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)













