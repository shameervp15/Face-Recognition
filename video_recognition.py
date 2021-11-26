import cv2
import face_recognition
import imutils
from imutils.video import VideoStream
import argparse
import pickle
import time

#commandline args
argp = argparse.ArgumentParser()
argp.add_argument("-e", "--encodings", required=True, help="path serialized facial encodings")
argp.add_argument("-o", "--output", type=str, help="path to output video")
argp.add_argument("-y", "--display", type=int, default=1, help="whether to display video")
argp.add_argument("-d", "--detection-method", type=str, default="cnn", help="detection method to use: hog or cnn")
args = vars(argp.parse_args())

#loading encodings
print("[INFO] Loading the ecnodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
#src=0 primary camera. src=1 secondary
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

#looping over each frames of video
while True:
    frame = vs.read()

    #converting BGR TO RGB, resizing to 750px to speedup processing
    rgbcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgbcolor, width=750)
    r = frame.shape[1]/float(rgb.shape[1])

    # detect the (x,y) coordinates of each input image faces
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=args["detection-method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchindexes = [i for (i,b) in enumerate(matches) if b]
            counts = {}

            for i in matchindexes:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom,left), name) in zip(boxes, names):
        #rescaling the face coordinates. r being the framewidth
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        #WRITING TO VIDEO STREAM
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[0], frame.shape[1]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)

        #display arg passed
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


cv2.destroyAllWindows()
vs.stop()

#releasing display frame
if writer is not None:
    writer.release()



