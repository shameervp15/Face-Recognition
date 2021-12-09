import pickle
import face_recognition
import argparse
import cv2
import sqlite3


# commandline args
argp = argparse.ArgumentParser()
argp.add_argument("-n", "--name", required=True, help="Input the name to view in db")
args = vars(argp.parse_args())

name = args["name"]

# database
connection = sqlite3.connect("namesandencodings.db")
cursor = connection.cursor()
print("[INFO] Connected to database...")

# capturing the image from webcam
print("[INFO] Capturing image...")
videocapture = cv2.VideoCapture(0)
ret, frame = videocapture.read()

cv2.imshow("image", frame)
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# boxing and getting x,y coordinates
box = face_recognition.face_locations(image, model="hog")



results = []
# checking if multiple or no faces detected
if len(box) == 1:
    try:
        print("[INFO] Detecting face...")
        # checking if the table exists and inserting values
        cursor.execute('CREATE TABLE IF NOT EXISTS Encodings (Id INTEGER PRIMARY KEY, Name VARCHAR(20) NOT NULL, Encodings NOT NULL)')
        connection.commit()

        encoding_raw = face_recognition.face_encodings(image, box)[0]
        encoding = pickle.dumps(encoding_raw)

        # if the encodings does not exist
        try:
            cursor.execute("SELECT Encodings FROM Encodings;")
            if len(cursor.fetchall()) == 0:
                cursor.execute("INSERT INTO Encodings (Name, Encodings) VALUES(?, ?)", (name, encoding,))
                connection.commit()
                print("Saved to database!")
            else:
                # checking if the same face encodings exist
                for db_encodings in cursor.execute("SELECT Encodings FROM Encodings;"):
                    data = pickle.loads(data=db_encodings[0])
                    match = face_recognition.compare_faces([data], encoding_raw)
                    results.append(match)

                if [True] in results:
                    print('The face_id has already been stored!')
                else:
                    cursor.execute("INSERT INTO Encodings (Name, Encodings) VALUES(?, ?)", (name, encoding,))
                    connection.commit()
                    print("Saved to database!")


        except:
            print("Did not save!")

        connection.close()
        videocapture.release()
        cv2.waitKey()
        cv2.destroyAllWindows()

    except:
        print('Exception detected!')

else:
    print("Multiple faces or No face detected")





