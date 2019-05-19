import math
from sklearn import neighbors
import cv2
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import time 
import datetime
import csv
import api as face_recognition 
import re
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]
        
def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.375):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def update(name):
    now = datetime.datetime.now()
    today8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
    today7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
    today5pm = now.replace(hour=17, minute=0, second=0, microsecond=0)
    today4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)
    localtime = time.localtime(time.time()) 
    file_name="Attendance/"+str(localtime[2])+"_"+str(localtime[1])+"_"+str(localtime[0]) +'.csv'
    a=1
    while a==1:
        try:
            if ((now<today8am)and(now>today7am)):
                kq=True
                with open(file_name, mode='r') as fileread:
                    att_reader = csv.reader(fileread, delimiter=',')
                    for row in att_reader:
                        if ((row[0]==name)and(row[1]=="Check in")):
                            kq=False
                    fileread.close()
                with open(file_name, mode='w') as localtime_file:
                    att_writer = csv.writer(localtime_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    att_writer.writerow([name,"Check in", time.asctime(localtime)])
                    localtime_file.close()
            if ((now<today5pm)and(now>today4pm)):
                kq=True
                with open(file_name, mode='r') as fileread:
                    att_reader = csv.reader(fileread, delimiter=',')
                    for row in att_reader:
                        if ((row[0]==name)and(row[1]=="Check out")):
                            kq=False 
                    fileread.close()
                with open(file_name, mode='w') as localtime_file:
                    att_writer = csv.writer(localtime_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    att_writer.writerow([name,"Check out", time.asctime(localtime)])
                    localtime_file.close()
            a=2
        except IOError:
            f= open(file_name,"w+")
            f.close()

    return name

if __name__ == "__main__":
    # STEP 2: Using the trained classifier, make predictions for unknown images
        # Find all people in the image using a trained classifier model
        # Note: Pass in either a classifier file name or a classifier model instance
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if (not cap.isOpened()):
        print('Default camera not found!')
        exit(0)
    else:
        width = (int)(cap.get(3))
        height = (int)(cap.get(4))
        fps = (int)(cap.get(5))

    out = cv2.VideoWriter('output.mp4',fourcc, fps, (width,height))

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while 1:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 180)
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_small_frame, model_path="trained_knn_model.clf")
        font = cv2.FONT_HERSHEY_DUPLEX
        localtime = time.asctime( time.localtime(time.time()) )
        cv2.putText(frame, localtime, (190, 40), font, 1.0, (255, 255, 255), 1)
        for name,(top, right, bottom, left) in predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            if (name!="Unknown"):
                process=update(name)
            fontScale = (29 * abs(left-right)) / (1000*8)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, fontScale, (255, 255, 255), 1)

        cv2.imshow('Face Check Attendance', frame)
        out.write(frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
        