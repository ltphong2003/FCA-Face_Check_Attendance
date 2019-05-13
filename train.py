import face_recognition as fr
import cv2
import glob
import numpy as np

# training
data_path = "data/"
training_image_files = glob.glob(data_path + "*.jpg")

known_face_encodings = []
known_face_names = []

for file in training_image_files:
    image = fr.load_image_file(file)
    frame = image
    face_locations = fr.face_locations(image)
    if len(face_locations) < 1:
        cv2.putText(frame, "Faces not found!", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    else:
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        known_face_encodings.append(fr.face_encodings(image, face_locations)[0])
        name = file[5:-4]
        known_face_names.append(name)
    cv2.imshow("Face Recognition - " + name, frame[:, :, ::-1])
    cv2.waitKey(0)

cv2.destroyAllWindows()
np.save("db/name", known_face_names)
np.save("db/encoding", known_face_encodings)
