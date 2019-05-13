import face_recognition as fr
import numpy as np
import cv2

# load database

known_face_names = np.load("db/name.npy")
known_face_encodings = np.load("db/encoding.npy")

cap = cv2.VideoCapture("video.mp4")
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
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding, 0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)
    out.write(frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
