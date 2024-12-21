import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace

faces = RetinaFace.detect_faces(img_path="example.jpg")
# example = faces.get("face_1")
# if example:
#     print(example.get("facial_area"))
# i = 10

cap = cv2.VideoCapture("video.mp4")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    faces = RetinaFace.detect_faces(frame)
    for face in faces.keys():
        faceGotten = faces.get(face)
        if faceGotten:
            facial_area = faceGotten.get("facial_area")
            if facial_area:
                x = facial_area[0]
                y = facial_area[1]
                w = facial_area[2]
                h = facial_area[3]
                frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    cv2.imshow("video", frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# for face in faces:
#     path = "ex" + str(i) + ".png"
#     plt.imshow(face)
#     plt.savefig(path)
#     plt.show()
#     i = i + 1
