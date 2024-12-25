import cv2
from deepface import DeepFace
from retinaface import RetinaFace

cap = cv2.VideoCapture("video.mp4")
#
# cap = cv2.VideoCapture("http://192.168.100.164:4747/video")
count = 0


def extract_faces(img_path):
    return DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="opencv",
        enforce_detection=False,
    )


def face_recognition(img_path):
    return DeepFace.find(
        img_path=img_path,
        db_path="./database/",
        model_name="Facenet",
        detector_backend="opencv",
        enforce_detection=False,
        silent=True,
    )


def draw_bounding_box(frame, x, y, w, h):
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    results = face_recognition(frame)

    for result in results:
        x = 0
        y = 0
        w = 0
        h = 0
        if len(result["source_x"]) is not 0:
            x = result["source_x"].iloc[0]
        if len(result["source_y"]) is not 0:
            y = result["source_y"].iloc[0]
        if len(result["source_w"]) is not 0:
            w = result["source_w"].iloc[0]
        if len(result["source_h"]) is not 0:
            h = result["source_h"].iloc[0]
        if x == 0 and y == 0 and w == 0 and h == 0:
            results_face = extract_faces(frame)
            for result in results_face:
                facial_area = result.get("facial_area")
                if facial_area:
                    x = facial_area.get("x")
                    y = facial_area.get("y")
                    w = facial_area.get("w")
                    h = facial_area.get("h")
                    draw_bounding_box(frame, x, y, w, h)
        else:
            draw_bounding_box(frame, x, y, w, h)
            if len(result["identity"]) is not 0:
                identity = result["identity"].iloc[0]
            else:
                break
            identity_array = identity.split("/")
            cv2.putText(
                frame,
                identity_array[2],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # for face in faces:
    #     face_img = face.get("face")
    #     facial_area = face.get("facial_area")
    #     if facial_area:
    #         x = facial_area.get("x")
    #         y = facial_area.get("y")
    #         w = facial_area.get("w")
    #         h = facial_area.get("h")
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     if face_img is not None:
    #         results = DeepFace.find(
    #             img_path=frame,
    #             db_path="./database/",
    #             model_name="Facenet",
    #             detector_backend="retinaface",
    #             enforce_detection=False,
    #             silent=True,
    #         )
    #         for result in results:
    #             print(result.keys())
    #             identity = result.get("identity")
    #             if identity is not None:
    #                 print(identity)

    # for face_bounding_box in faces_bounding_boxes.keys():
    #     faceGotten = faces_bounding_boxes.get(face_bounding_box)
    #     if faceGotten:
    #         facial_area = faceGotten.get("facial_area")
    #         if facial_area:
    #             x = facial_area[0]
    #             y = facial_area[1]
    #             w = facial_area[2]
    #             h = facial_area[3]
    #             frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow("video", frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
