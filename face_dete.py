import cv2
import dlib
from scipy.spatial import distance

url=0
cap=cv2.VideoCapture(url)
face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
# eyes_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_eye.xml')
while True:
    ret,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # eyes = eyes_cascade.detectMultiScale(img,
    #                                       scaleFactor=1.1,
    #                                       minNeighbors=5,
    #                                       minSize=(30, 30),
    #                                       flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 255, 0), 1)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]

    # for (x, y, w, h) in eyes:
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     # cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 255, 0), 1)
    #     roi_gray = gray[y: y + h, x: x + w]
    #     roi_color = img[y: y + h, x: x + w]

    cv2.imshow("frame",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()