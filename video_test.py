import cv2
url='rtsp://192.168.16.38:554/user=admin&password=&channel=1&stream=0.sdp?'
cap=cv2.VideoCapture(url)

while True:
    ret,frame=cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()