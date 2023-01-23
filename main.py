import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while True:
    abcd, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame, 1.2, 3)
    for x, y, w, h in face:
        ROI = frame[y:y+h, x:x+w]
        blur = cv2.medianBlur(ROI, 35)
        frame[y:y + h, x:x + w] = blur
        #img = cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 0, 255),3)
        #img[y:y+h, x:x+w] = cv2.medianBlur(img[y:y+h, x:x+w], 35)
    cv2.imshow('Face Blur', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
