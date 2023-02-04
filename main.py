import cv2

cap = cv2.VideoCapture(0)

cap.set(3, 224)
cap.set(4, 224)
print(cap.get(3))
print(cap.get(4))
print(cap.get(5))
while(1):
    ret, frame = cap.read(0)

    cv2.imshow("capture", frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break