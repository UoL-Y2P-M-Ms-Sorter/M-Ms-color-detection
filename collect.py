import serial.tools.list_ports
import cv2
from com import send
import time
import os
import json

with open('./class_indices.json', "r") as f:
    class_indict = json.load(f)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

device_exist = 0
port_list = list(serial.tools.list_ports.comports())




mode = 2
color = 0

img_index = [200, 200, 200, 200, 200, 200, 200]

while(1):
    if device_exist :
        send(serial, 0, 0)

    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

