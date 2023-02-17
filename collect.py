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


port_list = list(serial.tools.list_ports.comports())

port_name = "STM32"

for i in range(0, len(port_list)):
    if port_name in port_list[i].description:
        serial = serial.Serial(port=port_list[i].device, write_timeout=2)

mode = 2
color = 6

img_index = [0, 0, 0, 0, 0, 0, 0]

while(1):
    '''send(serial, mode, color)'''
    time.sleep(0.5)

    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    key = cv2.waitKey(0) & 0xFF
    color = key - 48
    cv2.imwrite(os.path.join('data/train', class_indict[str(color)], str(img_index[color]+1) + '.png'), frame)
    img_index[color] += 1

    if key == ord('q'):
        break

if serial.isOpen():
    print('串口已打开')
else:
    print('串口未打开')

data = (1).to_bytes(1, byteorder= "big")
serial.write(data)

while True:
    data = serial.read(20)    #串口读20位数据
    if data != b'':
        break
print ('receive data is :',data)

# 关闭串口
serial.close()

if serial.isOpen():
    print('串口未关闭')
else:
    print('串口已关闭')