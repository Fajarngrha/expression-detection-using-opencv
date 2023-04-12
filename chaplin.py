from chardet import detect
import cv2
import numpy as np
import pandas as pd
import requests
import os
import datetime
import pytz
import schedule
import time

yolo = cv2.dnn.readNet("yolov3-smile_final2.weights", "yolov3-smile2.cfg")
classes = []



with open("classes3.txt", "r") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed = (0,0,255)
colorGreen = (0,255,0)

# #Loading Images
# name = "image.jpg"
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://kamera:1234as@10.16.89.34/stream1')
person = 1
starting_time= time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
smile = 0
flat = 0
frown = 0

start = time.time()

def conect():
    try :
        r = requests.get(url,timeout=5)
        return True
    except (requests.ConnectionError, requests.Timeout) as exception:
        return False

while True :

    _,imgS= cap.read() # 
    #imgS = cv2.resize(imgS, (920, 540))
    imgS = cv2.flip(imgS, 1)
    height, width, channels = imgS.shape

    now = datetime.datetime.now()
    localtz = pytz.timezone("Asia/Jakarta")
    date = now.astimezone(localtz).strftime("%Y-%m-%d")
    waktos = now.astimezone(localtz).strftime("%H:%M")
    
    # # Detecting objects
    blob = cv2.dnn.blobFromImage(imgS, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id== 0 and  confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            if class_id== 1 and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            if class_id== 2 and confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            unique, counts = np.unique(class_ids, return_counts=True)
            tambah=0
            cv2.rectangle(imgS, (x, y), (x + w, y + h), colorGreen, 3)
            cv2.putText(imgS, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 5)
            #person+=1
            if label == 'Smile':
                smile += 1
                for i in range (len(counts)):
                    cv2.putText(imgS,label+" = "+str(counts[i]), (2,15+tambah),cv2.FONT_HERSHEY_PLAIN,1, (0,200,0), 2)
                    tambah=tambah +15
                

               
            if label == 'Netral':
                flat += 1
                for i in range (len(counts)):
                    cv2.putText(imgS,label+" = "+str(counts[i]), (2,15+tambah),cv2.FONT_HERSHEY_PLAIN,1, (0,200,0), 2)
                    tambah=tambah +15
              
            if label == 'Frown':
                frown += 1
                for i in range (len(counts)):
                    cv2.putText(imgS,label+" = "+str(counts[i]), (2,15+tambah),cv2.FONT_HERSHEY_PLAIN,1, (0,200,0), 2)
                    tambah=tambah +15
               

      
    cv2.putText(imgS, f'Smile : {smile-1}', (40,70), cv2.FONT_HERSHEY_PLAIN, 1, (50,255,), 2)
    cv2.putText(imgS, f'Netral : {flat-1}', (40,90), cv2.FONT_HERSHEY_PLAIN, 1, (50,255,), 2)
    cv2.putText(imgS, f'Frown : {frown-1}', (40,110), cv2.FONT_HERSHEY_PLAIN, 1, (50,255,), 2)

    cv2.imshow("Image", imgS)
   

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

# schedule.every(5).seconds.do(detect)
# while 1:
#     schedule.run_pending()
#     time.sleep(1)

cap.release()
cv2.destroyAllWindows()