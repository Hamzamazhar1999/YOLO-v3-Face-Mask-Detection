import cv2
import numpy as np

net = cv2.dnn.readNet('D:\FYP Stuff\Python Programs\YOLO\scripts\yolov3_training_final.weights', 'D:\FYP Stuff\Python Programs\YOLO\scripts\yolov3_testing.cfg')
classes = []
with open('D:\FYP Stuff\Python Programs\YOLO\scripts\obj.names', 'r') as f:
    classes = f.read().splitlines()

print(classes)

img = cv2.imread('D:\FYP Stuff\Python Programs\YOLO\scripts\Pictures\yolomolo.jpg')
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layersOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layersOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

#print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes.flatten())

font = cv2.FONT_HERSHEY_DUPLEX

colors = np.random.uniform(0, 255, size=(len(boxes), 3))
if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i]*100, 2))
        color = colors[i]
        if label=="Mask":
            cv2.rectangle(img, (x,y), (x+w, y+h),(0, 255, 0) , 2)
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255) , 2)
        text = f"{label} {confidence}%"
        if label=="Mask":
            cv2.putText(img, text, (x-10, y+130), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, text, (x-10, y+130), font, 0.5, (0,0,255) , 2)


        #for b in blob:
           #for n, img_blob in enumerate(b):
               #cv2.imshow(str(n), img_blob)

    cv2.imshow('Image', img)
    key=cv2.waitKey()
    cv2.destroyAllWindows()