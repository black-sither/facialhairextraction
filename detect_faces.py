import numpy as np
opencv.open
import argparse
import cv2
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

print("computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0,1):
	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		thface=copy.deepcopy(image[startY - int((0.8 * startY)) :endY + int((0.1 * endY)) ,startX - int((0.1 * startX)):endX + int((0.1 * endX))])
		#kan=cv2.Canny(thface,220,220)
		mini=cv2.Canny(thface,300,300)
		kan=cv2.Canny(thface,350,350)
		kan2=cv2.Canny(thface,200,200)
		kan4=kan2 - kan#cv2.bitwise_and(kan2,kan) 
		kan3=kan2 - kan4
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
gray=cv2.cvtColor(thface,cv2.COLOR_BGR2GRAY)
ret, mask1 = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
lower_red = np.array([0])
upper_red = np.array([100])
msk = cv2.inRange(gray, lower_red, upper_red)
res = cv2.bitwise_and(kan4,msk)  
cv2.imshow('face',thface)
#cv2.imshow('mini',mini)
#cv2.imshow('kan',kan)
#cv2.imshow('kan4',kan4)
#cv2.imshow('kan2',kan2)
#cv2.imshow('difference',kan4)
#cv2.imshow('mask',msk)
cv2.imshow('res',res)
#cv2.imshow('gray',gray)
res = cv2.bitwise_and(kan4,mask1)
#cv2.imshow('mask1',mask1)
cv2.imshow("Output", image)
cv2.waitKey(0)
