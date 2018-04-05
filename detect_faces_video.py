# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import copy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print('the value of args is: ' + str(args))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream('bman.mp4').start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0,1): #detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		#thface=frame[startX:startY,endX:endY]
		thface=copy.deepcopy(frame[startY - int((0.8 * startY)) :endY + int((0.1 * endY)) ,startX - int((0.1 * startX)):endX + int((0.1 * endX))])
		kan=cv2.Canny(thface,300,300)
		kan2=cv2.Canny(thface,50,50)
		kan4=kan2 - kan
		kan3=kan2 - kan4
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray=cv2.cvtColor(thface,cv2.COLOR_BGR2GRAY)
	ret, mask1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
	lower_red = np.array([0])
	upper_red = np.array([100])
	msk = cv2.inRange(gray, lower_red, upper_red)
	res = cv2.bitwise_and(kan4,mask1)    
	# show the output frame
	cv2.imshow("Frame", frame)
	try:
		cv2.imshow('face',thface)
	except Exception as e:
		print(str(e))
	#cv2.imshow('kan',kan)
	#cv2.imshow('kan2',kan2)
	#cv2.imshow('kanSub',kan4)
	#cv2.imshow('mask',mask1)
	cv2.imshow('res',res)
	cv2.imshow('gray',gray)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()