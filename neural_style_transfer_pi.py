# USAGE
# python neural_style_transfer_video.py --models models

# import the necessary packages
#from imutils.video import VideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
ap.add_argument("-w","--width",default=320,help="image width")
args = vars(ap.parse_args())

# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))
width=np.int(args['width'])
# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
camera=PiCamera()
camera.resolution=(320,240)
camera.framerate=32
raw=PiRGBArray(camera,size=(320,240))
time.sleep(0.10)
print("[INFO] {}. {}".format(modelID + 1, modelPath))

# loop over frames from the video file stream
#while True:
for raw_frame in camera.capture_continuous(raw,format='rgb',use_video_port=True):
	start=time.time()
	# grab the frame from the threaded video stream
	#camera.capture(raw,format='bgr')
	frame = raw_frame.array
	if frame is None:
		frame=np.random.randint(0,255,(240,320,3),dtype=np.uint8)
	frame=cv2.flip(frame, 1)
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=width)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	# construct a blob from the frame, set the input, and then perform a
	# forward pass of the network
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	#output /= 255.0
	output=np.clip(output,0,255)
	output = output.transpose(1, 2, 0)
	output=cv2.resize(output,(640,480))
	output=output.astype(np.uint8)
	# show the original frame along with the output neural style
	# transfer
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	raw.truncate(0)
	end=time.time()
	print("{} fps".format(1/(end-start)))
	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
	if key == ord("n"):
		# grab the next nueral style transfer model model and load it
		(modelID, modelPath) = next(modelIter)
		print("[INFO] {}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)

	# otheriwse, if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
