# USAGE
# python neural_style_transfer.py --image images/baden_baden.jpg --model models/instance_norm/starry_night.t7

# import the necessary packages
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")


def style_image(image,style):
# load the neural style transfer model from disk
    # print("[INFO] loading style transfer model...")
    net = cv2.dnn.readNetFromTorch(style)

    # load the input image, resize it to have a width of 600 pixels, and
    # then grab the image dimensions
    
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    return output, end-start
# show information on how long inference took
# print("[INFO] neural style transfer took {:.4f} seconds".format(end - start))



if __name__=='__main__':
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    output, t=style_image(image,args["model"])
    # show the images
    print(f"[INFO] neural style transfer took {t:.4f} seconds")
    cv2.imshow("Input", image)
    cv2.imshow("Output", output)
    cv2.waitKey(0)