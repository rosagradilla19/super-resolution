# Perform super resolution on images loaded from disk

import argparse
import cv2
import os
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
        help = "path to super resolution model")
ap.add_argument("-i", "--image", required=True,
        help="path to input image we want to increase resolution of")
ap.add_argument("-n", "--name", required=True,
        help="name for output image")

args = vars(ap.parse_args())

# extract model name and model scale from file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the
# super resolution model from disk, and set the model name and scale
print("Loading super resolution model: {}".format(args["model"]))
print("model name: {}".format(modelName))
print("model scale: {}".format(modelScale))

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

# load the input image from disk  
image = cv2.imread(args["image"])

# use the SR model to upscale the image, timing it
print("Starting super resolution...")
start = time.time()
upscaled = sr.upsample(image)
end = time.time()

# print time it took to upscale image
print("super resolution took: {:.6f}".format(end-start))
output_name = str(args["name"])

# Save the image
cv2.imwrite(f"outputs/{output_name}.png", upscaled)
print("All done")

#cv2.imshow("Original", image)
#cv2.imshow("Super Resolution", upscaled)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




