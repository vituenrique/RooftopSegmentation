import cv2
import os
import random
import numpy as np
from sys import argv
import pandas as pd

def saveHypothesisAsCSV(hypothesis, imgName):
	df = pd.DataFrame(hypothesis)
	df.to_csv(output_path + "hipoteses/" + imgName + ".csv", sep=",", header=False, index=False)

input_img = argv[1]
output_path = argv[2]
sigma = float(argv[3])
k = int(argv[4])
min_size = int(argv[5])

image_name = os.path.basename(input_img)[1]

segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=k, min_size=min_size)
src = cv2.imread(input_img)
segment = segmentator.processImage(src)
seg_image = np.zeros(src.shape, np.uint8)
regions_image = np.zeros(src.shape, np.uint8)

hypothesis = {}
hypothesis['top'] = []
hypothesis['bottom'] = []
hypothesis['left'] = []
hypothesis['right'] = []

for i in range(np.max(segment)):
	y, x = np.where(segment == i)

	top, bottom, left, right = min(y), max(y), min(x), max(x)
	hypothesis['top'].append(top)
	hypothesis['bottom'].append(bottom)
	hypothesis['left'].append(left)
	hypothesis['right'].append(right)

	cv2.rectangle(src, (left, bottom), (right, top), (0, 255, 0), 1)
	
	color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

	for xi, yi in zip(x, y):
		regions_image[yi, xi] = color

if not os.path.isdir(output_path):
	os.mkdir(output_path)

if not os.path.isdir(output_path + "hipoteses/"):
	os.mkdir(output_path + "hipoteses/")

cv2.imwrite(output_path + image_name + "_regions.png", regions_image) 
cv2.imwrite(output_path + image_name + "_hypothesis.png", src)
saveHypothesisAsCSV(hypothesis, image_name)
