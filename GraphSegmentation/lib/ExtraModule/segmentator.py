import cv2
import os
import random
import numpy as np
from sys import argv
import pandas as pd

def saveHypothesisAsCSV(hypothesis, imgName, output_path):
	df = pd.DataFrame(hypothesis)
	df.to_csv(output_path + "hipoteses/" + imgName + ".csv", sep=",", header=False, index=False)

def computeArea(img):
	_, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	M = cv2.moments(cnt)
	area = M['m00']
	return area

def main():

	input_img = argv[1]
	output_path = argv[2]
	sigma = float(argv[3])
	k = int(argv[4])
	min_size = int(argv[5])

	image_name = os.path.basename(input_img)[1]
	hypothesis = {}
	hypothesis['top'] = []
	hypothesis['bottom'] = []
	hypothesis['left'] = []
	hypothesis['right'] = []
	hypothesis['area'] = []

	for i in range(8):
		#print(k)
		segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=k, min_size=min_size)
		
		src = cv2.imread(input_img)

		blur = cv2.GaussianBlur(src, (5, 5), 0)
		smooth = cv2.addWeighted(blur, 1.5, src, -0.5, 0)

		segment = segmentator.processImage(smooth)
		seg_image = np.zeros(src.shape, np.uint8)
		regions_image = np.zeros(src.shape, np.uint8)

		for i in range(np.max(segment)):
			y, x = np.where(segment == i)

			top, bottom, left, right = min(y), max(y), min(x), max(x)
			hypothesis['top'].append(top)
			hypothesis['bottom'].append(bottom)
			hypothesis['left'].append(left)
			hypothesis['right'].append(right)

			cv2.rectangle(src, (left, bottom), (right, top), (0, 255, 0), 1)
			
			color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
			
			each_segment = np.zeros((src.shape[0], src.shape[1], 1), np.uint8)

			for xi, yi in zip(x, y):
				regions_image[yi, xi] = color
				each_segment[yi, xi] = 255
			
			# cv2.imshow("segments", each_segment)
			# cv2.waitKey(0)
			area_segment = computeArea(each_segment)
			hypothesis['area'].append(int(area_segment))

		if not os.path.isdir(output_path):
			os.mkdir(output_path)

		if not os.path.isdir(output_path + "hipoteses/"):
			os.mkdir(output_path + "hipoteses/")

		# cv2.imwrite(output_path + image_name + "_" + i + "_regions.png", regions_image) 
		# cv2.imwrite(output_path + image_name + "_" + i + "_hypothesis.png", src)

		k += 200
		#print(hypothesis)

	saveHypothesisAsCSV(hypothesis, image_name, output_path)

if __name__ == "__main__":
	main()