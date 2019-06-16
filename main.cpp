#include <opencv2/opencv.hpp>
#include <fstream>
#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include "GraphSegmentation/lib/graph_segmentation.h"

using namespace std;
using namespace cv;
using namespace experimental::filesystem;


vector<string> getAllImagesInDirectory(string path) {
	vector<string> imagesPath;
	cout << path << endl;
	for (const auto & entry : directory_iterator(path)) {
		imagesPath.push_back(entry.path().string());
	}
	return imagesPath;
}

/** \brief Check if the given pixel is a boundary pixel in the given
* segmentation.
* \param[in] labels segments as integer image
* \param[in] i y coordinate
* \param[in] j x coordinate
* \return true if boundary pixel, false otherwise
*/
bool is4ConnectedBoundaryPixel(const Mat &labels, int i, int j) {

	if (i > 0) {
		if (labels.at<int>(i, j) != labels.at<int>(i - 1, j)) {
			return true;
		}
	}

	if (i < labels.rows - 1) {
		if (labels.at<int>(i, j) != labels.at<int>(i + 1, j)) {
			return true;
		}
	}

	if (j > 0) {
		if (labels.at<int>(i, j) != labels.at<int>(i, j - 1)) {
			return true;
		}
	}

	if (j < labels.cols - 1) {
		if (labels.at<int>(i, j) != labels.at<int>(i, j + 1)) {
			return true;
		}
	}

	return false;
}

/** \brief Draw the segments as contours in the image.
* \param[in] image image to draw contours in (color image expected)
* \param[in] labels segments to draw as integer image
* \param[out] contours image with segments indicated by contours
*/
void drawContours(const Mat &image, const Mat &labels, Mat &contours) {

	assert(!image.empty());
	assert(image.channels() == 3);
	assert(image.rows == labels.rows && image.cols == labels.cols);
	assert(labels.type() == CV_32SC1);

	contours.create(image.rows, image.cols, CV_8UC3);
	Vec3b color(0, 0, 0); // Black contours

	for (int i = 0; i < contours.rows; ++i) {
		for (int j = 0; j < contours.cols; ++j) {
			if (is4ConnectedBoundaryPixel(labels, i, j)) {

				contours.at<Vec3b>(i, j) = color;
			} else {
				contours.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
			}
		}
	}
}

void applyHoughTranform(Mat image, string filename, string output_dir) {

	Mat src_gauss, finalImage = image.clone();

	GaussianBlur(image, src_gauss, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Mat dst, cdst;
	Canny(src_gauss, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 30, 20, 5);
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(finalImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}

	path linepath(output_dir / path(path(filename).stem().string() + ".png"));
	imwrite(linepath.string(), finalImage);
}

void applyGraphSegmentation(Mat image, string filename, string output_dir, float threshold, int minimum_segment_size) {

	Mat src_gauss, finalImage = image.clone();
	GaussianBlur(image, src_gauss, Size(3, 3), 0, 0, BORDER_DEFAULT);

	GraphSegmentationMagicThreshold magic(threshold);
	GraphSegmentationEuclideanRGB distance;

	GraphSegmentation segmenter;
	segmenter.setMagic(&magic);
	segmenter.setDistance(&distance);

	segmenter.buildGraph(src_gauss);
	segmenter.oversegmentGraph();
	segmenter.enforceMinimumSegmentSize(minimum_segment_size);

	Mat labels = segmenter.deriveLabels();

	path contours_file(output_dir / path(path(filename).stem().string() + ".png"));

	drawContours(src_gauss, labels, finalImage);
	imwrite(contours_file.string(), finalImage);

}

int main() {

	string africanos_path = "dataset_satelite/africanos/";
	string nhozinho_path = "dataset_satelite/nhozinho/";
	vector<string> imagesPaths = getAllImagesInDirectory(africanos_path);
	for (int i = 0; i < imagesPaths.size(); i++) {
		string imagePath = imagesPaths.at(i);

		Mat src = imread(imagePath);

		if (!src.data) return -1; 

		applyHoughTranform(src, imagePath, "output_lines_2/");

		//applyGraphSegmentation(src, imagePath, "output_150_210/", 150, 210);
	}



	return 0;
}