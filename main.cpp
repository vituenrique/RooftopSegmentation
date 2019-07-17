#include <opencv2/opencv.hpp>
#include <fstream>
#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <experimental/filesystem>
#include <climits>
#include <list>
#include "GraphSegmentation/lib/graph_segmentation.h"

using namespace std;
using namespace cv;
using namespace experimental::filesystem;

// Gets all files path that are in a given dirictory
vector<string> getAllImagesInDirectory(string path) {
	vector<string> imagesPath;
	cout << path << endl;
	for (const auto & entry : directory_iterator(path)) {
		imagesPath.push_back(entry.path().string());
	}
	return imagesPath;
}

// Normalizes a vector of doubles 
vector<double> normalizeData(vector<int> data) {
	int max = INT_MIN;
	int min = INT_MAX;
	for (int i = 0; i < data.size(); i++) {
		if (data[i] >= max) max = data[i];
		if (data[i] < min) min = data[i];
	}

	vector<double> normalizedVector;
	for (int i = 0; i < data.size(); i++) {
		double normalized_value = (double)(data[i] - min) / (max - min);
		normalizedVector.push_back(normalized_value);
	}

	return normalizedVector;
}

// Normalizes a vector of doubles 
vector<double> normalizeData(vector<double> data) {
	double max = INT_MIN;
	double min = INT_MAX;
	for (int i = 0; i < data.size(); i++) {
		if (data[i] >= max) max = data[i];
		if (data[i] < min) min = data[i];
	}

	vector<double> normalizedVector;
	for (int i = 0; i < data.size(); i++) {
		double normalized_value = (double)(data[i] - min) / (max - min);
		normalizedVector.push_back(normalized_value);
	}

	return normalizedVector;
}

// Computes area
int computeArea(int w, int h) {
	return w * h;
}

// Computes area for each hypothesis
vector<int> computeAllHypothesisCovagere(vector<vector<int>> hypothesis) {
	vector<int> coverageHypothesis;

	for (int i = 0; i < hypothesis[0].size(); i++) {
		int height_hypothesis = abs(hypothesis[0][i] - hypothesis[1][i]);
		int width_hypothesis = abs(hypothesis[2][i] - hypothesis[3][i]);

		int coverage = computeArea(width_hypothesis, height_hypothesis);
		coverageHypothesis.push_back(coverage);
	}

	return coverageHypothesis;
}

// Computes Local Refinement of all hypothesis
vector<int> localHypothesisRefinement(vector<int> area_hypothesis, vector<int> area_segments, double threshold1 = 0.5, double threshold2 = 0.5) {
	vector<double> ratios;

	for (int i = 0; i < area_segments.size(); i++) {
		double ratio = (double)area_segments[i] / area_hypothesis[i];
		ratios.push_back(ratio);
	}

	vector<double> ratios_normalized = normalizeData(ratios);
	vector<double> area_hypothesis_normalized = normalizeData(area_hypothesis);
	vector<double> area_segments_normalized = normalizeData(area_segments);
	vector<double> scores;
	for (int j = 0; j < ratios_normalized.size(); j++) {
		double score = (area_hypothesis_normalized[j] + ratios_normalized[j] + area_segments_normalized[j]) / 3;
		scores.push_back(score);
	}

	vector<int> refined_indexes;
	for (int i = 0; i < scores.size(); i++) {

		if (scores[i] > threshold1 && ratios_normalized[i] > threshold2 && area_hypothesis_normalized[i] > 0.03 && area_hypothesis_normalized[i] < 0.1) {
			refined_indexes.push_back(i);
		}
	}

	return refined_indexes;

}

// Checks whether or not two given bounding boxes are intersecting each other
bool checkOverlappaing(int top1, int bottom1, int left1, int right1, int top2, int bottom2, int left2, int right2, int tolerance = 5000) {
	Rect bb1(top1, left1, right1 - left1, bottom1 - top1); // Bounding box 1
	Rect bb2(top2, left2, right2 - left2, bottom2 - top2); // Bounding box 2

	bool intersects = ((bb1 & bb2).area() > tolerance);
	return intersects;
}

// Computes Global Refinement of all hypothesis
vector<int> globalHypothesisRefinement(vector<vector<int>> hypothesis, vector<int> local_refined_hypothesis_indexes, vector<int> area_hypothesis, vector<int> area_segments) {

	vector<int> indexes_to_be_removed;

	for (int i = 0; i < local_refined_hypothesis_indexes.size(); i++) {
		int index_i = local_refined_hypothesis_indexes[i];
		for (int j = 1; j < local_refined_hypothesis_indexes.size(); j++) {
			int index_j = local_refined_hypothesis_indexes[j];
			if (index_i == index_j) continue;
			if (checkOverlappaing(hypothesis[0][index_i], hypothesis[1][index_i], hypothesis[2][index_i], hypothesis[3][index_i], hypothesis[0][index_j], hypothesis[1][index_j], hypothesis[2][index_j], hypothesis[3][index_j])) {
				if (area_hypothesis[index_i] > area_hypothesis[index_j]) {
					indexes_to_be_removed.push_back(index_j);
				}else {
					indexes_to_be_removed.push_back(index_i);
				}
			}
		}
	}

	vector<int> global_refined_hypothesis_indexes;
	for (int i = 0; i < local_refined_hypothesis_indexes.size(); i++) {
		int index = local_refined_hypothesis_indexes[i];
		bool isIndexToBeRemoved = find(indexes_to_be_removed.begin(), indexes_to_be_removed.end(), index) != indexes_to_be_removed.end();
		if (!isIndexToBeRemoved) {
			global_refined_hypothesis_indexes.push_back(local_refined_hypothesis_indexes[i]);
		}
	}

	return global_refined_hypothesis_indexes;

}

// Draws and displays a bounding box 
void drawingHypothesis(Mat img, int top, int bottom, int left, int right, int line = -1) {
	Point pt1(left, bottom);
	Point pt2(right, top);
	rectangle(img, pt1, pt2, Scalar(0, 255, 0), line);
	namedWindow("Hipoteses", WINDOW_AUTOSIZE);
	imshow("Hipoteses", img);

	waitKey(0);
}

// Draws and displays a bounding box 
void saveHypothesis(Mat img, string path, vector<int> indexes, vector<int> tops, vector<int> bottoms, vector<int> lefts, vector<int> rights) {
	Mat img_bw = Mat(img.rows, img.cols, CV_64F, cvScalar(0.));
	for (int i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int top = tops[index];
		int bottom = bottoms[index];
		int left = lefts[index];
		int right = rights[index];
		
		Point pt1(left, bottom);
		Point pt2(right, top);
		
		rectangle(img_bw, pt1, pt2, Scalar(255, 255, 255), FILLED);
	}
		
	imwrite(path, img_bw);
}


int main() {

	string africanos_path = "dataset_satelite/africanos/";
	string nhozinho_path = "dataset_satelite/nhozinho/";
	vector<string> imagesPaths = getAllImagesInDirectory(africanos_path);

	float sigma = 0.3;
	int k = 100;
	int min_size = 1000;

	for (int i = 0; i < imagesPaths.size(); i++) {

		string imagePath = imagesPaths.at(i);
		Mat src = imread(imagePath);
		if (!src.data) return -1;

		GraphSegmentation segmenter;
		vector<vector<int>> hypothesis = segmenter.executeGraphSegmentation(imagePath, sigma, k, min_size);
		vector<int> area_hypothesis = computeAllHypothesisCovagere(hypothesis);

		//cout << "Computing Local Hypothesis Refinement..." << endl;

		// Refinamento local das hipoteses geradas
		vector<int> hypothesis_locally_refined_indexes = localHypothesisRefinement(area_hypothesis, hypothesis[4], 0.2, 0.6);
		/*for (int j = 0; j < hypothesis_locally_refined_indexes.size(); j++) {
			int index = hypothesis_locally_refined_indexes[j];
			cout << "Index (" << index << ") -->" << area_hypothesis[index] << endl;
			drawingHypothesis(src.clone(), hypothesis[0][index], hypothesis[1][index], hypothesis[2][index], hypothesis[3][index]);
		}*/
		//cout << "Local Hypothesis Refinement has finished!" << endl;

		string imageName = imagePath.substr(imagePath.find_last_of("/\\") + 1);

		//cout << "Computing Global Hypothesis Refinement..." << endl;

		// Refinamento global das hipoteses geradas
		vector<int> hypothesis_globaly_refined_indexes = globalHypothesisRefinement(hypothesis, hypothesis_locally_refined_indexes, area_hypothesis, hypothesis[4]);
		saveHypothesis(src.clone(), "./Output/" + imageName, hypothesis_globaly_refined_indexes, hypothesis[0], hypothesis[1], hypothesis[2], hypothesis[3]);

		/*for (int j = 0; j < hypothesis_globaly_refined_indexes.size(); j++) {
			int index = hypothesis_globaly_refined_indexes[j];
			cout << "Index (" << index << ") -->" << area_hypothesis[index] << endl;
			drawingHypothesis(src.clone(), hypothesis[0][index], hypothesis[1][index], hypothesis[2][index], hypothesis[3][index], -1);
		}*/

		//cout << "Global Hypothesis Refinement has finished!" << endl;

	}
	return 0;
}