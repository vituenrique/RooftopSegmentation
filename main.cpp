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

vector<string> getAllImagesInDirectory(string path) {
	vector<string> imagesPath;
	cout << path << endl;
	for (const auto & entry : directory_iterator(path)) {
		imagesPath.push_back(entry.path().string());
	}
	return imagesPath;
}

void applyHoughTranform(Mat image, string filename, string output_dir) {

	Mat src_gauss, finalImage = image.clone();

	GaussianBlur(image, src_gauss, Size(3, 3), 0, 0, BORDER_DEFAULT);
	addWeighted(src_gauss, 1.5, image, -0.5, 0, src_gauss);

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

int computeArea(int w, int h) {
	return w * h;
}

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

//TODO: - Descobrir como calcular a porcentagem de overlap entre o segmento e a hipotese
//      - Adicionar a porcentagem de overlap no calculo da média
vector<int> localHypothesisRefinement(vector<int> area_hypothesis, vector<int> area_segments, double threshold = 0.1){
	vector<double> ratios;
	
	for (int i = 0; i < area_segments.size(); i++) {
		double ratio = (double)area_hypothesis[i] / area_segments[i];
		cout << area_hypothesis[i] << " / " << area_segments[i] << "= " << ratio << endl;
		ratios.push_back(ratio);
		
	} 
	cout << endl;
	cout << endl;
	vector<double> ratios_normalized = normalizeData(ratios);
	vector<double> area_hypothesis_normalized = normalizeData(area_hypothesis);
	vector<double> scores;
	for (int j = 0; j < ratios_normalized.size(); j++) {
		double score = (area_hypothesis_normalized[j] + ratios_normalized[j]) / 2;
		scores.push_back(score);
		cout << "(" << area_hypothesis_normalized[j] << " + " << ratios_normalized[j] << ")/ 2 = " << score << endl;
	}

	cout << endl;
	cout << endl;
	vector<int> refined_indexes;
	for (int i = 0; i < scores.size(); i++) {
		if (scores[i] > threshold) {
			refined_indexes.push_back(i);
			cout << i << endl;
		}
	}

	return refined_indexes;

}

int main() {

	string africanos_path = "dataset_satelite/africanos/";
	string nhozinho_path = "dataset_satelite/nhozinho/";
	vector<string> imagesPaths = getAllImagesInDirectory(africanos_path);

	float sigma = 0.5;
	int k = 300;
	int min_size = 2000;
	
	for (int i = 0; i < imagesPaths.size(); i++) {
		string imagePath = imagesPaths.at(i);

		Mat src = imread(imagePath);
		 
		if (!src.data) return -1; 
		
		//applyHoughTranform(src, imagePath, "output_lines_3_enhanced/");

		GraphSegmentation segmenter;
		vector<vector<int>> hypothesis = segmenter.executeGraphSegmentation(imagePath, sigma, k, min_size);

		vector<int> area_hypothesis = computeAllHypothesisCovagere(hypothesis);

		// Refinamento das hipoteses geradas
		vector<int> hypothesis_refined = localHypothesisRefinement(area_hypothesis, hypothesis[4]);

		break;

	}

	system("pause");

	return 0;
}