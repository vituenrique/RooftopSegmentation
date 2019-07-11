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

//TODO: Veja o que da pra fazer aqui. Voce já tem a área de cada hipotese e a área de seus respectivos segmentos. Talvez seja o caso pensar melhor no calculo das probabilidades.
vector<int> localHypothesisRefinement(vector<double> area_hypothesis_normalized, vector<double> area_segments_normalized){
	for (int j = 0; j < area_segments_normalized.size(); j++) {
		cout << "Area da Hipotese: " << area_hypothesis_normalized[j] << " | " << " Area do Segmento: " << area_segments_normalized[j] << endl;
	}
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

		vector<double> area_hypothesis_normalized = normalizeData(area_hypothesis);
		vector<double> area_segments_normalized = normalizeData(hypothesis[4]);

		break;

	}

	system("pause");

	return 0;
}