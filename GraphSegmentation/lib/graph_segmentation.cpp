#include "graph_segmentation.h"
#include <limits>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

void GraphSegmentation::buildGraph(const cv::Mat &image) {
    
    H = image.rows;
    W = image.cols;
    
    int N = H*W;
    graph = ImageGraph(N);
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            int n = W*i + j;
            ImageNode & node = graph.getNode(n);
            
            cv::Vec3b bgr = image.at<cv::Vec3b>(i, j);
            node.b = bgr[0];
            node.g = bgr[1];
            node.r = bgr[2];

            // Initialize label.
            node.l = n;
            node.id = n;
            node.n = 1;
        }
    }
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int n = W*i + j;
            const ImageNode & node = graph.getNode(n);

            if (i < H - 1) {
                int m = W*(i + 1) + j;
                ImageNode & other = graph.getNode(m);

                ImageEdge edge;
                edge.n = n;
                edge.m = m;
                edge.w = (*distance)(node, other);

                graph.addEdge(edge);
            }

            if (j < W - 1) {
                int m = W*i + (j + 1);
                ImageNode & other = graph.getNode(m);

                ImageEdge edge;
                edge.n = n;
                edge.m = m;
                edge.w = (*distance)(node, other);

                graph.addEdge(edge);
            }
        }
    }
}

void GraphSegmentation::oversegmentGraph() {
    
    // Sort edges.
    graph.sortEdges();
    
    for (int e = 0; e < graph.getNumEdges(); e++) {
        ImageEdge edge = graph.getEdge(e%graph.getNumEdges());
        
        ImageNode & n = graph.getNode(edge.n);
        ImageNode & m = graph.getNode(edge.m);

        ImageNode & S_n = graph.findNodeComponent(n);
        ImageNode & S_m = graph.findNodeComponent(m);

        // Are the nodes in different components?
        if (S_m.id != S_n.id) {

            // Here comes the magic!
            if ((*magic)(S_n, S_m, edge)) {
                graph.merge(S_n, S_m, edge);
            }
        }
    }
}

std::vector<std::vector<int>>  GraphSegmentation::executeGraphSegmentation(std::string imagePath, float sigma, int k, int min_size) {
	std::string sigmaS = std::to_string(sigma);
	std::string kS = std::to_string(k);
	std::string min_sizeS = std::to_string(min_size);
	std::string output_path = extraModulesPath + "Hipoteses_" + min_sizeS + "/";
	std::string magic_call = executer + " " + source + " " + imagePath + " " + output_path + " " + sigmaS + " " + kS + " " + min_sizeS;

	system(magic_call.c_str());

	std::string imageName = imagePath.substr(imagePath.find_last_of("/\\") + 1);

	for (int i = 0; i < 4; i++) imageName.pop_back();

	int number = std::atoi(imageName.c_str());
	imageName = std::to_string(number);
	if(getHypothesisComputed(imageName, output_path))
		return hypothesis;
}

bool GraphSegmentation::getHypothesisComputed(std::string imageName, std::string segmentsPath) {
	std::ifstream file;
	std::string segments_path = segmentsPath + "hipoteses/" + imageName + ".csv";
	file.open(segments_path);

	std::vector<int> top;
	std::vector<int> bottom;
	std::vector<int> left;
	std::vector<int> right;
	std::vector<int> area_segment;

	while (file.good()) {
		std::string line;
		std::getline(file, line, ',');
		top.push_back(std::atoi(line.c_str()));

		std::getline(file, line, ',');
		bottom.push_back(std::atoi(line.c_str()));

		std::getline(file, line, ',');
		left.push_back(std::atoi(line.c_str()));

		std::getline(file, line, ',');
		right.push_back(std::atoi(line.c_str()));

		std::getline(file, line, '\n');
		area_segment.push_back(std::atoi(line.c_str()));
	}

	top.pop_back();
	bottom.pop_back();
	left.pop_back();
	right.pop_back();

	area_segment.pop_back();
	hypothesis.push_back(top);
	hypothesis.push_back(bottom);
	hypothesis.push_back(left);
	hypothesis.push_back(right);
	hypothesis.push_back(area_segment);

	return true;
}

void GraphSegmentation::enforceMinimumSegmentSize(int M) {
    assert(graph.getNumNodes() > 0);
    // assert(graph.getNumEdges() > 0);
    
    for (int e = 0; e < graph.getNumEdges(); e++) {
        ImageEdge edge = graph.getEdge(e);
        
        ImageNode & n = graph.getNode(edge.n);
        ImageNode & m = graph.getNode(edge.m);

        ImageNode & S_n = graph.findNodeComponent(n);
        ImageNode & S_m = graph.findNodeComponent(m);

        if (S_n.l != S_m.l) {
            if (S_n.n < M || S_m.n < M) {
                graph.merge(S_n, S_m, edge);
            }
        }
    }
}

cv::Mat GraphSegmentation::deriveLabels() {
    
    cv::Mat labels(H, W, CV_32SC1, cv::Scalar(0));
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int n = W*i + j;

            ImageNode & node = graph.getNode(n);
            ImageNode & S_node = graph.findNodeComponent(node);

            const int max = std::numeric_limits<int>::max();

            labels.at<int>(i, j) = S_node.id;
        }
    }
    
    return labels;
}