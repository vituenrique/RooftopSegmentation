#ifndef GRAPH_SEGMENTATION_H
#define	GRAPH_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include "image_graph.h"

#define RAND() ((float) std::rand() / (RAND_MAX))


class GraphSegmentationDistance {
public:

    GraphSegmentationDistance() {};

    virtual ~GraphSegmentationDistance() {};
    
    virtual float operator()(const ImageNode & n, const ImageNode & m) = 0;
    
};


class GraphSegmentationManhattenRGB : public GraphSegmentationDistance {
public:

    GraphSegmentationManhattenRGB() {
        // Normalization.
        D = 255 + 255 + 255;
    }
    
    virtual float operator()(const ImageNode & n, const ImageNode & m) {
        float dr = std::abs(n.r - m.r);
        float dg = std::abs(n.g - m.g);
        float db = std::abs(n.b - m.b);
        
        return (dr + dg + db);
    }
    
private:
    
    float D;
    
};

class GraphSegmentationEuclideanRGB : public GraphSegmentationDistance {
public:

    GraphSegmentationEuclideanRGB() {
        // Normalization.
        D = std::sqrt(255*255 + 255*255 + 255*255);
    }
   
	virtual float operator()(const ImageNode & n, const ImageNode & m) {
		long rmean = ((long) n.r + (long) m.r) / 2;
		float r = n.r - m.r;
		float g = n.g - m.g;
		float b = n.b - m.b;
		float p1 = 4 + pow(r, 2) * (2 + (rmean / 256));
		float p2 = pow(g, 2) + pow(b, 2) * (2 + ((256 - rmean) / 256));
		return sqrt(p1 + p2);
	}
    
private:
    
    /** \brief Normalization term. */
    float D;
    
};

class GraphSegmentationMagic {
public:

    GraphSegmentationMagic() {};
    
    virtual bool operator()(const ImageNode & S_n, const ImageNode & S_m, 
            const ImageEdge & e) = 0;
    
};

class GraphSegmentationMagicThreshold : public GraphSegmentationMagic {
public:

    GraphSegmentationMagicThreshold(float c) : c(c) {};
    
    virtual bool operator()(const ImageNode & S_n, const ImageNode & S_m, 
            const ImageEdge & e) {
        
        float threshold = std::min(S_n.max_w + c/S_n.n, S_m.max_w + c/S_m.n);
        
        if (e.w < threshold) {
            return true;
        }
        
        return false;
    }
    
private:
    
    /** \brief T hreshold. */
    float c;
    
};

class GraphSegmentation {
public:

    GraphSegmentation() : distance(new GraphSegmentationManhattenRGB()), 
            magic(new GraphSegmentationMagicThreshold(1)) {
        
    };
    
    virtual ~GraphSegmentation() {};
    
    void setDistance(GraphSegmentationDistance* _distance) {
        distance = _distance;
    }
    
    void setMagic(GraphSegmentationMagic* _magic) {
        magic = _magic;
    }
    
    void buildGraph(const cv::Mat &image);

	void executeGraphSegmentation(std::string imagePath, float sigma, int k, int min_size);
    
    void oversegmentGraph();
    
    void enforceMinimumSegmentSize(int M);
    
    cv::Mat deriveLabels();

	std::string executer = "C:/Users/victo/Anaconda3/envs/OpenCV/python";
	std::string extraModulesPath = "./GraphSegmentation/lib/ExtraModule/";
	std::string source = extraModulesPath + "segmentator.py";

    
protected:
    
    int H;
    
    int W;
    
    ImageGraph graph;
    
    GraphSegmentationDistance* distance;
    
    GraphSegmentationMagic* magic;

};

#endif	/* GRAPH_SEGMENTATION_H */

