// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HELPER_CUH
#define TUM_HELPER_CUH

#include "mat.h"
#include <cuda_runtime_api.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>


// CUDA utility functions

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

// compute grid size from block size
inline dim3 computeGrid1D(const dim3 &block, const int w)
{
    return dim3(0, 0, 0);   // TODO (3.2) compute 1D grid size from block size
}

inline dim3 computeGrid2D(const dim3 &block, const int w, const int h)
{
    return dim3(0, 0, 0);   // TODO (3.2) compute 2D grid size from block size
}

inline dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s)
{
    return dim3(0, 0, 0);   // TODO (3.2) compute 3D grid size from block size
}


Eigen::MatrixXf computeA(Vec6f &gradient);   //phi is phi_current from the paper
Eigen::MatrixXf computeb(Vec6f &gradient, Vec6f &ksi, float phi_cur, float phi_ref);

Mat4f normalizeProjection(Mat4f proj);

bool _isRotation(Mat3f& rotation);

// OpenCV image conversion
// interleaved to layered
void convertMatToLayered(float *aOut, const cv::Mat &mIn);

// layered to interleaved
void convertLayeredToMat(cv::Mat &mOut, const float *aIn);


// OpenCV GUI functions
// open camera
bool openCamera(cv::VideoCapture &camera, int device, int w = 640, int h = 480);

// show image
void showImage(std::string title, const cv::Mat &mat, int x, int y);

// show histogram
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);


// adding Gaussian noise
void addNoise(cv::Mat &m, float sigma);

// convert ksi vector representation of a transformation matrix back
void ksiToTransformationMatrix(const Eigen::VectorXf &ksi, Mat4f &transform);

// convert a projection matrix to ksi vector representation
void projectionMatrixToKsi(const Mat4f &proj, Vec6f &ksi);

// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

#endif
