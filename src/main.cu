#include <iostream>
#include <vector>

#include "mat.h"

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "helper.cuh"
#include "dataset.h"
#include "tsdf_volume.h"
#include "marching_cubes.h"


#define STR1(x)  #x
#define STR(x)  STR1(x)


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


bool depthToVertexMap(const Mat3f &K, const cv::Mat &depth, cv::Mat &vertexMap)
{
    if (depth.type() != CV_32FC1 || depth.empty())
        return false;

    int w = depth.cols;
    int h = depth.rows;
    vertexMap = cv::Mat::zeros(h, w, CV_32FC3);
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;
    float* ptrVert = (float*)vertexMap.data;

    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float depthMeter = ptrDepth[y*w + x];
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;

            size_t off = (y*w + x) * 3;
            ptrVert[off] = x0 * depthMeter;
            ptrVert[off+1] = y0 * depthMeter;
            ptrVert[off+2] = depthMeter;
        }
    }

    return true;
}


Vec3f centroid(const cv::Mat &vertexMap)
{
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y)
    {
        for (int x = 0; x < vertexMap.cols; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0)
            {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centroid /= float(cnt);

    return centroid;
}


int main(int argc, char *argv[])
{
//    Vec6f gradient;
    Vec6f ksi;

//    gradient << 1,2,3,4,5,6;
//    ksi << 10,11,12,13,14,15;

//    Eigen::MatrixXf _res_A = A(gradient,7);
//    Eigen::MatrixXf _res_b = b(gradient, ksi, 8, 10);

//    std::cout << "A: " << _res_A << std::endl;
//    std::cout << "b: " << _res_b << std::endl;

    Mat4f projection_example;
    projection_example << 2, 0, 0, 2,  0, 0, -2, 4,  0, 2, 0, 6,  0, 0, 0, 2;

    projectionMatrixToKsi(projection_example, ksi);
    std::cout<<"ksi: \n"<<ksi<<std::endl;

    ksiToTransformationMatrix(ksi, projection_example);
    std::cout<<"projection: \n"<<projection_example<<std::endl;

//    Mat3f rotation = projection_example.block(0,0,3,3);
//    std::cout<<"rotation: "<<rotation<<std::endl;

//    std::cout<<" isRotation: "<<_isRotation(rotation);

    // default input sequence in folder
    std::string dataFolder = std::string(STR(SDF2SDF_SOURCE_DIR)) + "/data/kinect1/";

    // parse command line parameters
    const char *params = {
        "{i|input| |input rgb-d sequence}"
        "{f|frames|10000|number of frames to process (0=all)}"
        "{n|iterations|100|max number of GD iterations}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input sequence
    // download from http://campar.in.tum.de/personal/slavcheva/3d-printed-dataset/index.html
    std::string inputSequence = cmd.get<std::string>("input");
    if (inputSequence.empty())
    {
        inputSequence = dataFolder;
        //std::cerr << "No input sequence specified!" << std::endl;
        //return 1;
    }
    std::cout << "input sequence: " << inputSequence << std::endl;
    // number of frames to process
    size_t frames = (size_t)cmd.get<int>("frames");
    std::cout << "# frames: " << frames << std::endl;
    // max number of GD iterations
    size_t iterations = (size_t)cmd.get<int>("iterations");
    std::cout << "iterations: " << iterations << std::endl;

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    // load camera intrinsics
    Eigen::Matrix3f K;
    if (!loadIntrinsics(inputSequence + "/intrinsics_kinect1.txt", K))
    {
        std::cerr << "No intrinsics file found!" << std::endl;
        return 1;
    }
    std::cout << "K: " << std::endl << K << std::endl;

    // create tsdf volume
    Vec3i volDim(256, 256, 256);
    Vec3f volSize(1.0f, 1.0f, 1.0f);
    TSDFVolume* tsdf = new TSDFVolume(volDim, volSize, K);

    // create windows
    cv::namedWindow("color");
    cv::namedWindow("depth");
    cv::namedWindow("mask");

    // process frames
    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth, mask;
    for (size_t i = 0; i < frames; ++i)
    {
        // load input frame
        if (!loadFrame(inputSequence, i, color, depth, mask))
        {
            //std::cerr << "Frame " << i << " could not be loaded!" << std::endl;
            //return 1;
            break;
        }

        // filter depth values outside of mask
        filterDepth(mask, depth);

        // show input images
        cv::imshow("color", color);
        cv::imshow("depth", depth);
        cv::imshow("mask", mask);
        cv::waitKey();

        // get initial volume pose from centroid of first depth map
        if (i == 0)
        {
            // initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;
        }
        // integrate frame into tsdf volume
        tsdf->integrate(poseVolume, color, depth);

        // TODO: compute next ksi
    }

    // extract mesh using marching cubes
    std::cout << "Extracting mesh..." << std::endl;
    MarchingCubes mc(volDim, volSize);
    mc.computeIsoSurface(tsdf->ptrTsdf(), tsdf->ptrTsdfWeights(), tsdf->ptrColorR(), tsdf->ptrColorG(), tsdf->ptrColorB());

    // save mesh
    std::cout << "Saving mesh..." << std::endl;
    const std::string meshFilename = inputSequence + "/mesh.ply";
    if (!mc.savePly(meshFilename))
    {
        std::cerr << "Could not save mesh!" << std::endl;
    }

    // clean up
    delete tsdf;
    cv::destroyAllWindows();

    return 0;
}
