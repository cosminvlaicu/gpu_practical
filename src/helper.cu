// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "helper.cuh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <limits>

#include "tsdf_volume.h"



// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cout << std::endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line > 0)
            std::cout << "Previous CUDA call:" << std::endl << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


// OpenCV image conversion: layered to interleaved
void convertLayeredToInterleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convertLayeredToMat(cv::Mat &mOut, const float *aIn)
{
    convertLayeredToInterleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


// OpenCV image conversion: interleaved to layered
void convertInterleavedToLayered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}

void convertMatToLayered(float *aOut, const cv::Mat &mIn)
{
    convertInterleavedToLayered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}

// show cv:Mat in OpenCV GUI
// open camera using OpenCV
bool openCamera(cv::VideoCapture &camera, int device, int w, int h)
{
    if(!camera.open(device))
    {
        return false;
    }
    camera.set(CV_CAP_PROP_FRAME_WIDTH, w);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, h);
    return true;
}

// show cv:Mat in OpenCV GUI
void showImage(std::string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cv::moveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}

// show histogram in OpenCV GUI
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    const int nbins = 256;
    cv::Mat canvas = cv::Mat::ones(125, 512, CV_8UC3);

    float hmax = 0;
    for(int i = 0; i < nbins; ++i)
        hmax = max((int)hmax, histogram[i]);

    for (int j = 0, rows = canvas.rows; j < nbins-1; j++)
    {
        for(int i = 0; i < 2; ++i)
            cv::line(
                        canvas,
                        cv::Point(j*2+i, rows),
                        cv::Point(j*2+i, rows - (histogram[j] * 125.0f) / hmax),
                        cv::Scalar(255,128,0),
                        1, 8, 0
                        );
    }

    showImage(windowTitle, canvas, windowX, windowY);
}


// add Gaussian noise
float noise(float sigma)
{
    float x1 = (float)rand()/RAND_MAX;
    float x2 = (float)rand()/RAND_MAX;
    return sigma * sqrtf(-2*log(std::max(x1,0.000001f)))*cosf(2*M_PI*x2);
}

void addNoise(cv::Mat &m, float sigma)
{
    float *data = (float*)m.data;
    int w = m.cols;
    int h = m.rows;
    int nc = m.channels();
    size_t n = (size_t)w*h*nc;
    for(size_t i=0; i<n; i++)
    {
        data[i] += noise(sigma);
    }
}

void skewSymmetric(const Vec3f &vec, Mat3f &matX)
{
    matX << 0, -vec(2), vec(1),
          vec(2), 0, -vec(0),
          -vec(1), vec(0), 0;
}

Mat4f normalizeProjection(Mat4f proj)
{
    return (proj / proj(3,3));
}

bool _isRotation(Mat3f& rotation)
{
    Vec3f v1 = rotation.block(0,0,3,1);
    Vec3f v2 = rotation.block(0,1,3,2);
    Vec3f v3 = rotation.block(0,2,3,3);

    if(abs(v1.norm() - 1) > 0.01 || abs(v2.norm() - 1) > 0.01 || abs(v2.norm() - 1) > 0.01 )
        return false;

    float value1 = v1.transpose()*v2;
    float value2 = v1.transpose()*v3;
    float value3 = v2.transpose()*v3;

    if(abs(value1) > 0.01 || abs(value2) > 0.01 || abs(value3) > 0.01)
        return false;

    return true;
}

void ksiToTransformationMatrix(const VecXf &ksi, Mat4f &transform)
{
    auto u = ksi.head(3);
    auto w = ksi.tail(3);
    assert(u.size() + w.size() == ksi.size());

    float theta = sqrt(w.dot(w));
    float A = sin(theta) / theta;
    float B = (1 - cos(theta)) / (theta * theta);
    float C = (1 - A) / (theta * theta);

    Mat3f wx;
    skewSymmetric(w, wx);
    Mat3f R = Mat3f::Identity() + A * wx + B * (wx * wx);
    Mat3f V = Mat3f::Identity() + B * wx + C * (wx * wx);

    transform = Mat4f::Identity();
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = V * u;
    transform = normalizeProjection(transform);
}


void projectionMatrixToKsi(const Mat4f &proj, Vec6f &ksi)
{
    //inverse transformation (from matrix to twist)
    Mat4f projection_normalized = normalizeProjection(proj);

    //extract rotation
    Mat3f rotation = projection_normalized.block(0,0,3,3);

    //extract translation
    Vec3f translation(3);
    translation(0) = projection_normalized(0,3);
    translation(1) = projection_normalized(1,3);
    translation(2) = projection_normalized(2,3);

    //angle
    float theta = acos( (rotation.trace() -1) / 2 );

    //coefficients
    float A = (sin(theta))/theta;
    float B = ( 1 - cos(theta) ) / (theta*theta);
    float C = (1 - A) / (theta * theta);

    //skew-symmetric
    Mat3f omega = (theta / ( 2*sin(theta)) ) * (rotation - rotation.transpose() );
    Mat3f V = Mat3f::Identity() + B * omega + C * (omega * omega);

    Vec3f u = V.inverse()*translation;

    ksi(0) = u(0);
    ksi(1) = u(1);
    ksi(2) = u(2);
    ksi(3) = omega(2,1);
    ksi(4) = omega(0,2);
    ksi(5) = omega(1,0);
}


void computeGradientByX(const TSDFVolume &tsdf, Vec3f *gradientByX)
{
    Vec3i dim = tsdf.dimensions();
    float *sdfs = tsdf.ptrTsdf();
    Vec3f dX = tsdf.voxelSize() * 2;
    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                size_t off = z*dim[0]*dim[1] + y*dim[0] + x;
                size_t offX2 = z*dim[0]*dim[1] + y*dim[0] + x + (x == dim[0] - 1) ? 0 : 1;
                size_t offX1 = z*dim[0]*dim[1] + y*dim[0] + x - (x == 0) ? 0 : 1;
                size_t offY2 = z*dim[0]*dim[1] + (y + (y == dim[1] - 1) ? 0 : 1)*dim[0] + x;
                size_t offY1 = z*dim[0]*dim[1] + (y - (y == 0) ? 0 : 1)*dim[0] + x;
                size_t offZ2 = (z + (z == dim[2] - 1) ? 0 : 1)*dim[0]*dim[1] + y*dim[0] + x;
                size_t offZ1 = (z - (z == 0) ? 0 : 1)*dim[0]*dim[1] + y*dim[0] + x;
                gradientByX[off][0] = (sdfs[offX2] - sdfs[offX1]) / dX[0];
                gradientByX[off][1] = (sdfs[offY2] - sdfs[offY1]) / dX[1];
                gradientByX[off][2] = (sdfs[offZ2] - sdfs[offZ1]) / dX[2];
            }
        }
    }
}

Eigen::MatrixXf computeA(Vec6f &gradient)   //phi is phi_current from the paper
{
    //define as matrices to ease computation
    Eigen::MatrixXf gradient_matrix(1,6);

    //initialize gradient as a matrix
    gradient_matrix(0,0) = gradient(0);
    gradient_matrix(0,1) = gradient(1);
    gradient_matrix(0,2) = gradient(2);
    gradient_matrix(0,3) = gradient(3);
    gradient_matrix(0,4) = gradient(4);
    gradient_matrix(0,5) = gradient(5);

    //initialize result
    Eigen::MatrixXf A_m(6,6);

    A_m = gradient_matrix.transpose() * gradient_matrix;
    return A_m;
}

Eigen::MatrixXf computeb(Vec6f &gradient, Vec6f &ksi, float phi_cur, float phi_ref)
{
    //phi is phi_cur in the paper
    Eigen::MatrixXf b(6,1);

    //define as matrices to ease computation
    Eigen::MatrixXf gradient_matrix(1,6);
    Eigen::MatrixXf ksi_matrix(6,1);

    //initialize gradient as a matrix
    gradient_matrix(0,0) = gradient(0);
    gradient_matrix(0,1) = gradient(1);
    gradient_matrix(0,2) = gradient(2);
    gradient_matrix(0,3) = gradient(3);
    gradient_matrix(0,4) = gradient(4);
    gradient_matrix(0,5) = gradient(5);

    //initialize ksi as a matrix
    ksi_matrix(0,0) = ksi(0,0);
    ksi_matrix(1,0) = ksi(1,0);
    ksi_matrix(2,0) = ksi(2,0);
    ksi_matrix(3,0) = ksi(3,0);
    ksi_matrix(4,0) = ksi(4,0);
    ksi_matrix(5,0) = ksi(5,0);

    //first paranthesis from computation of b
    Eigen::MatrixXf aux_res = (gradient_matrix * ksi_matrix);
    float aux_b = aux_res(0,0) + phi_ref - phi_cur;

    b = gradient_matrix.transpose();
    b *= aux_b;

    return b;
}

void computeGradient(const TSDFVolume &tsdf, const Mat4f &ksiProjection, Vec6f *gradient);


Vec6f computeStepParameters(const TSDFVolume &tsdf_cur, const TSDFVolume &tsdf_ref, Vec6f &ksi_cur)
{
    //compute result (update step)
    Eigen::MatrixXf result(1,6);

    //volume dimensions
    Vec3i dim = tsdf_cur.dimensions();

    //pointers to sdf current
    float* sdfs_cur = tsdf_cur.ptrTsdf();
    float* wsdfs_cur = tsdf_cur.ptrTsdfWeights();

    //pointers to sdf reference
    float* sdfs_ref = tsdf_ref.ptrTsdf();
    float* wsdfs_ref = tsdf_ref.ptrTsdfWeights();


    //declare intermediate results
    Eigen::MatrixXf b_m(6,1);
    Eigen::MatrixXf A_m(6,6);

    //initialize intermediate results with 0
    A_m << 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0;
    b_m<<0,0,0,0,0,0;

    //pass pose for gradient computation and compute gradient
    Mat4f ksiMatrix_cur;
    ksiToTransformationMatrix(ksi_cur, ksiMatrix_cur);

    //compute gradient
    Vec6f* gradient = new Vec6f[tsdf_ref.gridSize() ];
    computeGradient(tsdf_cur, ksiMatrix_cur, gradient);

    //(later TODO: CUDA)
    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                //compute global offset
                size_t off = z*dim[0]*dim[1] + y*dim[0] + x;

                //parameteres for current frame
                float phi_cur = sdfs_cur[off];
                float weight_cur = wsdfs_cur[off];
                phi_cur *= weight_cur;

                //parameteres for reference frame
                float phi_ref = sdfs_ref[off];
                float weight_ref = wsdfs_ref[off];
                phi_ref *= weight_ref;

                //do the summation (later TODO: CUDA)
                A_m += computeA(gradient[off]);
                b_m += computeb(gradient[off], ksi_cur, phi_cur, phi_ref);
            }
        }
    }

    delete[] gradient;
    //assert it is invertible
    return (A_m.inverse() * b_m);
}

void findStep(const TSDFVolume &tsdf_reference, const TSDFVolume &tsdf_new_frame, const float threshold, const float step_size)    //tsdf is the general, unmodified, global tsdf; ksi is the pose of the current frame
{
    //pose for this frame is identity
    Mat4f reference_pose = Mat4f::Identity();

    //define twist coordinates poses
    Vec6f ksi_reference, ksi_update;

    //initialize twist coordinates poses : our initial guess is the identity
    ksiToTransformationMatrix(ksi_reference, reference_pose);
    ksi_update = ksi_reference;

    //initialize error to maximum float
    float error = std::numeric_limits<float>::max();

    while(error > threshold)    //do until convergence
    {
        //compute update step
        Vec6f update = computeStepParameters(tsdf_new_frame, tsdf_reference, ksi_update);

        //update current guess
        ksi_update = ksi_update + step_size * (update - ksi_update);

        //compute the error: here the error is absolute value of the max element of update*step_size
        error = update.array().abs().sum();

    }


}

void computeGradient(const TSDFVolume &tsdf, const Mat4f &ksiProjection, Vec6f *gradient)
{
    //TODO (Mykola)

    Vec3i dim = tsdf.dimensions();
    Vec3f *gradientByX = new Vec3f[tsdf.gridSize()];
    computeGradientByX(tsdf, gradientByX);

    Mat4f ksiProjectionInverse = ksiProjection.inverse();

    // calculate gradient for each voxel
    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                Vec3i vx(x, y, z);
                Vec3f pt = tsdf.voxelToWorld(vx);

                Vec3f ksiV = (ksiProjectionInverse * pt.homogeneous()).head(3);
                Mat3f ksiVX;
                skewSymmetric(ksiV, ksiVX);

                Eigen::MatrixXf concatenation(3, 6);
                concatenation << Mat3f::Identity(), -1 * ksiVX;

                size_t off = z*dim[0]*dim[1] + y*dim[0] + x;
                gradient[off] = gradientByX[off].transpose() * concatenation;
            }
        }
    }
    delete[] gradientByX;
}

