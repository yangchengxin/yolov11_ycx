#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
//#include "utils.h"
#include "TRTLogger.h"
#include "common.h"
#include "postprocess.h"

#ifndef CPPDLL_EXPORTS
#define CPPDLL_API __declspec(dllexport)
#else
#define CPPDLL_API __declspec(dllimport)
#endif

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class CPPDLL_API yolov11
{

public:

    yolov11();
    ~yolov11();

    //std::vector<float> DetectiontRects;

    void setGpu() { this->useGpu == true; }
    void setCpu() { this->useGpu == false; }

    int init(const char* det_model, const char* cls_model, const char* cls2_model, const char* machine_type);
    cv::Mat gammaCorrection(const cv::Mat image, double gamma);
    void preprocess(cv::Mat image);
    void infer();
    void postprocess(vector<Detection>& output, std::string& result, int inputx, int inputy, int inputw, int inputh);
    void cuda_postprocess(cv::Mat src, std::string& result, std::vector<float>& DetectiontRects, int width, int height, int inputx, int inputy);

    std::string det(unsigned char* imgbuf, std::vector<Detection>& output, int width, int height, float thresh, int inputx, int inputy, int inputw, int inputh, int lingmindu, int dongjingtai);
    std::string cuda_det(unsigned char* imgbuf, std::vector<float>& output, int width, int height, float thresh, int inputx, int inputy, int inputw, int inputh, int lingmindu, int dongjingtai);

    void draw(Mat& image, const vector<Detection>& output);
    void cuda_draw(cv::Mat& image, std::vector<float>& DetectiontRects);

private:
    //void init(std::string engine_path, nvinfer1::ILogger& logger);
    bool useGpu;
    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    int* GpuOutputCount = nullptr;
    DetectRect* GpuOutputRects = nullptr;
    int* CpuOutputCount = nullptr;
    DetectRect* CpuOutputRects = nullptr;

    cudaStream_t stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    Logger logger;
    string modelPath;  //engine
    int input_w;
    int input_h;
    int batchsize;
    int num_detections;
    int detection_attribute_size;
    int num_classes = BenignNames.size();
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;
    int det_nums = 1000;
    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};