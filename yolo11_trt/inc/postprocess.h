#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
//#include <thrust/device_vector.h>
//#include <thrust/sort.h>
//#include <thrust/sequence.h>

struct DetectRect
{
    float classId;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

struct AffineMatrix {
    float value[6];
};

// 提取参与nms的检测结果（conf thresh）
void GetNmsBeforeBoxes(float* SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int* OutputCount, cudaStream_t Stream);
// nms + 结果写入指定的结构体数组中
void GetConvDetectionResult(DetectRect* OutputRects, int* OutputCount, std::vector<float>& DetectiontRects, int InputW, int InputH, float NmsThresh, std::string& result);