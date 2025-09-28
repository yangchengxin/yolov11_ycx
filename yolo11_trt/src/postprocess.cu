#include "postprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <cmath>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

__global__ void GetNmsBeforeBoxesKernel(float* SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int* OutputCount)
{
    /***
    功能说明：用8400个线程，实现对80个类别选出最大值，并判断是否大于阈值，把大于阈值的框记录下来后面用于参加mns
    SrcInput: 模型输出（1,84,8400）
    AnchorCount: 8400
    ClassNum: 80
    ObjectThresh: 目标阈值（大于该阈值的目标才输出）
    NmsBeforeMaxNum: 输入nms检测框的最大数量，前面申请的了一块儿显存来装要参加nms的框，防止越界
    OutputRects: 大于阈值的目标框
    OutputCount: 大于阈值的目标框个数
    ***/

    int ThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (ThreadId >= AnchorCount)
    {
        return;
    }

    float* XywhConf = SrcInput + ThreadId;
    float CenterX = 0, CenterY = 0, CenterW = 0, CenterH = 0;

    float MaxScore = 0;
    int MaxIndex = 0;

    DetectRect TempRect;
    for (int j = 4; j < ClassNum + 4; j++)
    {
        if (4 == j)
        {
            MaxScore = XywhConf[j * AnchorCount];
            MaxIndex = j;
        }
        else
        {
            if (MaxScore < XywhConf[j * AnchorCount])
            {
                MaxScore = XywhConf[j * AnchorCount];
                MaxIndex = j;
            }
        }
    }

    if (MaxScore > ObjectThresh)
    {
        int index = atomicAdd(OutputCount, 1);

        if (index > NmsBeforeMaxNum)
        {
            return;
        }

        CenterX = XywhConf[0 * AnchorCount];
        CenterY = XywhConf[1 * AnchorCount];
        CenterW = XywhConf[2 * AnchorCount];
        CenterH = XywhConf[3 * AnchorCount];

        TempRect.classId = MaxIndex - 4;
        TempRect.score = MaxScore;
        TempRect.xmin = CenterX - 0.5 * CenterW;
        TempRect.ymin = CenterY - 0.5 * CenterH;
        TempRect.xmax = CenterX + 0.5 * CenterW;
        TempRect.ymax = CenterY + 0.5 * CenterH;

        OutputRects[index] = TempRect;
    }
}

void GetNmsBeforeBoxes(float* SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int* OutputCount, cudaStream_t Stream)
{
    int Block = 512;
    int Grid = (AnchorCount + Block - 1) / Block;

    GetNmsBeforeBoxesKernel << <Grid, Block, 0, Stream >> > (SrcInput, AnchorCount, ClassNum, ObjectThresh, NmsBeforeMaxNum, OutputRects, OutputCount);
    return;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

void GetConvDetectionResult(DetectRect* OutputRects, int* OutputCount, std::vector<float>& DetectiontRects, int InputW, int InputH, float NmsThresh, std::string& result)
{
    int ret = 0;
    std::vector<DetectRect> detectRects;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;

    DetectRect temp;
    for (int i = 0; i < *OutputCount; i++)
    {
        xmin = OutputRects[i].xmin;
        ymin = OutputRects[i].ymin;
        xmax = OutputRects[i].xmax;
        ymax = OutputRects[i].ymax;

        xmin = xmin > 0 ? xmin : 0;
        ymin = ymin > 0 ? ymin : 0;
        xmax = xmax < InputW ? xmax : InputW;
        ymax = ymax < InputH ? ymax : InputH;

        temp.xmin = xmin;
        temp.ymin = ymin;
        temp.xmax = xmax;
        temp.ymax = ymax;
        temp.classId = OutputRects[i].classId;
        temp.score = OutputRects[i].score;
        detectRects.push_back(temp);
    }

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect& Rect1, DetectRect& Rect2) -> bool
        { return (Rect1.score > Rect2.score); });

    // std::cout << "NMS Before num :" << detectRects.size() << std::endl;
    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > NmsThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }
}

