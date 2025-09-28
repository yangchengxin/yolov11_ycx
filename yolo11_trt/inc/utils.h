#pragma once
#include <iostream>
#include <NvInfer.h>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

struct alignas(float) Detectionx {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
    float keypoints[51];  // 17*3 keypoints
};

void nms(std::vector<Detectionx>& res, float* output, float conf_thresh, float nms_thresh);

class Utils
{
public:
    Utils()
    {    }
    ~Utils()
    {    }

    /* ------ 将cv图像编码成base64 ------- */
    std::string base64Encode(unsigned char const* bytes_to_encode, unsigned int in_len);

    /* ------ 将base64图像解码成cv ------- */
    std::vector<unsigned char> base64Decode(std::string base64img);
};
