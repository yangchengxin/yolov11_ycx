#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov11.h"
//#include "dll_manager.h"
#include <queue>

#ifdef MANAGE
    using namespace Thyroid_Det;
#endif

bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}


int main()
{
    //const string videoPath = R"(D:\company_Tenda\tend.server\thyroid_test_video\20240312_131310_202403120003.mp4)";
    const string videoPath = R"(D:\company_Tenda\27.yolov11-tensorrt\build\1036.mp4)";

    /* 反序列化engine，并且获取模型的输入尺寸 */
    yolov11 model;
    int a = model.init("bt_he_gamma_1.5_1_0.8.engine", "", "", "");

    
    cv::VideoCapture cap(videoPath);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Mat image;
    cv::Mat frame;
    int frame_count = 0;

    /* 存储每一帧的图像 */
    queue<cv::Mat> frame_q;

    vector<Detection> object;
    vector<Detection> add_object;
    int width;
    int height;
    std::vector<cv::Mat> img_batch;
    while (true)
    {
        cap >> image;

        if (image.empty() && frame_q.empty())
        {
            break;
        }

        width = image.cols;
        height = image.rows;

        unsigned char* imgptr = image.data;
        /* 后处理是cpu时用它 */
        std::vector<Detection> objects;
        /* 后处理时gpu时用它 */
        std::vector<float> DetectiontRects;
        if (!image.empty())
        {
            img_batch.push_back(image);
            /* output结构体数组 */
            auto start = std::chrono::system_clock::now();
            
            //string b = model.det(imgptr, objects, width, height, 0.0f, 0, 0, 0, 0, 0, 0);
            string b = model.cuda_det(imgptr, DetectiontRects, width, height, 0.0f, 0, 0, 0, 0, 0, 0);

            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            std::cout << b << std::endl;
            printf("cost %2.4lf ms\n", tc);
        }
        /* 可视化抓取结果 */
        //model.draw(image, objects);
        model.cuda_draw(image, DetectiontRects);
        cv::imshow("prediction", image);
        cv::waitKey(fps);
        DetectiontRects.clear();
    }
    // Release resources
    destroyAllWindows();
    cap.release();
}
