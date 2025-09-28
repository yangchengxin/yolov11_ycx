#include "yolov11.h"
#include "cuda_utils.h"
//#include "logging.h"
#include "macros.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <NvOnnxParser.h>

#define warmup true

yolov11::yolov11()
{
}

yolov11::~yolov11()
{
    // Release stream and buffers
    /* ------------- cuda流同步之后释放 -------------- */
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;

    // release postprocess data
    free(CpuOutputRects);
    free(CpuOutputCount);
    CUDA_CHECK(cudaFree(GpuOutputRects));
    CUDA_CHECK(cudaFree(GpuOutputCount));

    // Destroy the engine
    cuda_preprocess_destroy();

    delete context;
    delete engine;
    delete runtime;
}


int yolov11::init(const char* det_model, const char* cls_model, const char* cls2_model, const char* machine_type)
{
    // Read the engine file
    ifstream engineStream(det_model, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    batchsize = engine->getBindingDimensions(1).d[0];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[batchsize * detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], batchsize * 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], batchsize * detection_attribute_size * num_detections * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&GpuOutputCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&GpuOutputRects, sizeof(DetectRect) * det_nums));
    CpuOutputCount = (int*)malloc(sizeof(int));
    CpuOutputRects = (DetectRect*)malloc(sizeof(DetectRect) * det_nums);

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));


    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }

    /* -------- 根据engine获取输入的尺度 --------- */
#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif

    return 1;
}

cv::Mat yolov11::gammaCorrection(const cv::Mat image, double gamma) {
    if (image.empty() || gamma <= 0) {
        return cv::Mat();
    }
    // 归一化到[0,1]并转换为浮点型
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // 应用Gamma校正
    cv::Mat corrected;
    cv::pow(normalized, gamma, corrected);

    // 将数值缩放回[0,255]并转换为8位无符号整型
    corrected.convertTo(corrected, CV_8U, 255.0);

    return corrected;
}

void yolov11::preprocess(cv::Mat image)
{
    cv::Mat dst(image.rows, image.cols, CV_8UC3);
    cuda_gamma_merge(image, dst);
    cuda_preprocess(dst.ptr(), dst.cols, dst.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    dst.release();
}

void yolov11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
    
}

void yolov11::postprocess(vector<Detection>& output, std::string& result, int inputx, int inputy, int inputw, int inputh)
{
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
    for(int i = 0;i < det_output.cols; ++i)
    {
        const cv::Mat class_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(class_scores, nullptr, &score, nullptr, &class_id_point);

        if(score > conf_threshold)
        {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);

            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    /* ---------------------------- nms -----------------------------*/
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
    for(int i = 0;i < nms_result.size();++i)
    {
        Detection detections;
        int idx = nms_result[i];
        detections.class_id = class_ids[idx];
        detections.conf = confidences[idx];
        detections.bbox = boxes[idx];
        output.push_back(detections);

        result += std::to_string(detections.bbox.x + inputx) + "," + std::to_string(detections.bbox.y + inputy) + "," 
            + std::to_string(detections.bbox.width) + "," + std::to_string(detections.bbox.height) + "," + std::to_string(detections.conf)
            + "," + "0" + "," + BenignNames[detections.class_id] + ";";
    }
}

void yolov11::cuda_postprocess(cv::Mat src, std::string& result, std::vector<float>& DetectiontRects, int width, int height, int inputx, int inputy)
{
    cudaMemsetAsync(GpuOutputCount, 0, 4, stream);
    GetNmsBeforeBoxes(gpu_buffers[1], num_detections, BenignNames.size(), conf_threshold, det_nums, GpuOutputRects, GpuOutputCount, stream);
    cudaMemcpyAsync(CpuOutputCount, GpuOutputCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(CpuOutputRects, GpuOutputRects, sizeof(DetectRect) * det_nums, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    GetConvDetectionResult(CpuOutputRects, CpuOutputCount, DetectiontRects, input_w, input_h, nms_threshold, result);

    /* ---------------- 输出处理 ----------------- */
    const float ratio_h = input_h / (float)height;
    const float ratio_w = input_w / (float)width;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int class_id = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2]);
        int ymin = int(DetectiontRects[i + 3]);
        int xmax = int(DetectiontRects[i + 4]);
        int ymax = int(DetectiontRects[i + 5]);

        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            xmin = xmin / ratio_w;
            ymin = (ymin - (input_h - ratio_w * height) / 2) / ratio_w;
            xmax = xmax / ratio_w;
            ymax = (ymax - (input_h - ratio_w * height) / 2) / ratio_w;
        }
        else
        {
            xmin = (xmin - (input_w - ratio_h * width) / 2) / ratio_h;
            ymin = ymin / ratio_h;
            xmax = (xmax - (input_w - ratio_h * width) / 2) / ratio_h;
            ymax = ymax / ratio_h;
        }

        // 如果检测出来是low的结节，做一下判断是否是middlehigh
        if (class_id == 1)
        {
            // 边界检查
            xmin = std::max(0, xmin);
            ymin = std::max(0, ymin);
            xmax = std::min(xmax, src.cols);
            ymax = std::min(ymax, src.rows);
            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
            cv::Mat src_roi = src(roi);
            cv::Mat gray_src_roi;
            cv::cvtColor(src_roi, gray_src_roi, COLOR_BGR2GRAY);
            //cv::GaussianBlur(gray_src_roi, gray_src_roi, cv::Size(5,5), 0);
            
            /*cv::Scalar meanVals = cv::mean(gray_src_roi);
            int meanVal = static_cast<int>(meanVals[0]);
            cv::Mat binary_src_roi;
            cv::threshold(
                gray_src_roi,
                binary_src_roi,
                meanVal,
                255,
                cv::THRESH_BINARY
            );*/

            int pixelCount[10] = {0};
            for (int y = 0; y < gray_src_roi.rows; y++)
            {
                for (int x = 0; x < gray_src_roi.cols; x++)
                {
                    int pixelValue = gray_src_roi.at<uchar>(y, x);
                    int index = pixelValue / 26;
                    pixelCount[index]++;
                }
            }

            for (int i = 0; i < 10; i++)
            {
                std::cout << "range [ " << i * 26 << "," << (i + 1) * 26 - 1 << "]:" << pixelCount[i] << " pixels " << std::endl;
            }
            int maxPixelIndex = 0;
            int maxPixelCount = 0;
            for (int i = 0; i < 10; i++)
            {
                if (pixelCount[i] > maxPixelCount)
                {
                    maxPixelIndex = i;
                    maxPixelCount = pixelCount[i];
                }
            }
            if (maxPixelIndex < 3)
            {
                DetectiontRects[i + 0] = 1.0f;
            }
            else
            {
                DetectiontRects[i + 0] = 10.0f;
            }
        }

        result += std::to_string(xmin + inputx) + "," + std::to_string(ymin + inputy) + ","
            + std::to_string(xmax - xmin) + "," + std::to_string(ymax - ymin) + "," + std::to_string(conf)
            + "," + "0" + "," + BenignNames[int(DetectiontRects[i + 0])] + ";";
    }
}

std::string yolov11::cuda_det(unsigned char* imgbuf, std::vector<float>& DetectiontRects, int width, int height, float thresh, int inputx, int inputy, int inputw, int inputh, int lingmindu, int dongjingtai)
{
    /***************************************************************
     * @brief      检测接口
     * @author     chengxin.yang
     * @param      imgbuf                   输入图像的指针
                   DetectiontRects          检测结果的结构体数组
                   width                    原图的宽度
                   height                   原图的高度
                   inputx                   工作区域的xmin
                   inputy                   工作区域的ymin
                   
     * @version    版本号
     * @date       2025.5.19
     **************************************************************/

    std::string result = "";

    cv::Mat src = cv::Mat(height, width, CV_8UC3, imgbuf);

    /* ----------- 截取工作区域 ------------ */
    // cv::Mat work_src = src(cv::Rect(inputx, inputy, inputw, inputh));

    /* ---------------- cuda前处理 --------------- */
    preprocess(src);
    infer();
    /* ---------------- cuda后处理 --------------- */
    cuda_postprocess(src, result, DetectiontRects, width, height, inputx, inputy);

    /* ----------- 释放资源 ------------ */
    src.release();
    return result;
}

std::string yolov11::det(unsigned char* imgbuf, std::vector<Detection>& output, int width, int height, float thresh, int inputx, int inputy, int inputw, int inputh, int lingmindu, int dongjingtai)
{

    std::string result = "";

    cv::Mat src = cv::Mat(height, width, CV_8UC3, imgbuf);

    /* ----------- 截取工作区域 ------------ */
    // cv::Mat work_src = src(cv::Rect(inputx, inputy, inputw, inputh));

    /* ---------------- cuda前处理 --------------- */
    preprocess(src);
    infer();
    /* ---------------- cpu后处理 ---------------- */
    postprocess(output, result, inputx, inputy, inputw, inputh);

    /* ----------- 释放资源 ------------ */
    src.release();
    return result;
}

void yolov11::cuda_draw(cv::Mat& image, std::vector<float>& DetectiontRects)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < DetectiontRects.size(); i+=6)
    {
        int class_id = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2]);
        int ymin = int(DetectiontRects[i + 3]);
        int xmax = int(DetectiontRects[i + 4]);
        int ymax = int(DetectiontRects[i + 5]);

        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            xmin = xmin / ratio_w;
            ymin = (ymin - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            xmax = xmax / ratio_w;
            ymax = (ymax - (input_h - ratio_w * image.rows) / 2) / ratio_w;
        }
        else
        {
            xmin = (xmin - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            ymin = ymin / ratio_h;
            xmax = (xmax - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            ymax = ymax / ratio_h;
        }

        rectangle(image, Point(xmin, ymin), Point(xmax, ymax), color, 3);

        // Detection box text
        string class_string = BenignNames[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(xmin, ymin - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        cv::putText(image, class_string, Point(xmin + 5, ymin - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}

void yolov11::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = BenignNames[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        //rectangle(image, text_rect, color, FILLED);
        cv::putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
