
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace yolo {
    /**
     * @brief
     * base class for all Yolo runtimes.
     * the main duty is running Yolo model with input tensor and get raw output tensors.
     * 
     * two stuffs to care:
     * 1. which inference backend you want to use? opencv::dnn, onnxruntime, or tenssorrt...
     * 2. which hardware platform you want to run? nvidia, ascend, or rockchip...
    */
    class YoloRuntime {
    private:
        std::string __rt_name = "default_rt";
    public:
        YoloRuntime(const std::string& rt_name);
        ~YoloRuntime();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) = 0;
        virtual std::string to_string();
    };
}