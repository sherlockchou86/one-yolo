
#pragma once
#include "YoloConfig.h"
#include "YoloResult.h"
#include "YoloTask.h"

namespace yolo {
    /**
     * @brief
     * wrapper class for unified interface accessing Yolo powered by ultralytics (ultralytics/ultralytics repo from github). 
     * 
     * support:
     * 1. yolo:  yolov5/yolov5u/yolov8/yolov11/yolov26
     * 2. tasks: classification/detection/segmentation/pose/obb
    */
    class Yolo final {
    private:
        YoloConfig                __cfg;
        std::shared_ptr<YoloTask> __task = nullptr;
    public:
        Yolo(const YoloConfig& cfg);
        ~Yolo();

        /**
         * @brief
         * predict with single image mode.
         * 
         * @param image image to be predicted.
         * @return structured result, single `YoloResult` object.
        */
        YoloResult predict(const cv::Mat& image);

        /**
         * @brief
         * predict with batch mode.
         * 
         * @param images a list of images to be predicted with batch mode.
         * @return structured results, a list of `YoloResult` objects.
        */
        std::vector<YoloResult> predict(const std::vector<cv::Mat>& images);

        /**
         * @brief
         * make `Yolo` callable, act as same as predict(...) with single image mode.
         * 
         * @param image image to be predicted.
         * @return structured result, single `YoloResult` object.
        */
        YoloResult operator()(const cv::Mat& image);

        /**
         * @brief
         * make `Yolo` callable, act as same as predict(...) with batch mode.
         * 
         * @param images a list of images to be predicted with batch mode.
         * @return structured results, a list of `YoloResult` objects.
        */
        std::vector<YoloResult> operator()(const std::vector<cv::Mat>& images);

        /**
         * @brief
         * get summary for `Yolo`(such as config data initialized using `YoloConfig`).
         * 
         * @param print print summary to console or not.
         * @return summary for `Yolo`.
        */
        std::string info(bool print = true);
    };
}