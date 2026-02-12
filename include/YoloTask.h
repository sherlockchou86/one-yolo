#pragma once
#include <memory>
#include "YoloConfig.h"
#include "YoloResult.h"
#include "YoloRuntime.h"
#include "YoloUtils.h"

namespace yolo {
    class YoloTask {
    private:
    protected:
        YoloConfig                   _cfg;
        std::vector<cv::Size>        _orig_sizes;
        std::vector<LetterBoxInfo>   _letterbox_infos;
        std::vector<cv::Mat>         _input_images;
        std::shared_ptr<YoloRuntime> _rt = nullptr;

        /**
         * @brief
         * preprocess for input images and return a 4D matrix to be used later.
         * 
         * @param images input images with batch mode.
         * @return a 4D matrix.
        */
        virtual cv::Mat preprocess(const std::vector<cv::Mat>& images);

        /**
         * @brief
         * inference based on different Yolo runtimes.
         * 
         * @param blob a 4D matrix to be sent to Yolo network.
         * @return raw output matrixs from Yolo network, support multi-heads.
        */
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob);

        /**
         * @brief
         * postprocess for raw outputs, extract structured results and create `YoloResult`s.
         * 
         * @param raw_outputs raw output matrixs from Yolo network.
         * @param batch_size batch size to process.
         * 
         * @return a list of `YoloResult`, has the same size of input images.
        */
        virtual std::vector<yolo::YoloResult> postprocess(const std::vector<cv::Mat>& raw_outputs, int batch_size);

        /**
         * @brief
         * preprocess for input images one by one.
         * 
         * @param image single image.
         * @return preprocessed image (scale/padding).
        */
        virtual cv::Mat preprocess_one(const cv::Mat& image);

        /**
         * @brief
         * postprocess for raw outputs one by one, extract structured result and fill `YoloResult`.
         * 
         * @param raw_outputs raw output matrixs from Yolo network.
         * @param batch_id bacth index to be processed.
         * @param orig_size original image size.
         * @param lb_info letterbox info (scale/padding).
         * @param result `YoloResult` to be filled.
         * 
         * @note
         * MUST override in child classes.
        */
        virtual void postprocess_one(
            const std::vector<cv::Mat>& raw_outputs,
            int                         batch_id,
            cv::Size                    orig_size,
            LetterBoxInfo               lb_info,
            YoloResult&                 result
        ) = 0;
    public:
        YoloTask(const YoloConfig& cfg);
        ~YoloTask();

        /**
         * @brief
         * run task with input images.
         * 
         * @param images input images with batch mode.
         * @return a list of `YoloResult`, has the same size of input images.
        */
        std::vector<yolo::YoloResult> run(const std::vector<cv::Mat>& images);

        /**
         * @brief
         * make `YoloTask` callable, act as same as `run(...)`.
         * 
         * @param images input images with batch mode.
         * @return a list of `YoloResult`, has the same size of input images.
         * 
        */
        std::vector<yolo::YoloResult> operator()(const std::vector<cv::Mat>& images);
    };
}