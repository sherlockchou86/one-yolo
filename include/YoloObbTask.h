#pragma once
#include "YoloTask.h"

namespace yolo {
    class YoloObbTask: public YoloTask {
    private:
        void collect_rboxes_yolo5u_8_11(
            const cv::Mat&                output,
            int                           batch_id,
            const cv::Size&               orig_size,
            const LetterBoxInfo&          lb,
            std::vector<cv::RotatedRect>& rboxes,
            std::vector<float>&           scores,
            std::vector<int>&             cls_ids
        );
        void collect_rboxes_yolo26(
            const cv::Mat&                output,
            int                           batch_id,
            const cv::Size&               orig_size,
            const LetterBoxInfo&          lb,
            std::vector<cv::RotatedRect>& rboxes,
            std::vector<float>&           scores,
            std::vector<int>&             cls_ids
        );
    protected:
        virtual void postprocess_one(
            const std::vector<cv::Mat>& raw_outputs,
            int                         batch_id,
            cv::Size                    orig_size,
            LetterBoxInfo               lb_info,
            YoloResult&                 result
        ) override;
    public:
        YoloObbTask(const YoloConfig& cfg);
        ~YoloObbTask();
    };
}