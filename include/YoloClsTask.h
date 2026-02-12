#pragma once
#include "YoloTask.h"

namespace yolo {
    class YoloClsTask: public YoloTask {
    private:
        /* data */
        cv::Mat softmax(const cv::Mat& logits);
        bool is_prob_distribution(const cv::Mat& out, double eps = 1e-3);
    protected:
        virtual void postprocess_one(
            const std::vector<cv::Mat>& raw_outputs,
            int                         batch_id,
            cv::Size                    orig_size,
            LetterBoxInfo               lb_info,
            YoloResult&                 result
        ) override;
    public:
        YoloClsTask(const YoloConfig& cfg);
        ~YoloClsTask();
    };
}