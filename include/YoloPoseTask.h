#pragma once
#include "YoloTask.h"

namespace yolo {
    class YoloPoseTask: public YoloTask {
    private:
        void collect_boxes_yolo5u_8_11(
            const cv::Mat&                          output,
            int                                     batch_id,
            const cv::Size&                         orig_size,
            const LetterBoxInfo&                    lb,
            std::vector<cv::Rect>&                  boxes,
            std::vector<float>&                     scores,
            std::vector<int>&                       cls_ids,
            std::vector<std::vector<YoloKeyPoint>>& kpts
        );
        void collect_boxes_yolo26(
            const cv::Mat&                          output,
            int                                     batch_id,
            const cv::Size&                         orig_size,
            const LetterBoxInfo&                    lb,
            std::vector<cv::Rect>&                  boxes,
            std::vector<float>&                     scores,
            std::vector<int>&                       cls_ids,
            std::vector<std::vector<YoloKeyPoint>>& kpts
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
        YoloPoseTask(const YoloConfig& cfg);
        ~YoloPoseTask();
    };
}