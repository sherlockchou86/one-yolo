#pragma once
#include "YoloTask.h"

namespace yolo {
    class YoloDetTask: public YoloTask {
    private:
        /* data */
        void collect_boxes_yolo5(
            const cv::Mat&         output,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<float>&    scores,
            std::vector<int>&      cls_ids
        );
        void collect_boxes_yolo5u_8_11(
            const cv::Mat&         output,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<float>&    scores,
            std::vector<int>&      cls_ids
        );
        void collect_boxes_yolo26(
            const cv::Mat&         output,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<float>&    scores,
            std::vector<int>&      cls_ids
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
        YoloDetTask(const YoloConfig& cfg);
        ~YoloDetTask();
    };
}