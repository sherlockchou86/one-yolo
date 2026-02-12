#pragma once
#include "YoloTask.h"

namespace yolo {
    class YoloSegTask: public YoloTask {
    private:
        /* data */
        cv::Mat sigmoid(const cv::Mat& x);
        cv::Mat process_mask_one(
            const cv::Mat&   protos,
            int              batch_id,
            const cv::Mat&   coeff,
            const cv::Rect&  bbox,           
            const cv::Size&  input_size,
            const cv::Point& padding,
            const float x_scale,
            const float y_scale
        );
        void collect_boxes_yolo5(
            const cv::Mat&         output0,
            const cv::Mat&         output1,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<cv::Mat>&  masks,
            std::vector<float>&    scores,
            std::vector<int>&      cls_ids
        );
        void collect_boxes_yolo5u_8_11(
            const cv::Mat&         output0,
            const cv::Mat&         output1,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<cv::Mat>&  masks,
            std::vector<float>&    scores,
            std::vector<int>&      cls_ids
        );
        void collect_boxes_yolo26(
            const cv::Mat&         output0,
            const cv::Mat&         output1,
            int                    batch_id,
            const cv::Size&        orig_size,
            const LetterBoxInfo&   lb,
            std::vector<cv::Rect>& boxes,
            std::vector<cv::Mat>&  masks,
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
        YoloSegTask(const YoloConfig& cfg);
        ~YoloSegTask();
    };
}