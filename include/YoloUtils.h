#pragma once
#include <sstream>
#include <opencv2/opencv.hpp>
#include "YoloObjs.h"

namespace yolo {
    /**
     * @brief
     * convert float value to string with precision.
    */
    std::string to_string(const float f, const int precision = 2);

    /**
     * @brief
     * convert std::vector<T> to string with array format in json.
    */
    template<typename T>
    std::string to_string(const std::vector<T>& v) {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }
    
    /**
     * @brief
     * create 48 kind of colors.
    */
    std::vector<cv::Scalar> get_colors_48();

    /**
     * @brief
     * make std::vector<T> serializable with array format in json.
    */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
        os << json(v).dump();
        return os;
    }

    /**
     * @brief
     * toolkit for drawing YoloResult.
    */
    cv::Mat draw_results(
        const cv::Mat&                  image,
        const DrawParam&                param,
        const std::vector<int>&         top5,
        const std::vector<float>&       top5_confs,
        const std::vector<std::string>& top5_labels,
        const std::vector<int>&         cls_ids,
        const std::vector<float>&       confs,
        const std::vector<std::string>& labels,
        const std::vector<cv::Rect>&    boxes,
        const std::vector<cv::RotatedRect>&           rboxes,
        const std::vector<cv::Mat>&                   masks,
        const std::vector<std::vector<cv::Point>>&    contours,
        const std::vector<std::vector<YoloKeyPoint>>& kpts,
        const std::vector<int>&                       track_ids,
        const std::vector<std::vector<cv::Point>>&    tracks);

    class YoloUtils {
    private:
        /* data */
    public:
        YoloUtils(/* args */);
        ~YoloUtils();
        cv::Mat letterbox(
            const cv::Mat& img,
            int new_w,
            int new_h,
            LetterBoxInfo& info,
            const cv::Scalar& color = cv::Scalar(114,114,114)
        );
        void class_aware_nms(
            const std::vector<cv::Rect>& boxes,
            const std::vector<float>& scores,
            const std::vector<int>& cls_ids,
            float conf_thresh,
            float nms_thresh,
            std::vector<int>& keep_indices
        );
        void class_aware_nms(
            const std::vector<cv::RotatedRect>& rboxes,
            const std::vector<float>& scores,
            const std::vector<int>& cls_ids,
            float conf_thresh,
            float nms_thresh,
            std::vector<int>& keep_indices
        );
        cv::Rect decode_box(
            float cx, float cy, float w, float h,
            const LetterBoxInfo& lb,
            const cv::Size& orig_size
        );
        YoloKeyPoint decode_keypoint(
            float x, float y, float conf,
            const LetterBoxInfo& lb,
            const cv::Size& orig_size
        );
        cv::RotatedRect decode_rbox(
            float cx, float cy, float w, float h, float angle,
            const LetterBoxInfo& lb,
            const cv::Size& orig_size
        );
    };
}