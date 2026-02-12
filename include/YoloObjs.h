#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace yolo {
    struct YoloClsObj {
        int         cls_id;                       // class id for classification task
        float       conf;                         // confidence for classification task
        std::string label;                        // label for classification task
    };

    struct YoloDetObj {
        cv::Rect               box;               // bounding box for detected object in detection task
        int                    cls_id;            // class id for detected object in detection task
        float                  conf;              // confidence for detected object in detection task
        std::string            label;             // label for detected object in detection task
        int                    track_id = -1;     // track id for detected object in detection task (-1 means no tracked yet)
        std::vector<cv::Point> track_points;      // track points for detected object in detection task (empty means no tracked yet)
    };

    struct YoloSegObj {
        cv::Rect               box;               // bounding box for detected object in segmentation task
        int                    cls_id;            // class id for detected object in segmentation task
        float                  conf;              // confidence for detected object in segmentation task
        std::string            label;             // label for detected object in segmentation task
        cv::Mat                mask;              // mask for detected object in segmentation task(it's a local & binary mask, has the same size as bounding box, has 2 values: 0 or 255)
        std::vector<cv::Point> contour;           // contour for detected object in segmentation task
        int                    track_id = -1;     // track id for detected object in segmentation task (-1 means no tracked yet)
        std::vector<cv::Point> track_points;      // track points for detected object in segmentation task (empty means no tracked yet)
    };

    struct YoloKeyPoint {
        float x;                                  // x value for keypoint in pose task
        float y;                                  // y value for keypoint in pose task
        float conf;                               // confidence for keypoint in pose task
    };

    struct YoloPoseObj {
        cv::Rect                  box;            // bounding box for detected object in pose task
        int                       cls_id;         // class id for detected object in pose task
        float                     conf;           // confidence for detected object in pose task
        std::string               label;          // label for detected object in pose task
        std::vector<YoloKeyPoint> keypoints;      // keypoints for detected object in pose task
        int                       track_id = -1;  // track id for detected object in pose task (-1 means no tracked yet)
        std::vector<cv::Point>    track_points;   // track points for detected object in pose task (empty means no tracked yet)
    };

    struct YoloObbObj {
        cv::RotatedRect rbox;                     // rotated box for detected object in obb task
        int             cls_id;                   // class id for detected object in obb task
        float           conf;                     // confidence for detected object in obb task
        std::string     label;                    // label for detected object in obb task
    };

    struct LetterBoxInfo {
        float scale;
        int   pad_w;
        int   pad_h;
    };


    using KptPairs      = std::vector<std::pair<int, int>>;
    using KptPairColors = std::vector<cv::Scalar>;
    struct DrawParam {
        bool top1_only      = false;              // draw top1 only or not for classification task
        bool color_by_class = true;               // choose background color by class or by instance

        bool cls_ids        = false;              // draw class ids or not
        bool confs          = true;               // draw confidences or not
        bool labels         = true;               // draw labels or not
        bool boxes          = true;               // draw boxes or not
        bool rboxes         = true;               // draw rotated boxes or not
        bool masks          = true;               // draw masks or not
        bool kpts           = true;               // draw keypointss or not
        bool track_ids      = true;               // draw track ids or not
        bool tracks         = true;               // draw tracks or not

        int box_line_width  = 2;                  // line width of box, <=0 means no drawing
        int rbox_line_width = 2;                  // line width of rotated box,  <=0 means no drawing
        int mask_line_width = 1;                  // line width for contour of mask, <=0 means no drawing
        float mask_alpha    = 0.5f;               // alpha of mask area, 0 means fully transparent, 1 means solid fill, and values in between represent semi-transparency
        int kpt_line_width  = 2;                  // line width of keypoint pair, <=0 means no link at all
        int kpt_radius      = 5;                  // radius of keypoint, <=0 means no drawing
        KptPairs kpt_pairs  =                     // indice pairs to link keypoints(show skeleton for example), empty means no link at all. 
                                                  // should refer to the num_kpts in `YoloConfig` (coco human pose 19 pairs by default).
                              {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4},             // head part
                               {0, 5}, {0, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},    // hand part
                               {5, 11},  {5, 6},   {6, 12},                        // body part
                               {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};  // leg part
        KptPairColors 
        kpt_pair_colors     = {cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 250),                                           // color of head part(BGR)
                               cv::Scalar(220,  0, 250), cv::Scalar(220,  0, 250), cv::Scalar(220,  0, 250), cv::Scalar(220,  0, 250), cv::Scalar(220,  0, 250), cv::Scalar(220,  0, 250),  // color of hand part(BGR)
                               cv::Scalar(110,  250, 0), cv::Scalar(110,  250, 0), cv::Scalar(110,  250, 0),                                                                                // color of body part(BGR)
                               cv::Scalar(0, 255, 250), cv::Scalar(0, 255, 250), cv::Scalar(0, 255, 250), cv::Scalar(0, 255, 250), cv::Scalar(0, 255, 250)};                                // color of leg part(BGR)
        int track_line_width = 1;                             // line width of track, <=0 means no drawing
        int loc_radius       = 4;                             // radius of located point, <=0 means no drawing 
        cv::Scalar 
        font_color           = cv::Scalar(255, 255, 255);     // color for drawing text(BGR)
        float font_scale     = 0.5f;                          // font scale for drawing text
        int   font_face      = cv::FONT_HERSHEY_SIMPLEX;      // font face for drawing text
        int   font_thickness = 1;                             // font thickness for drawing text
        float scale          = 1.0f;                          // resize canvas or not
    };

    void to_json(json& j, const YoloClsObj& obj);
    void to_json(json& j, const YoloDetObj& obj);
    void to_json(json& j, const YoloSegObj& obj);
    void to_json(json& j, const YoloPoseObj& obj);
    void to_json(json& j, const YoloObbObj& obj);
    void to_json(json& j, const YoloKeyPoint& obj);
}

namespace cv {
    void to_json(json& j, const cv::Point& obj);
    void to_json(json& j, const cv::Rect& obj);
    void to_json(json& j, const cv::RotatedRect& obj);
}