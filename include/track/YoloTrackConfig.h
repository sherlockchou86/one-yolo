#pragma once
#include <string>

namespace yolo {
    enum class YoloTrackAlgo {SORT, BYTE_TRACK};
    enum class YoloTrackLoc {CENTER, BOTTOM_CENTER, BOTTOM_CUSTOM};
    struct YoloTrackConfig {
        YoloTrackAlgo algo = YoloTrackAlgo::SORT;
        YoloTrackLoc  loc  = YoloTrackLoc::BOTTOM_CENTER;

        float loc_f        = 0.5f;        // valid only if loc == YoloTrackLoc::BOTTOM_CUSTOM
        int   max_miss     = 1;
        int   min_hits     = 3;
        float iou_thresh   = 0.8f;
    };

    std::string to_string(YoloTrackAlgo algo);
    std::string to_string(YoloTrackLoc loc);
}