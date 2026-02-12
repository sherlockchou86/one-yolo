#pragma once
#include <memory>
#include "YoloResult.h"
#include "track/YoloTrackConfig.h"
#include "track/BaseTrackAlgo.h"

namespace yolo {
    /**
     * @brief
     * wrapper class for multiple objects tracking(MOT).
     * 
     * @note
     * 1. support tracking objects from: detection, segmentation, and pose tasks.
     * 2. obb task not supported so far.
    */
    class YoloTracker final {
    private:
        YoloTrackConfig __cfg;
        std::shared_ptr<BaseTrackAlgo> __tracker = nullptr;
        // track_id -> track points
        std::map<int, std::vector<cv::Point>> __tracking_points;
        std::map<int, int>                    __tracking_miss_times;
        void preprocess(
            const YoloResult& res, 
            std::vector<cv::Rect>& boxes, 
            std::vector<std::vector<float>>& embeddings);
        void run(
            const std::vector<cv::Rect>& boxes, 
            const std::vector<std::vector<float>>& embeddings, 
            std::vector<int>& track_ids);
        void postprocess(
            const std::vector<cv::Rect>& boxes, 
            const std::vector<std::vector<float>>& embeddings, 
            const std::vector<int>& track_ids,
            YoloResult& res);
        void init();
    public:
        YoloTracker(const YoloTrackConfig& cfg);
        ~YoloTracker();

        /**
         * @brief
         * reset all status for `YoloTracker` just as initialized the first time.
        */
        void reset();

        /**
         * @brief
         * track for `YoloResult`, update its track data such as track_id, track points.
         * 
         * @param res `YoloResult` to be tracked.
         * 
        */
        void track(YoloResult& res);

        /**
         * @brief
         * track for `YoloResult` and return a new instance let original unchanged.
         * 
         * @param res `YoloResult` to be tracked.
         * 
         * @return
         * new `YoloResult` instance as same as input with tracked data(such as track_id, track points).
        */
        YoloResult track_copy(const YoloResult& res);

        /**
         * @brief
         * make `YoloTracker` callable, act as same as track(...).
         * 
         * @param res `YoloResult` to be tracked.
        */
        void operator()(YoloResult& res);

        /**
         * @brief
         * get summary for `YoloTracker`(such as config data initialized using `YoloTrackConfig`).
         * 
         * @param print print summary to console or not.
         * @return summary for `YoloTracker`.
        */
        std::string info(bool print = true);
    };
}