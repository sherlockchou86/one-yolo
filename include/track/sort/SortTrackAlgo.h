#pragma once
#include "track/BaseTrackAlgo.h"
#include "track/sort/KalmanTracker.h"
#include "track/sort/Hungarian.h"

namespace yolo {
    class SortTrackAlgo: public BaseTrackAlgo {
    private:
        typedef struct TrackingBox {
            int id;
            Rect_<float> box;
        }TrackingBox;

        std::vector<KalmanTracker> trackers;
        std::vector<cv::Rect_<float>> predictedBoxes;
        std::vector<vector<double>> iouMatrix;
        std::vector<int> assignment;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::set<int> allItems;
        std::set<int> matchedItems;
        std::vector<cv::Point> matchedPairs;
        std::vector<TrackingBox> frameTrackingResult;
        double getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    public:
        SortTrackAlgo(const YoloTrackConfig& cfg);
        ~SortTrackAlgo();
        virtual void run(
            const std::vector<cv::Rect>& boxes, 
            const std::vector<std::vector<float>>& embeddings, 
            std::vector<int>& track_ids) override;
    };
}