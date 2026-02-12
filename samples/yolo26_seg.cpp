#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "coco segmentation task using yolov26n(model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO26;
    cfg.task        = YoloTaskType::SEG;
    cfg.target_rt   = YoloTargetRT::OVN_AUTO;
    cfg.model_path  = "./vp_data/models/ultralytics/yolo26n-seg.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 640;
    cfg.batch_size  = 1;
    cfg.num_classes = 80;
    cfg.names       = COCO_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo       = YoloTrackAlgo::SORT;
    t_cfg.iou_thresh = 0.6f;

    /* 4. create YoloTracker using YoloTrackConfig */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* 5. open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/face.mp4");
    while (cap.isOpened()) {
        // collect frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // predict with batch mode (batch size == 1)
        auto results = model(std::vector<cv::Mat>{frame});
        
        // track result
        tracker(results[0]);

        // show and print
        results[0].info();           // print summary
        results[0].to_json(true);    // convert structured result to json and print
        results[0].to_csv(true);     // convert structured result to csv and print
        if (results[0].show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*640) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        /**
         * you can also get structured results like below:
         * auto boxes        = results[0].boxes();          // get bounding boxes in segmentation task
         * auto cls_ids      = results[0].cls_ids();        // get class ids in segmentation task
         * auto confs        = results[0].confs();          // get confidences in segmentation task
         * auto labels       = results[0].labels();         // get labels in segmentation task
         * auto masks        = results[0].masks();          // get masks in segmentation task
         * auto contours     = results[0].contours();       // get contours in segmentation task
         * auto track_ids    = results[0].track_ids();      // get track ids in segmentation task
         * auto track_points = results[0].track_points();   // get track points in segmentation task
        */
    }
}