#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "coco detection task using yolov11n(model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO11;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolo11n_b2.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 640;
    cfg.batch_size  = 2;
    cfg.num_classes = 80;
    cfg.names       = COCO_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;

    /* 4. create YoloTrackers using YoloTrackConfig */
    auto tracker0 = YoloTracker(t_cfg);
    tracker0.info();
    auto tracker1 = YoloTracker(t_cfg);
    tracker1.info();

    /* open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/face.mp4");
    cv::VideoCapture cap2("./vp_data/test_video/face2.mp4");
    while (cap.isOpened() && cap2.isOpened()) {
        // collect frames
        cv::Mat frame, frame2;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        if (!cap2.read(frame2)) {
            cap2.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // predict with batch mode (batch size == 2)
        auto results = model(std::vector<cv::Mat>{frame, frame2});
        
        // track results with 2 separate trackers
        tracker0(results[0]);
        tracker1(results[1]);

        // show and print
        results[0].info();           // print summary for the first result
        results[0].to_json(true);    // convert structured result to json and print
        results[0].to_csv(true);     // convert structured result to csv and print
        if (results[0].show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*640) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        results[1].info();           // print summary for the second result
        results[1].to_json(true);    // convert structured result to json and print
        results[1].to_csv(true);     // convert structured result to csv and print
        if (results[1].show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*640) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        /*
         * you can also get structured results like below:
         * auto boxes        = results[0].boxes();          // get bounding boxes in detection task
         * auto cls_ids      = results[0].cls_ids();        // get class ids in detection task
         * auto confs        = results[0].confs();          // get confidences in detection task
         * auto labels       = results[0].labels();         // get labels in detection task
         * auto track_ids    = results[0].track_ids();      // get track ids in detection task
         * auto track_points = results[0].track_points();   // get track points in detection task
         * 
         * auto boxes        = results[1].boxes();          // get bounding boxes in detection task
         * auto cls_ids      = results[1].cls_ids();        // get class ids in detection task
         * auto confs        = results[1].confs();          // get confidences in detection task
         * auto labels       = results[1].labels();         // get labels in detection task
         * auto track_ids    = results[1].track_ids();      // get track ids in detection task
         * auto track_points = results[1].track_points();   // get track points in detection task
        */
    }
}