#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "vehicle detection task using yolov5s(custom model), print structured results separately.";
    cfg.version     = YoloVersion::YOLO5;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/vehicle_yolov5s-det_c5_b1_20260129.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 1;
    cfg.num_classes = 5;
    cfg.names       = {"person", "car", "bus", "truck", "2wheel"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;
    t_cfg.iou_thresh = 0.6f;

    /* 4. create YoloTracker using YoloTrackConfig */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/vehicle_stop.mp4");
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
            false, 1.0f, DrawParam(), // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        // get structured results separately
        auto boxes        = results[0].boxes();          // get bounding boxes in detection task
        auto cls_ids      = results[0].cls_ids();        // get class ids in detection task
        auto confs        = results[0].confs();          // get confidences in detection task
        auto labels       = results[0].labels();         // get labels in detection task
        auto track_ids    = results[0].track_ids();      // get track ids in detection task
        auto track_points = results[0].track_points();   // get track points in detection task

        // print structured results separately （just like in python）
        std::cout << "bounding boxes of detected objects:" << std::endl;
        std::cout << boxes        << std::endl;
        // print bounding boxes of 5 objects:
        // [{"height":45,"width":85,"x":865,"y":431},{"height":24,"width":29,"x":580,"y":269},{"height":11,"width":17,"x":615,"y":159},{"height":21,"width":36,"x":612,"y":120},{"height":25,"width":24,"x":495,"y":150}]
        // you can also use: 
        // std::cout << yolo::to_string(boxes)        << std::endl;

        std::cout << "class ids of detected objects:" << std::endl;
        std::cout << cls_ids      << std::endl;
        // print class ids of 5 objects:
        // [1,1,1,3,3]
        // you can also use: 
        // std::cout << yolo::to_string(cls_ids)        << std::endl;

        std::cout << "confidences of detected objects:" << std::endl;
        std::cout << confs        << std::endl;
        // print confidences of 5 objects:
        // [0.9370721578598022,0.9073358178138733,0.6327939033508301,0.8769270181655884,0.835636556148529]
        // you can also use: 
        // std::cout << yolo::to_string(confs)        << std::endl;

        std::cout << "labels of detected objects:" << std::endl;
        std::cout << labels       << std::endl;
        // print labels of 5 objects:
        // ["car","car","car","truck","truck"]
        // you can also use: 
        // std::cout << yolo::to_string(labels)        << std::endl;     

        std::cout << "track ids of detected & tracked objects:" << std::endl;
        std::cout << track_ids    << std::endl;
        // print track ids of 5 objects:
        // [8,14,1,11,3]
        // you can also use: 
        // std::cout << yolo::to_string(track_ids)        << std::endl;    

        std::cout << "track points of detected & tracked objects:" << std::endl;
        std::cout << track_points << std::endl;
        // print track points of 5 objects:
        // [[{"x":1237,"y":687},{"x":1232,"y":684},...], [{"x":1097,"y":583},{"x":1091,"y":579},...], [{"x":947,"y":480},{"x":944,"y":478},...], [{"x":582,"y":203},{"x":582,"y":203},...], [{"x":609,"y":169},{"x":610,"y":169},...]]
        // you can also use: 
        // std::cout << yolo::to_string(track_points)     << std::endl;    
    }
}