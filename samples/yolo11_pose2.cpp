#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "coco pose task using yolov11n(model from ultralytics/ultralytics repo), get structured results separately.";
    cfg.version     = YoloVersion::YOLO11;
    cfg.task        = YoloTaskType::POSE;
    cfg.target_rt   = YoloTargetRT::ORT_CPU;   // choose onnxruntime
    cfg.model_path  = "./vp_data/models/ultralytics/yolo11n-pose.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 640;
    cfg.batch_size  = 1;
    cfg.num_classes = 1;
    cfg.names       = {"person"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;
    t_cfg.loc  = YoloTrackLoc::CENTER;
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

        // predict with single image
        auto result = model(frame);
        
        // track result
        tracker(result);

        // show and print
        result.info();           // print summary
        result.to_json(true);    // convert structured result to json and print
        result.to_csv(true);     // convert structured result to csv and print
        if (result.show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*640) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        // get structured results separately
        auto boxes        = result.boxes();        // get bounding boxes in pose task
        auto cls_ids      = result.cls_ids();      // get class ids in pose task
        auto confs        = result.confs();        // get confidences in pose task
        auto labels       = result.labels();       // get labels in pose task
        auto kpts         = result.kpts();         // get keypoints in pose task
        auto track_ids    = result.track_ids();    // get track ids in pose task
        auto track_points = result.track_points(); // get track points in pose task

        // print/save/send results(as string format) via kafka/udp/... to 3rd-party system
        auto str_boxes        = yolo::to_string(boxes);          // [{"height":636,"width":699,"x":403,"y":70}, ...]
        auto str_cls_ids      = yolo::to_string(cls_ids);        // [0, ...]
        auto str_confs        = yolo::to_string(confs);          // [0.9240275025367737, ...]
        auto str_labels       = yolo::to_string(labels);         // ["person", ...]
        auto str_kpts         = yolo::to_string(kpts);           // [[{"conf":0.9939126968383789,"x":614.1455078125,"y":313.71728515625},{"conf":0.9957867860794067,"x":661.4625854492188,"y":259.56353759765625},...], ...]
        auto str_track_ids    = yolo::to_string(track_ids);      // [3, ...]
        auto str_track_points = yolo::to_string(track_points);   // [[{"x":755,"y":394},{"x":755,"y":394},{"x":755,"y":393},...], ...]

        std::cout << str_boxes << std::endl;
        std::cout << str_cls_ids << std::endl;
        std::cout << str_confs << std::endl;
        std::cout << str_labels << std::endl;
        std::cout << str_kpts << std::endl;
        std::cout << str_track_ids << std::endl;
        std::cout << str_track_points << std::endl;
    }
}