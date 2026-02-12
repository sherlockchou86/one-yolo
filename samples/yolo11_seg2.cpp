#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "coco segmentation task using yolov11n(model from ultralytics/ultralytics repo), get structured results separately.";
    cfg.version     = YoloVersion::YOLO11;
    cfg.task        = YoloTaskType::SEG;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolo11n-seg.onnx";
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
    t_cfg.algo = YoloTrackAlgo::SORT;

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

        // get structured results separately
        auto boxes        = results[0].boxes();         // get bounding boxes in segmentation task
        auto cls_ids      = results[0].cls_ids();       // get class ids in segmentation task
        auto confs        = results[0].confs();         // get confidences in segmentation task
        auto labels       = results[0].labels();        // get labels in segmentation task
        auto masks        = results[0].masks();         // get masks in segmentation task
        auto contours     = results[0].contours();      // get contours in segmentation task
        auto track_ids    = results[0].track_ids();     // get track ids in segmentation task
        auto track_points = results[0].track_points();  // get track points in segmentation task

        // print/save/send results(as string format) via kafka/udp/... to 3rd-party system
        auto str_boxes        = yolo::to_string(boxes);          // [{"height":642,"width":655,"x":196,"y":68},{"height":222,"width":100,"x":168,"y":475}]
        auto str_cls_ids      = yolo::to_string(cls_ids);        // [0,56]
        auto str_confs        = yolo::to_string(confs);          // [0.9387140870094299,0.5151318311691284]
        auto str_labels       = yolo::to_string(labels);         // ["person","chair"]
        auto str_contours     = yolo::to_string(contours);       // [[{"x":1088,"y":634},{"x":1082,"y":627},...], [{"x":739,"y":676},{"x":739,"y":600},{"x":740,"y":599},...]]
        auto str_track_ids    = yolo::to_string(track_ids);      // [1,2]
        auto str_track_points = yolo::to_string(track_points);   // [[{"x":755,"y":394},{"x":755,"y":394},{"x":755,"y":393},...], [{"x":518,"y":709},{"x":520,"y":709},...]]

        std::cout << str_boxes << std::endl;
        std::cout << str_cls_ids << std::endl;
        std::cout << str_confs << std::endl;
        std::cout << str_labels << std::endl;
        std::cout << str_contours << std::endl;
        std::cout << str_track_ids << std::endl;
        std::cout << str_track_points << std::endl;
    }
}