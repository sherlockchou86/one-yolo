#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "coco detection task using yolov5nu(anchor-free & model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO5U;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CPU;  // use CPU since it's a nano model
    cfg.model_path  = "./vp_data/models/ultralytics/yolov5nu.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 640;
    cfg.batch_size  = 1;
    cfg.num_classes = 80;
    cfg.names       = COCO_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* open video and predict frames in a loop */
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
        
        // show and print
        results[0].info();           // print summary
        results[0].to_json(true);    // convert structured result to json and print
        results[0].to_csv(true);     // convert structured result to csv and print
        if (results[0].show(
            false) == 27) {          // show annotated image with unblock mode, exit loop if user has pressed ESC
            break;
        }
    }
}