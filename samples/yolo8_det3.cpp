#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "fire/smoke detection task using yolov8m(custom model)";
    cfg.version     = YoloVersion::YOLO8;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/firesmoke_v8m_map616_nhwc_uint8_normalize_20251217.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 1;
    cfg.nchw        = false;    // channel last
    cfg.scale_f     = 1.0f;     // no normalize
    cfg.num_classes = 3;
    cfg.names       = {"fire", "smoke", "light"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/fire3.mp4");
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
            false, 1.0f, DrawParam(), // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        /*
         * you can also get structured results like below:
         * auto boxes   = results[0].boxes();    // get bounding boxes in detection task
         * auto cls_ids = results[0].cls_ids();  // get class ids in detection task
         * auto confs   = results[0].confs();    // get confidences in detection task
         * auto labels  = results[0].labels();   // get labels in detection task
        */
    }
}